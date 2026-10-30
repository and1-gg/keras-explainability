import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, Dense, Lambda, LayerNormalization

from typing import Union

from .utils import fuse_batchnorm, remove_activation
from .utils.reference import ReferenceActivations, ReferenceShift, \
                             find_linear_dense_head, restructure_for_reference
from ..layers import get_lrp_layer, PoolingLRPLayer, StandardLRPLayer
from ..utils import infer_graph_structure, topological_sort, tensor_key
from ..utils.strategies import LRPStrategy


class LayerwiseRelevancePropagator(Model):
    """LRP as a Keras model mapping inputs to input relevances.

    ``reference_value`` (optional) switches to the *regression* variant of
    Letzgus et al. (2022, arXiv:2112.11407, Sec. IV-B): the network top is
    restructured (not retrained) so that ``g(x) = f(x) - reference_value``
    is explained instead of ``f(x)``. The value must be given in the units of
    the network output (``model.layers[layer]``). ``reference_asymmetry`` is
    the factor ``kappa`` of the flooding rule (0: one-sided flooding, 0.25:
    variant of Sec. VI-A of the paper). Without ``reference_value`` the
    behaviour is unchanged (implicit reference value 0).

    With a reference value the attributes ``restructured_model`` (Keras model
    ``x -> f(x) - reference_value``) and ``reference_activations_model``
    (``x -> a_ref`` of the last hidden layer) are available.
    """

    def __init__(self, model: Model, *, layer: Union[int, str], idx: int,
                 include_prediction: bool = False,
                 epsilon: float = None, gamma: float = None,
                 alpha: float = None, beta: float = None,
                 strategy: LRPStrategy = None,
                 reference_value: float = None,
                 reference_asymmetry: float = 0.0,
                 name: str = 'LRP'):

        if reference_value is not None:
            # Range-clipping (restrict_range) is not a Dense head. Explain the
            # linear Dense output, which is the quantity the paper's
            # restructuring is defined on.
            try:
                target = model.get_layer(layer) if isinstance(layer, str) \
                    else model.layers[layer]
            except (KeyError, IndexError, ValueError, TypeError):
                target = None
            if not isinstance(target, Dense):
                layer = find_linear_dense_head(model)

        if isinstance(layer, str):
            original_output = model.get_layer(layer).output
        else:
            original_output = model.layers[layer].output
        model = Model(model.input, original_output)
        model = remove_activation(model, ['sigmoid', 'softmax'])
        model = fuse_batchnorm(model)

        if idx < 0:
            raise NotImplementedError('Negative node indexes (e.g. idx) not '
                                      'implemented')

        reference_pre_layer = None
        restructured_model = None
        reference_activations_model = None

        if reference_value is not None:
            reference_value = float(reference_value)
            restructured_model, reference_shift, reference_pre_layer = \
                restructure_for_reference(
                    model,
                    idx=idx,
                    reference_value=reference_value,
                    asymmetry=reference_asymmetry,
                    name=name,
                )
            a_ref = ReferenceActivations(
                reference_shift,
                name=f'{name}_reference_activations'
            )(reference_pre_layer.output)
            reference_activations_model = Model(model.input, a_ref)
            model = restructured_model

        if strategy is not None:
            assert epsilon is None, \
                'Unable to instantiate LRP with both epsilon and strategy'

            assert gamma is None, \
                'Unable to instantiate LRP with both gamma and strategy'

            assert alpha is None, \
                'Unable to instantiate LRP with both alpha/beta and strategy'

        dependencies = infer_graph_structure(model)
        order = topological_sort(dependencies)
        order = order[::-1]
        output = model.output
        output_shape = (1,) + output.shape[1:]

        if len(output.shape) != 2:
            raise NotImplementedError('Unable to handle non-flat target layers')

        indexes = tf.range(output.shape[-1], name=f'{name}_output_indexes')
        mask = indexes == idx
        zeros = tf.zeros(output_shape, name=f'{name}_output_zeros')

        masked_output = Lambda(
            lambda params: tf.where(*params, name=f'{name}_output_masked'),
            name=f'{name}_output_mask_lambda',
            output_shape=output_shape
        )([mask, output, zeros])

        relevances = {
            tensor_key(output): masked_output
        }

        layers = [model.layers[i] for i in order]

        i = 0
        while i < len(layers):
            layer = layers[i]
            kwargs = {
                'name': f'{name}_{i}',
                'epsilon': epsilon,
                'gamma': gamma,
                'alpha': alpha,
                'beta':beta
            }

            extra_inputs = []

            if isinstance(layer, ReferenceShift):
                # Reference-value LRP: skip the shift (sample dependent
                # offsets) and explain the triplicated linear layer. Biases
                # including those offsets are ignored so LRP-0 cannot undo
                # the flooding (Letzgus et al., Sec. VI-A: biases ignored
                # in fully connected LRP). The reference is already encoded
                # in the triplicated ReLU activations.
                pre = layers[i + 1]
                if pre is not reference_pre_layer:
                    raise RuntimeError('ReferenceShift must directly follow '
                                       'the triplicated pre-activation layer')
                inputs = pre.input
                outputs = layer.output
                R = relevances[tensor_key(outputs)]
                kwargs['ignore_bias'] = True
                layer = pre
                i += 2
            elif isinstance(layer, LayerNormalization):
                if isinstance(layers[i + 1], Dense):
                    inputs = layers[i + 1].input
                    outputs = layer.output
                    R = relevances[tensor_key(outputs)]
                    kwargs['norm'] = layer
                    layer = layers[i + 1]
                else:
                    raise NotImplementedError('Unable to handle '
                                              'LayerNormalization preceded by '
                                              'something other than a Dense '
                                              'layer.')
                i += 2
            else:
                inputs = layer.input
                outputs = layer.output
                R = relevances[tensor_key(outputs)]
                i += 1
            relevance = get_lrp_layer(
                layer,
                **kwargs
            )([inputs, R] + extra_inputs)

            if not isinstance(relevance, list):
                inputs = [inputs]
                relevance = [relevance]

            assert len(relevance) == len(inputs), \
                (f'Layer {layer} has {len(inputs)} inputs but '
                 f'{len(relevances)} relevances')

            for j in range(len(inputs)):
                if isinstance(inputs[j], list):
                    if len(inputs[j]) != 0:
                        raise ValueError('Inputs is a nested list where the '
                                         'inner list is not empty')
                    if not isinstance(layer, InputLayer):
                        raise ValueError('Inputs is a nested list but layer '
                                         f'is not an InputLayer ({layer})')
                    key = ("layer", layer.name)
                else:
                    key = tensor_key(inputs[j])

                relevances[key] = relevance[j]

        # Keras 3: model.inputs is always a list. Unwrap the common
        # single-input case so callers get a tensor, not [tensor].
        inputs = model.inputs
        input_list = inputs if isinstance(inputs, list) else [inputs]
        outputs = [relevances[tensor_key(t)] for t in input_list]

        if include_prediction:
            outputs = [original_output] + outputs
        elif len(outputs) == 1:
            outputs = outputs[0]

        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]

        super().__init__(inputs, outputs)

        self.reference_value = reference_value
        self.reference_asymmetry = float(reference_asymmetry)
        self.restructured_model = restructured_model
        self.reference_activations_model = reference_activations_model

        if strategy is not None:
            standard_lrp_layers = []
            pooling_lrp_layers = []

            for i in range(len(self.layers)):
                if isinstance(self.layers[i], StandardLRPLayer):
                    standard_lrp_layers.append(self.layers[i])
                elif isinstance(self.layers[i], PoolingLRPLayer):
                    pooling_lrp_layers.append(self.layers[i])

            if strategy.layers is not None:
                assert len(strategy.layers) == len(standard_lrp_layers), \
                    ('Unable to instantiate LRP with strategy that does not have '
                    'a configuration for each standard LRP layer')

                configurations = strategy.layers[::-1]

                for i in range(len(configurations)):
                    for variable in configurations[i]:
                        assert hasattr(standard_lrp_layers[i], variable), \
                            ('LRPStrategy contains invalid configuration variable '
                            f'{variable} for layer '
                            f'{standard_lrp_layers[i].__class__.__name__}')
                        setattr(standard_lrp_layers[i], variable,
                                configurations[i][variable])

            if strategy.pooling is not None:
                assert len(strategy.pooling) == len(pooling_lrp_layers), \
                    ('Unable to instantiate LRP with strategy that does not have '
                    'a configuration for each standard LRP layer')

                configurations = strategy.pooling[::-1]

                for i in range(len(configurations)):
                    pooling_lrp_layers[i].strategy = configurations[i]['strategy']

LRP = LayerwiseRelevancePropagator

