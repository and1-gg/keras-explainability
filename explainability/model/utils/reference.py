"""Reference-value restructuring for LRP on regression models.

Implements the *restructuring* approach of

    S. Letzgus, P. Wagner, J. Lederer, W. Samek, K.-R. Müller, G. Montavon,
    "Toward Explainable AI for Regression Models", IEEE SPM 2022
    (arXiv:2112.11407), Section IV-B.

Instead of explaining ``f(x)`` (implicit reference value 0), the network is
rewritten -- without retraining -- so that it outputs ``g(x) = f(x) - y_ref``
and LRP can be applied unchanged to the rewritten network.

Assumed top-layer structure (after BatchNorm fusion)::

    pre (Conv2D/Conv3D/Dense, linear) -> ReLU -> [Dropout | GlobalAvgPool |
    Flatten]* -> Dense (target layer)

The two restructuring steps of the paper are:

* **Step 1 (linear layer).**  With ``y = sum_i a_i c_i + b`` (``c`` are the
  effective output weights per ReLU unit, e.g. ``w_j / V`` behind a global
  average pooling over ``V`` positions) a reference activation ``a_ref`` with
  ``sum_i a_ref_i c_i = y_ref`` is searched (eq. 5 in the paper).  Then
  ``y - y_ref = sum_i (a_i - a_ref_i) c_i + b``.  ``a_ref`` is found with the
  *flooding* rule ``a_ref = (a - t * d)_+`` where ``t`` is a scalar chosen per
  sample such that the equation holds.  ``d`` is ``+1`` for units whose
  evidence has to be removed to reach ``y_ref`` and ``-asymmetry`` for the
  units of the opposite sign (``asymmetry = 0`` reproduces the one-sided
  version of the rule, ``asymmetry = 0.25`` corresponds to the variant used in
  Section VI-A of the paper).  With this choice the equation is monotone in
  ``t`` and solved by bisection.
* **Step 2 (ReLU layer).**  The shifted ReLU ``relu(z) - a_ref`` is written as
  ``relu(z - a_ref) + relu(-z) - relu(-z + a_ref)``.  Every unit of the
  pre-activation layer is therefore triplicated: incoming weights are
  multiplied by ``(1, -1, -1)``, the offsets ``(-a_ref, 0, +a_ref)`` are added
  to the biases and the outgoing weights are multiplied by ``(1, 1, -1)``.

Because ``a_ref`` depends on the sample, the offsets are not Keras weights
but are computed inside :class:`ReferenceShift` from the pre-activations.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    GlobalAveragePooling3D,
    Layer,
    ReLU,
)

from ...utils import tensor_key

_GLOBAL_AVG_POOLS = (GlobalAveragePooling2D, GlobalAveragePooling3D)
_PASSTHROUGH = (Dropout, Flatten) + _GLOBAL_AVG_POOLS
_PRE_LAYERS = (Conv2D, Conv3D, Dense)


def _is_plain_relu(layer: Layer) -> bool:
    if isinstance(layer, ReLU):
        return (
            layer.max_value is None
            and float(layer.negative_slope) == 0.0
            and float(layer.threshold) == 0.0
        )
    return isinstance(layer, Activation) and \
        getattr(layer.activation, '__name__', '') == 'relu'


def _is_linear(layer: Layer) -> bool:
    activation = getattr(layer, 'activation', None)
    return activation is None or \
        getattr(activation, '__name__', '') == 'linear'


def _producer(model: Model, tensor) -> Layer:
    key = tensor_key(tensor)
    for layer in model.layers:
        outputs = layer.output
        outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        if any(tensor_key(o) == key for o in outputs):
            return layer
    raise ValueError('Unable to find the layer producing the tensor '
                     f'{tensor}')


def _inbound_layers(layer: Layer) -> List[Layer]:
    found: List[Layer] = []
    for node in getattr(layer, '_inbound_nodes', []) or []:
        inbound = getattr(node, 'inbound_layers', None)
        if inbound is None:
            continue
        if not isinstance(inbound, (list, tuple)):
            inbound = [inbound]
        found.extend(src for src in inbound if src is not None)
    return found


def _strip_range_clip(model: Model) -> Model:
    """Drop a trailing ``restrict_range`` head (clipped ReLU + Add offset).

    Regression-SFCN models clip the Dense output to ``[min, max]``. LRP with a
    reference value has to explain the linear Dense output ``f(x)``, so the
    clip is removed without changing weights.
    """
    try:
        out = _producer(model, model.output)
    except ValueError:
        return model
    if not isinstance(out, Add):
        return model

    inbound = _inbound_layers(out)
    relus = [
        layer for layer in inbound
        if isinstance(layer, ReLU) and layer.max_value is not None
    ]
    if len(relus) != 1:
        return model

    try:
        pre = _producer(model, relus[0].input)
    except ValueError:
        return model
    if not isinstance(pre, Dense) or not _is_linear(pre):
        return model

    return Model(model.input, pre.output, name=model.name)


def find_linear_dense_head(model: Model) -> int:
    """Index of the linear Dense regression head in ``model.layers``.

    Skips a trailing range-clip (``restrict_relu`` / ``restrict_add``) so that
    LRP can target the same tensor that ``restructure_for_reference`` explains.
    """
    stripped = _strip_range_clip(model)
    dense = _producer(stripped, stripped.output)
    if isinstance(dense, Dense) and _is_linear(dense):
        for i, layer in enumerate(model.layers):
            if layer is dense or layer.name == dense.name:
                return i

    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        if isinstance(layer, Dense) and _is_linear(layer):
            return i

    raise ValueError(
        'No linear Dense regression head found. For models with a '
        'range-clipping head pass the Dense layer (layer=...) explicitly.'
    )


class ReferenceShift(Layer):
    """Adds the sample dependent offsets ``(-a_ref, 0, +a_ref)`` to the
    triplicated pre-activations ``(z, -z, -z)``.

    ``a_ref`` is derived from the first third of the channels (which equals
    the original pre-activation ``z``) with the flooding rule, see module
    docstring.
    """

    def __init__(self, coefficients: np.ndarray, *, reference_value: float,
                 asymmetry: float = 0.0, bisection_steps: int = 60,
                 name: str = 'reference_shift', **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)

        coefficients = np.asarray(coefficients, dtype=np.float32)
        if coefficients.ndim < 1:
            raise ValueError('coefficients must have at least one dimension')
        if asymmetry < 0:
            raise ValueError('asymmetry must be >= 0')

        self._coefficients = coefficients
        self.units = int(coefficients.shape[-1])
        self.reference_value = float(reference_value)
        self.asymmetry = float(asymmetry)
        self.bisection_steps = int(bisection_steps)

    def compute_output_shape(self, input_shape):
        return input_shape

    def _flooding_setup(self, z_tri: tf.Tensor):
        """Activations ``a``, coefficients ``c``, flooding direction ``d``
        (per unit), target ``y_ref`` and the reduction axes."""
        a = tf.nn.relu(z_tri[..., :self.units])
        c = tf.constant(self._coefficients[None, ...], dtype=a.dtype)
        axes = list(range(1, len(a.shape)))
        y_ref = tf.cast(self.reference_value, a.dtype)
        kappa = tf.cast(self.asymmetry, a.dtype)

        g0 = tf.reduce_sum(a * c, axis=axes, keepdims=True)
        # +1: evidence has to be removed ("down"), -1: added ("up").
        direction = tf.where(g0 >= y_ref, tf.ones_like(g0), -tf.ones_like(g0))
        cd = c * direction
        d = tf.where(cd > 0, tf.ones_like(cd),
                     tf.where(cd < 0, -kappa * tf.ones_like(cd),
                              tf.zeros_like(cd)))

        return a, c, d, y_ref, direction, axes

    def flooding_level(self, z_tri: tf.Tensor) -> tf.Tensor:
        """Per-sample flooding level ``t`` (shape ``(batch, 1, ..., 1)``)."""
        a, c, d, y_ref, direction, axes = self._flooding_setup(z_tri)

        def h(t):
            g = tf.reduce_sum(tf.nn.relu(a - t * d) * c, axis=axes,
                              keepdims=True)
            # Positive at t = 0 and monotonically decreasing in t.
            return (g - y_ref) * direction

        # Beyond max(a) all "d = +1" units are flooded completely. Only the
        # "d = -kappa" units keep changing the output linearly, so the upper
        # bracket is extended until the target is passed (if possible).
        hi = tf.reduce_max(a, axis=axes, keepdims=True) + 1.0
        if self.asymmetry > 0:
            def _cond(hi_, k_):
                return tf.logical_and(tf.reduce_any(h(hi_) > 0), k_ < 64)

            def _body(hi_, k_):
                return tf.where(h(hi_) > 0, hi_ * 2.0, hi_), k_ + 1

            hi, _ = tf.while_loop(_cond, _body, (hi, tf.constant(0)))

        check = tf.debugging.assert_less_equal(
            h(hi), tf.zeros_like(hi),
            message=('LRP reference value is not reachable by flooding the '
                     'activations of the last hidden layer. Choose a '
                     'reference value closer to the prediction or set '
                     'reference_asymmetry > 0.'))
        with tf.control_dependencies([check]):
            lo = tf.zeros_like(hi)
        for _ in range(self.bisection_steps):
            mid = 0.5 * (lo + hi)
            above = h(mid) > 0
            lo = tf.where(above, mid, lo)
            hi = tf.where(above, hi, mid)

        return 0.5 * (lo + hi)

    def reference_activations(self, z_tri: tf.Tensor) -> tf.Tensor:
        """``a_ref = (relu(z) - t * d)_+`` with the same shape as ``z``."""
        a, _, d, _, _, _ = self._flooding_setup(z_tri)
        t = self.flooding_level(z_tri)

        return tf.nn.relu(a - t * d)

    def compute_shift(self, z_tri: tf.Tensor) -> tf.Tensor:
        """Offsets ``(-a_ref, 0, +a_ref)`` concatenated along the channels."""
        a_ref = self.reference_activations(z_tri)

        return tf.concat([-a_ref, tf.zeros_like(a_ref), a_ref], axis=-1)

    def call(self, z_tri: tf.Tensor) -> tf.Tensor:
        return z_tri + self.compute_shift(z_tri)


class ReferenceActivations(Layer):
    """Exposes ``a_ref`` of a :class:`ReferenceShift` layer as a tensor."""

    def __init__(self, shift: ReferenceShift, name: str = 'reference_activations',
                 **kwargs):
        super().__init__(trainable=False, name=name, **kwargs)
        self.shift = shift

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.shift.units,)

    def call(self, z_tri: tf.Tensor) -> tf.Tensor:
        return self.shift.reference_activations(z_tri)


def _walk_top_block(model: Model) -> Tuple[Dense, List[Layer], Layer, Layer,
                                            List[BatchNormalization]]:
    """Return ``(dense, chain, relu, pre, norms)``.

    ``chain`` (between ReLU and Dense) and ``norms`` (BatchNormalization
    layers between ``pre`` and the ReLU, usually already fused and therefore
    close to identity) are in forward order.
    """
    dense = _producer(model, model.output)
    if not isinstance(dense, Dense):
        raise NotImplementedError(
            'LRP with a reference value requires a Dense target layer, got '
            f'{dense.__class__.__name__}. Range-clipping heads are stripped '
            'automatically; otherwise pass the Dense layer (layer=...).')
    if not _is_linear(dense):
        raise NotImplementedError(
            'LRP with a reference value requires a linear Dense target layer')

    chain: List[Layer] = []
    layer = _producer(model, dense.input)
    while isinstance(layer, _PASSTHROUGH):
        chain.insert(0, layer)
        layer = _producer(model, layer.input)

    norms: List[BatchNormalization] = []
    if isinstance(layer, _PRE_LAYERS) and \
            getattr(layer.activation, '__name__', '') == 'relu':
        # Built-in activation (e.g. Dense(..., activation='relu')).
        relu = None
        pre = layer
    elif _is_plain_relu(layer):
        relu = layer
        pre = _producer(model, relu.input)
        while isinstance(pre, BatchNormalization):
            axis = pre.axis if isinstance(pre.axis, int) else pre.axis[0]
            if axis not in (-1, len(pre.input.shape) - 1):
                raise NotImplementedError('Only channels_last '
                                          'BatchNormalization is supported')
            norms.insert(0, pre)
            pre = _producer(model, pre.input)
    else:
        raise NotImplementedError(
            'LRP with a reference value requires a ReLU directly below the '
            'linear top (optionally separated by Dropout / Flatten / global '
            f'average pooling), got {layer.__class__.__name__}')

    if not isinstance(pre, _PRE_LAYERS) or \
            not (relu is None or _is_linear(pre)):
        raise NotImplementedError(
            'LRP with a reference value requires a linear Conv2D/Conv3D/Dense '
            f'layer in front of the ReLU, got {pre.__class__.__name__}')
    if isinstance(pre, (Conv2D, Conv3D)):
        if pre.data_format != 'channels_last' or pre.groups != 1:
            raise NotImplementedError('Only channels_last convolutions with '
                                      'groups=1 are supported')

    if sum(isinstance(l, Flatten) for l in chain) > 1 or \
            sum(isinstance(l, _GLOBAL_AVG_POOLS) for l in chain) > 1:
        raise NotImplementedError('At most one Flatten and one global average '
                                  'pooling layer are supported between the '
                                  'ReLU and the target Dense layer')

    return dense, chain, relu, pre, norms


def _fold_norms(kernel: np.ndarray, bias: np.ndarray,
                norms: List[BatchNormalization]) -> Tuple[np.ndarray, np.ndarray]:
    """Fold per-channel affine BatchNormalization maps into kernel/bias."""
    for norm in norms:
        gamma = norm.gamma.numpy() if norm.scale else 1.0
        beta = norm.beta.numpy() if norm.center else 0.0
        mean = norm.moving_mean.numpy()
        var = norm.moving_variance.numpy()
        scale = gamma / np.sqrt(var + norm.epsilon)
        kernel = kernel * scale
        bias = bias * scale + (beta - mean * scale)
    return kernel, bias


def _dense_input_shape(dense: Dense, chain: List[Layer]) -> Tuple[int, ...]:
    """Shape (without batch) of the Dense input *before* a Flatten."""
    for layer in chain:
        if isinstance(layer, Flatten):
            return tuple(int(s) for s in layer.input.shape[1:])
    return tuple(int(s) for s in dense.input.shape[1:])


def output_coefficients(model: Model, idx: int) -> np.ndarray:
    """Effective linear weights ``c`` of output ``idx`` w.r.t. the ReLU units.

    ``c`` has the shape of the ReLU output (without batch dimension) and
    satisfies ``f_idx(x) = sum(relu_output * c) + bias``.
    """
    model = _strip_range_clip(model)
    dense, chain, relu, pre, _ = _walk_top_block(model)
    w = dense.get_weights()[0][:, idx]
    c = w.reshape(_dense_input_shape(dense, chain))

    unit_shape = tuple(int(s) for s in pre.output.shape[1:])
    gaps = [l for l in chain if isinstance(l, _GLOBAL_AVG_POOLS)]
    if gaps:
        n_spatial = len(unit_shape) - 1
        positions = int(np.prod(unit_shape[:-1]))
        c = np.broadcast_to(c.reshape((1,) * n_spatial + (unit_shape[-1],)),
                            unit_shape) / positions
    if tuple(c.shape) != unit_shape:
        raise ValueError(f'Coefficient shape {c.shape} does not match the '
                         f'ReLU output shape {unit_shape}')

    return np.ascontiguousarray(c, dtype=np.float32)


def restructure_for_reference(model: Model, *, idx: int,
                              reference_value: float,
                              asymmetry: float = 0.0,
                              name: str = 'LRP'
                              ) -> Tuple[Model, ReferenceShift, Layer]:
    """Rewrite the top of ``model`` so that it outputs ``f(x) - reference_value``.

    Returns the restructured model, the :class:`ReferenceShift` layer and the
    triplicated pre-activation layer. ``model`` must already be free of
    BatchNormalization layers in the top block (see ``fuse_batchnorm``).
    """
    model = _strip_range_clip(model)
    dense, chain, relu, pre, norms = _walk_top_block(model)
    coefficients = output_coefficients(model, idx)

    # --- Step 2: triplicated pre-activation layer -----------------------
    pre_weights = pre.get_weights()
    kernel = pre_weights[0]
    bias = pre_weights[1] if pre.use_bias else np.zeros(kernel.shape[-1],
                                                        dtype=kernel.dtype)
    kernel, bias = _fold_norms(kernel, bias, norms)
    kernel_tri = np.concatenate([kernel, -kernel, -kernel], axis=-1)
    bias_tri = np.concatenate([bias, -bias, -bias], axis=-1)

    if isinstance(pre, Dense):
        pre_tri = Dense(3 * pre.units, activation=None, use_bias=True,
                        name=f'{name}_reference_{pre.name}')
    else:
        pre_tri = pre.__class__(
            3 * pre.filters,
            kernel_size=pre.kernel_size,
            strides=pre.strides,
            padding=pre.padding,
            dilation_rate=pre.dilation_rate,
            activation=None,
            use_bias=True,
            name=f'{name}_reference_{pre.name}',
        )
    z_tri = pre_tri(pre.input)
    pre_tri.set_weights([kernel_tri, bias_tri])

    shift = ReferenceShift(coefficients, reference_value=reference_value,
                           asymmetry=asymmetry,
                           name=f'{name}_reference_shift')
    x = shift(z_tri)
    relu_name = relu.name if relu is not None else f'{pre.name}_relu'
    x = Activation('relu', name=f'{name}_reference_{relu_name}')(x)

    # --- Step 1: rebuild the linear top on the triplicated units --------
    for layer in chain:
        if isinstance(layer, Dropout):
            continue
        if isinstance(layer, Flatten):
            x = Flatten(name=f'{name}_reference_{layer.name}')(x)
        else:
            pool_kwargs = {
                'keepdims': layer.keepdims,
                'name': f'{name}_reference_{layer.name}',
            }
            data_format = getattr(layer, 'data_format', None)
            if data_format is not None:
                pool_kwargs['data_format'] = data_format
            x = layer.__class__(**pool_kwargs)(x)

    w, *b = dense.get_weights()
    in_shape = _dense_input_shape(dense, chain)
    w = w.reshape(in_shape + (dense.units,))
    w_tri = np.concatenate([w, w, -w], axis=len(in_shape) - 1)
    w_tri = w_tri.reshape((-1, dense.units))
    dense_tri = Dense(dense.units, activation=None, use_bias=dense.use_bias,
                      name=f'{name}_reference_{dense.name}')
    output = dense_tri(x)
    dense_tri.set_weights([w_tri] + b)

    restructured = Model(model.input, output,
                         name=f'{name}_reference_{model.name}')

    return restructured, shift, pre_tri
