import tensorflow as tf

from tensorflow.keras.layers import Add, Subtract
from typing import List, Union

from .layer import LRPLayer


def _compute_add_lrp(a: tf.Tensor, b: tf.Tensor, R: tf.Tensor,
                     *, name: str, epsilon: float = 1e-9) -> tf.Tensor:
    forward = tf.add(a, b, name=f'{name}/forward')
    forward = tf.add(forward, epsilon, name=f'{name}/forward/epsilon')
    a = tf.divide(a, forward, name=f'{name}/a')
    a = tf.multiply(a, R, name=f'{name}/a/R')
    b = tf.divide(b, forward, name=f'{name}/b')
    b = tf.multiply(b, R, name=f'{name}/b/R')

    return [a, b]


def _unpack_binary_or_constant(inputs):
    """Unpack ``[[a, b], R]`` or Keras-3 ``[a, R]`` when the other Add
    operand is a plain ``tf.constant`` (not tracked as a layer input)."""
    xs, R = inputs
    if isinstance(xs, (list, tuple)):
        if len(xs) != 2:
            raise ValueError(
                f'Expected two operands for binary LRP, got {len(xs)}'
            )
        return xs[0], xs[1], R, False
    # Constant operand: all relevance stays on the tracked activation.
    return xs, None, R, True


class AddLRP(LRPLayer):
    def __init__(self, layer, *args, name: str = 'add_lrp', **kwargs):
        assert isinstance(layer, Add), \
            ('AddLRP should only be called with an Add layer')

        super().__init__(layer, *args, name=name, **kwargs)

    def compute_output_shape(self, input_shape):
        xs = input_shape[0]
        if isinstance(xs, (list, tuple)) and len(xs) == 2 and \
                not isinstance(xs[0], (int, type(None))):
            return [xs[0], xs[1]]
        return xs

    def call(self, inputs: List[tf.Tensor]) -> Union[tf.Tensor, List[tf.Tensor]]:
        a, b, R, constant_operand = _unpack_binary_or_constant(inputs)
        if constant_operand:
            return R
        return _compute_add_lrp(a, b, R, name=self.name)

class SubtractLRP(LRPLayer):
    def __init__(self, layer, *args, name: str = 'add_lrp', **kwargs):
        assert isinstance(layer, Subtract), \
            ('SubtractLRP should only be called with an Subtract layer')

        super().__init__(layer, *args, name=name, **kwargs)

    def compute_output_shape(self, input_shape):
        xs = input_shape[0]
        if isinstance(xs, (list, tuple)) and len(xs) == 2 and \
                not isinstance(xs[0], (int, type(None))):
            return [xs[0], xs[1]]
        return xs

    def call(self, inputs: List[tf.Tensor]) -> Union[tf.Tensor, List[tf.Tensor]]:
        a, b, R, constant_operand = _unpack_binary_or_constant(inputs)
        if constant_operand:
            return R
        b = tf.math.negative(b, name=f'{self.name}/negate')
        return _compute_add_lrp(a, b, R, name=self.name)
