"""Tests für NoOpLRP – die Passthrough-Schicht der LRP-Implementierung.

NoOpLRP wird für Schichten verwendet, die keine eigene Relevanzrückpropagation
benötigen (z. B. Dropout, Lambda). Die Relevanz wird unverändert durchgereicht.
"""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda

from explainability.layers.noop import NoOpLRP
from explainability.layers.layer import LRPLayer


class _DummyKerasLayer(tf.keras.layers.Layer):
    """Minimale Keras-Schicht, die NoOpLRP als Wrapper akzeptiert."""
    def call(self, x):
        return x


def _make_dummy_lrp_layer():
    dummy = _DummyKerasLayer()
    dummy(tf.zeros((1, 4)))  # build
    return NoOpLRP(dummy)


def test_noop_lrp_passes_relevance_unchanged():
    """NoOpLRP gibt R unverändert zurück, a wird ignoriert."""
    layer = _make_dummy_lrp_layer()

    a = tf.constant([[1.0, 2.0, 3.0, 4.0]])
    R = tf.constant([[10.0, 20.0, 30.0, 40.0]])

    result = layer([a, R])

    assert np.array_equal(result.numpy(), R.numpy()), (
        "NoOpLRP verändert die Relevanz R"
    )


def test_noop_lrp_ignores_activation():
    """Die Aktivierung a hat keinen Einfluss auf das Ergebnis."""
    layer = _make_dummy_lrp_layer()

    R = tf.constant([[5.0, -3.0, 0.0]])

    result_with_zeros = layer([tf.zeros((1, 3)), R])
    result_with_ones = layer([tf.ones((1, 3)), R])
    result_with_neg = layer([tf.constant([[-1.0, -2.0, -3.0]]), R])

    assert np.array_equal(result_with_zeros.numpy(), R.numpy())
    assert np.array_equal(result_with_ones.numpy(), R.numpy())
    assert np.array_equal(result_with_neg.numpy(), R.numpy())


def test_noop_lrp_preserves_zeros():
    """NoOpLRP gibt einen Null-Relevanzvektor unverändert zurück."""
    layer = _make_dummy_lrp_layer()

    a = tf.ones((1, 5))
    R = tf.zeros((1, 5))

    result = layer([a, R])

    assert np.array_equal(result.numpy(), np.zeros((1, 5)))


def test_noop_lrp_is_lrp_layer():
    """NoOpLRP ist eine Unterklasse von LRPLayer."""
    layer = _make_dummy_lrp_layer()
    assert isinstance(layer, LRPLayer)
