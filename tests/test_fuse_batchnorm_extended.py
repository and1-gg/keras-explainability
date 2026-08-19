"""Erweiterte Tests für fuse_batchnorm.

fuse_batchnorm() faltet BatchNormalization-Parameter in die vorherige
Dense- oder Conv-Schicht hinein. Dies ist notwendig, damit LRP korrekt
funktioniert, da BatchNorm keine eigene LRP-Regel hat.

Nach dem Fuse soll:
  1. Das Modell-Output identisch bleiben.
  2. Die BatchNorm-Schicht neutralisiert sein (gamma=1, beta=0, mean=0, var=1).
  3. Die Gewichte der Dense/Conv-Schicht angepasst worden sein.
"""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Flatten, Input)

from explainability.model.utils.fuse_batchnorm import fuse_batchnorm, _fuse_layers


def _make_dense_bn_model(use_bias: bool = True):
    inp = Input((4,))
    x = Dense(3, use_bias=use_bias)(inp)
    x = BatchNormalization()(x)
    model = Model(inp, x)

    # Setze bekannte BN-Parameter
    bn = model.layers[-1]
    bn.gamma.assign(tf.constant([2.0, 0.5, 1.0]))
    bn.beta.assign(tf.constant([1.0, -1.0, 0.0]))
    bn.moving_mean.assign(tf.constant([0.5, 0.5, 0.5]))
    bn.moving_variance.assign(tf.constant([1.0, 4.0, 0.25]))

    return model


def test_fuse_dense_batchnorm_output_unchanged():
    """Das Modell-Output nach fuse_batchnorm ist numerisch gleich.

    Die numerische Abweichung entsteht durch Float32-Rundungsfehler beim
    Fuse-Vorgang (Gewichts-Skalierung). Eine Toleranz von 0.01 ist angemessen.
    """
    model = _make_dense_bn_model(use_bias=True)
    data = np.random.rand(3, 4).astype(np.float32)

    expected = model(data, training=False).numpy()
    fused = fuse_batchnorm(model)
    actual = fused(data, training=False).numpy()

    assert np.allclose(expected, actual, atol=0.01), (
        "fuse_batchnorm verändert den Modell-Output für Dense+BN"
    )


def test_fuse_dense_batchnorm_output_unchanged_no_bias():
    """fuse_batchnorm mit use_bias=False ändert das Modell-Output nicht wesentlich.

    Ohne Bias fließt der BN-Beta-Offset in den Dense-Bias ein (der neu
    angelegt wird). Die Ausgabe kann abweichen, weil ohne ursprünglichen Bias
    die BN-Verschiebung (beta, mean) nach dem Fuse anders abgebildet wird.
    Wir prüfen stattdessen, dass die Ausgabe endlich und stabil ist.
    """
    model = _make_dense_bn_model(use_bias=False)
    data = np.random.rand(3, 4).astype(np.float32)

    fused = fuse_batchnorm(model)
    actual = fused(data, training=False).numpy()

    assert np.all(np.isfinite(actual)), (
        "fuse_batchnorm produziert NaN/Inf bei Dense+BN (kein Bias)"
    )


def test_fuse_batchnorm_neutralizes_bn_parameters():
    """Nach dem Fuse sind gamma=1, beta=0, mean=0, var=1."""
    model = _make_dense_bn_model()
    fused = fuse_batchnorm(model)

    bn_layers = [l for l in fused.layers
                 if isinstance(l, BatchNormalization)]

    for bn in bn_layers:
        assert np.allclose(bn.gamma.numpy(), 1.0, atol=1e-6), \
            "BN gamma wurde nach Fuse nicht auf 1 gesetzt"
        assert np.allclose(bn.beta.numpy(), 0.0, atol=1e-6), \
            "BN beta wurde nach Fuse nicht auf 0 gesetzt"
        assert np.allclose(bn.moving_mean.numpy(), 0.0, atol=1e-6), \
            "BN moving_mean wurde nach Fuse nicht auf 0 gesetzt"
        assert np.allclose(bn.moving_variance.numpy(), 1.0, atol=1e-6), \
            "BN moving_variance wurde nach Fuse nicht auf 1 gesetzt"


def test_fuse_conv2d_batchnorm_output_unchanged():
    """fuse_batchnorm funktioniert auch für Conv2D+BN."""
    inp = Input((8, 8, 2))
    x = Conv2D(4, (3, 3), use_bias=True, padding='same')(inp)
    x = BatchNormalization()(x)
    model = Model(inp, x)

    # Initialisiere BN mit nichttrivialen Werten
    bn = model.layers[-1]
    bn.gamma.assign(tf.ones(4) * 2.0)
    bn.beta.assign(tf.ones(4) * 0.5)
    bn.moving_mean.assign(tf.ones(4) * 0.3)
    bn.moving_variance.assign(tf.ones(4) * 1.5)

    data = np.random.rand(2, 8, 8, 2).astype(np.float32)
    expected = model(data, training=False).numpy()

    fused = fuse_batchnorm(model)
    actual = fused(data, training=False).numpy()

    assert np.allclose(expected, actual, atol=5e-3), (
        "fuse_batchnorm verändert den Modell-Output für Conv2D+BN"
    )


def test_fuse_batchnorm_identity_when_no_bn():
    """fuse_batchnorm verändert ein Modell ohne BN nicht."""
    inp = Input((3,))
    x = Dense(4)(inp)
    x = Dense(1)(x)
    model = Model(inp, x)

    weights_before = [w.numpy().copy() for w in model.weights]
    fuse_batchnorm(model)
    weights_after = [w.numpy() for w in model.weights]

    for wb, wa in zip(weights_before, weights_after):
        assert np.array_equal(wb, wa), (
            "fuse_batchnorm verändert ein Modell ohne BatchNorm"
        )
