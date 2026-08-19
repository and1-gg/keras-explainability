"""Tests für AveragePoolingLRP.

AveragePoolingLRP propagiert Relevanz durch Average-Pooling-Schichten zurück.
Die Standard-Strategie ist 'redistribute' (Relevanz proportional zur
Aktivierungsstärke verteilt). Außerdem unterstützt die Schicht die
'flat'-Strategie (gleichmäßige Verteilung).
"""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, Input

from explainability.layers import AveragePoolingLRP
from explainability import LRP


def test_avgpool2d_wrong_layer_raises():
    """AveragePoolingLRP darf nur mit AvgPooling-Schichten verwendet werden."""
    from tensorflow.keras.layers import MaxPooling2D
    layer = MaxPooling2D(pool_size=(2, 2))
    inp = Input((4, 4, 1))
    out = layer(inp)
    Model(inp, out)

    with pytest.raises(AssertionError):
        AveragePoolingLRP(layer)


def test_avgpool2d_redistribute_relevance_sum_preserved():
    """Die redistribute-Strategie erhält die Gesamtrelevanz näherungsweise."""
    inp = Input((4, 4, 1))
    pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(inp)
    model = Model(inp, pool)

    data = np.ones((1, 4, 4, 1), dtype=np.float32)
    R = np.ones((1, 2, 2, 1), dtype=np.float32)

    lrp_layer = AveragePoolingLRP(model.layers[1], strategy='redistribute')
    result = lrp_layer([inp, R])

    explainer = Model(inp, result)
    explanations = explainer(data).numpy()

    assert explanations.shape == data.shape, (
        "AveragePoolingLRP (redistribute) gibt falsche Shape zurück"
    )
    assert np.allclose(np.sum(explanations), np.sum(R), atol=1e-4), (
        "Gesamtrelevanz wird bei redistribute nicht erhalten"
    )


def test_avgpool2d_flat_uniform_distribution():
    """Die flat-Strategie verteilt Relevanz gleichmäßig über alle Pixel."""
    inp = Input((4, 4, 1))
    pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(inp)
    model = Model(inp, pool)

    data = np.random.rand(1, 4, 4, 1).astype(np.float32)
    R_val = 4.0
    R = np.full((1, 2, 2, 1), R_val, dtype=np.float32)

    lrp_layer = AveragePoolingLRP(model.layers[1], strategy='flat')
    result = lrp_layer([inp, R])

    explainer = Model(inp, result)
    explanations = explainer(data).numpy()

    assert explanations.shape == data.shape

    # Bei flat: alle Werte sollen gleich sein
    unique_values = np.unique(np.round(explanations, 5))
    assert len(unique_values) == 1, (
        "flat-Strategie verteilt Relevanz nicht gleichmäßig"
    )


def test_global_avgpool2d_in_lrp_model():
    """GlobalAveragePooling2D in einem vollständigen LRP-Modell funktioniert."""
    inp = Input((8, 8, 2))
    x = GlobalAveragePooling2D()(inp)
    from tensorflow.keras.layers import Dense
    x = Dense(1, activation=None)(x)
    model = Model(inp, x)
    model.layers[-1].set_weights([
        np.ones((2, 1), dtype=np.float32),
        np.zeros(1, dtype=np.float32)
    ])

    explainer = LRP(model, layer=2, idx=0)
    data = np.ones((1, 8, 8, 2), dtype=np.float32)
    explanations = explainer(data).numpy()

    assert explanations.shape == (1, 8, 8, 2), (
        "LRP mit GlobalAveragePooling2D gibt falsche Shape zurück"
    )
    assert np.all(explanations >= 0), (
        "LRP-Erklärungen sind bei positivem Eingabewert negativ"
    )


def test_avgpool2d_strategy_invalid_raises():
    """Eine unbekannte Strategie soll einen Fehler auslösen."""
    inp = Input((4, 4, 1))
    pool = AveragePooling2D(pool_size=(2, 2))(inp)
    model = Model(inp, pool)

    with pytest.raises(ValueError):
        AveragePoolingLRP(model.layers[1], strategy='nonexistent_strategy')
