"""Erweiterte Tests für remove_activation, remove_softmax und remove_sigmoid.

remove_activation entfernt eine bestimmte Aktivierungsfunktion von der letzten
Dense-Schicht eines Keras-Modells. Dies ist notwendig, weil LRP für die rohen
Logits arbeitet – Softmax und Sigmoid würden die Relevanzpropagation verzerren.
"""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Input

from explainability.model.utils.remove_activation import (
    remove_activation,
    remove_sigmoid,
    remove_softmax,
)


def _make_model(activation: str):
    inp = Input((4,))
    x = Dense(3, activation=activation)(inp)
    return Model(inp, x)


def test_remove_softmax_changes_activation():
    """Nach remove_softmax hat die letzte Schicht keine Aktivierung."""
    model = _make_model('softmax')
    modified = remove_softmax(model)
    last = modified.layers[-1]

    assert isinstance(last, Dense)
    assert last.activation.__name__ == 'linear', (
        "remove_softmax entfernt Softmax nicht korrekt"
    )


def test_remove_sigmoid_changes_activation():
    """Nach remove_sigmoid hat die letzte Schicht keine Aktivierung."""
    model = _make_model('sigmoid')
    modified = remove_sigmoid(model)
    last = modified.layers[-1]

    assert isinstance(last, Dense)
    assert last.activation.__name__ == 'linear', (
        "remove_sigmoid entfernt Sigmoid nicht korrekt"
    )


def test_remove_activation_preserves_weights():
    """Die Gewichte der Dense-Schicht bleiben nach remove_activation erhalten."""
    model = _make_model('softmax')
    original_weights = model.layers[-1].get_weights()

    modified = remove_softmax(model)
    new_weights = modified.layers[-1].get_weights()

    for orig, new in zip(original_weights, new_weights):
        assert np.allclose(orig, new, atol=1e-6), (
            "remove_softmax verändert die Gewichte der Dense-Schicht"
        )


def test_remove_activation_preserves_output_shape():
    """Die Output-Shape bleibt nach remove_activation gleich."""
    model = _make_model('softmax')
    modified = remove_softmax(model)

    assert model.output_shape == modified.output_shape, (
        "remove_softmax verändert die Ausgabe-Shape"
    )


def test_remove_activation_skips_non_matching():
    """remove_activation mit nicht-passender Aktivierung verändert Modell nicht."""
    model = _make_model('relu')
    original_id = id(model)
    modified = remove_activation(model, ['softmax'])

    # Model soll unverändert zurückgegeben werden
    assert modified is model, (
        "remove_activation gibt ein neues Modell zurück, obwohl die "
        "Aktivierung nicht übereinstimmt"
    )


def test_remove_activation_non_dense_last_layer_skips():
    """Wenn die letzte Schicht keine Dense-Schicht ist, bleibt Modell unverändert."""
    inp = Input((4,))
    x = Dense(3, activation=None)(inp)
    x = Flatten()(x)
    model = Model(inp, x)

    modified = remove_activation(model, ['softmax'])
    assert modified is model, (
        "remove_activation verändert Modell, obwohl letzte Schicht kein Dense ist"
    )


def test_remove_softmax_logits_match_linear_model():
    """Die Logits nach remove_softmax stimmen mit einem linear-Modell überein.

    Wir bauen dasselbe Modell ohne Aktivierung und setzen identische Gewichte.
    """
    inp = Input((4,))
    dense_out = Dense(3, name='ref_dense', activation='softmax')(inp)
    model_softmax = Model(inp, dense_out)

    weights = model_softmax.layers[-1].get_weights()

    # Referenzmodell ohne Softmax
    inp2 = Input((4,))
    dense_linear = Dense(3, name='ref_dense_linear', activation=None)(inp2)
    model_linear = Model(inp2, dense_linear)
    model_linear.layers[-1].set_weights(weights)

    modified = remove_softmax(model_softmax)
    data = np.random.rand(5, 4).astype(np.float32)

    logits_modified = modified(data).numpy()
    logits_linear = model_linear(data).numpy()

    assert np.allclose(logits_modified, logits_linear, atol=1e-5), (
        "remove_softmax-Logits stimmen nicht mit einem linearen Referenzmodell überein"
    )
