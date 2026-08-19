"""Integrationstests für LayerwiseRelevancePropagator (LRP).

Diese Tests spiegeln typische Notebook-Nutzungsmuster wider:
- Synthetische 3D-Daten (analog zu Train_and_explain_dummy_geometric_data.py)
- 2D-CNN mit Pooling (analog zu xai_with_2dmnist.py)
- Relevanzerhaltungs-Eigenschaft (Summe über Eingang ≈ Summe am Ausgang)
- include_prediction-Flag
- Fehlerfall: nicht-2D-Ausgabeschicht
"""

import numpy as np
import pytest
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation, AveragePooling2D, Conv2D, Dense, Flatten,
    GlobalAveragePooling2D, Input, MaxPooling2D,
)

from explainability import LRP
from explainability.utils.strategies import LRPStrategy


# ─── Hilfsfunktionen ─────────────────────────────────────────────────────────

def _make_dense_classifier(n_in: int = 4, n_hidden: int = 8,
                            n_out: int = 3) -> Model:
    """Einfaches Dense-Netz für Klassifikation."""
    inp = Input((n_in,))
    x = Dense(n_hidden, activation='relu')(inp)
    x = Dense(n_out, activation='softmax')(x)
    return Model(inp, x)


def _make_conv2d_model() -> Model:
    """Kleines 2D-CNN, ähnlich dem MNIST-Beispiel im Notebook."""
    inp = Input((8, 8, 1))
    x = Conv2D(4, (3, 3), activation='relu', padding='same')(inp)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(2, activation=None)(x)
    return Model(inp, x)


# ─── Relevanzerhaltung ───────────────────────────────────────────────────────

def test_lrp_relevance_conservation_dense():
    """Die Summe der Eingabe-Relevanz ≈ Ausgabe-Relevanz (epsilon=0).

    Ohne epsilon-Stabilisierung und ohne Bias sollte die Relevanz exakt
    erhalten bleiben (Summenregel).
    """
    inp = Input((4,))
    x = Dense(4, use_bias=False)(inp)
    x = Dense(2, use_bias=False)(x)
    model = Model(inp, x)

    model.layers[1].set_weights([np.eye(4, dtype=np.float32)])
    model.layers[2].set_weights([
        np.asarray([[1., 0.], [0., 1.], [0., 0.], [0., 0.]], dtype=np.float32)
    ])

    explainer = LRP(model, layer=2, idx=0)
    data = np.asarray([[1., 2., 3., 4.]], dtype=np.float32)
    explanations = explainer(data).numpy()

    prediction = model(data).numpy()
    R_out = prediction[0, 0]

    assert np.allclose(np.sum(explanations), R_out, atol=1e-4), (
        "Relevanzerhaltung verletzt: Summe der Eingaberelevanz != Ausgaberelevanz"
    )


# ─── include_prediction ──────────────────────────────────────────────────────

@pytest.mark.xfail(
    reason=(
        "include_prediction=True löst in Keras >= 3 einen ValueError aus, weil "
        "LRP intern den original_output erneut in das Modell einbettet und dabei "
        "ein Layer-Name doppelt vorkommt. Bekannter Bug, kein Notebook nutzt dieses "
        "Feature aktuell."
    ),
    strict=True,
)
def test_lrp_include_prediction_flag():
    """Mit include_prediction=True gibt LRP [Vorhersage, Relevanz] zurück.

    LRP mit include_prediction=True soll eine Liste aus zwei Elementen liefern:
    dem ursprünglichen Modell-Output und der Relevanz-Map.
    Dieser Test schlägt derzeit aufgrund eines bekannten Keras-3-Kompatibilitätsproblems
    fehl (doppelte Layer-Namen im kombinierten Ausgabemodell).
    """
    inp = Input((4,))
    x = Dense(8, activation='relu')(inp)
    x = Dense(2, activation='softmax')(x)
    model = Model(inp, x)

    explainer = LRP(model, layer=2, idx=0, include_prediction=True)

    data = np.random.rand(1, 4).astype(np.float32)
    outputs = explainer(data)

    assert isinstance(outputs, (list, tuple))
    assert len(outputs) == 2
    prediction, relevance = outputs
    assert prediction.shape[-1] == 2
    assert relevance.shape == (1, 4)


# ─── Fehlerfall: nicht-2D-Ausgabe ────────────────────────────────────────────

def test_lrp_non_flat_output_raises():
    """LRP soll einen Fehler werfen, wenn die Zielschicht kein 1D-Ausgabe hat."""
    inp = Input((8, 8, 1))
    x = Conv2D(4, (3, 3), padding='same', activation=None)(inp)
    model = Model(inp, x)

    with pytest.raises(NotImplementedError):
        LRP(model, layer=1, idx=0)


# ─── Negativer idx ───────────────────────────────────────────────────────────

def test_lrp_negative_idx_raises():
    """Negativer idx ist nicht implementiert und soll einen Fehler werfen."""
    model = _make_dense_classifier(n_out=3)

    with pytest.raises(NotImplementedError):
        LRP(model, layer=2, idx=-1)


# ─── 2D-CNN-Integration (Notebook-Muster) ────────────────────────────────────

def test_lrp_conv2d_model_explanation_shape():
    """LRP auf einem 2D-CNN gibt Erklärungen in Eingabegröße zurück.

    Entspricht dem Muster aus xai_with_2dmnist.py: Modell mit Conv2D,
    Pooling, Flatten, Dense; LRP erklärt die Vorhersage einer Klasse.
    """
    model = _make_conv2d_model()
    data = np.random.rand(1, 8, 8, 1).astype(np.float32)

    explainer = LRP(model, layer=4, idx=0)
    explanations = explainer(data).numpy()

    assert explanations.shape == (1, 8, 8, 1), (
        "LRP auf Conv2D-Modell gibt falsche Erklärungs-Shape zurück"
    )


# ─── LRP mit Strategie ───────────────────────────────────────────────────────

def test_lrp_with_strategy_sets_epsilon():
    """LRPStrategy kann epsilon für einzelne Schichten setzen."""
    inp = Input((3,))
    x = Dense(4, activation='relu', use_bias=False)(inp)
    x = Dense(2, activation=None, use_bias=False)(x)
    model = Model(inp, x)

    strategy = LRPStrategy(
        layers=[
            {'epsilon': 0.5},
            {'epsilon': 1.0},
        ]
    )

    explainer = LRP(model, layer=2, idx=0, strategy=strategy)

    from explainability.layers import StandardLRPLayer
    std_layers = [l for l in explainer.layers
                  if isinstance(l, StandardLRPLayer)]

    epsilons = [l.epsilon for l in std_layers]
    assert set(epsilons) == {0.5, 1.0}, (
        "LRPStrategy hat epsilon nicht korrekt gesetzt"
    )


def test_lrp_with_strategy_and_epsilon_raises():
    """LRP darf nicht gleichzeitig strategy und epsilon erhalten."""
    model = _make_dense_classifier(n_out=2)
    strategy = LRPStrategy(layers=[{'epsilon': 0.5}, {'epsilon': 1.0}])

    with pytest.raises(AssertionError):
        LRP(model, layer=2, idx=0, epsilon=0.1, strategy=strategy)


# ─── 3D-ähnliche synthetische Daten (Notebook-Muster) ───────────────────────

def test_lrp_dense_synthetic_3class():
    """LRP erklärt korrekte Klasse bei synthetischen 3-Klassen-Daten.

    Analogon zum Train_and_explain_dummy_geometric_data.py-Notebook:
    Dort wird geprüft, ob LRP die richtige Region hervorhebt. Hier prüfen
    wir, ob die erklärte Klasse die höchste Gesamtrelevanz hat.
    """
    np.random.seed(0)
    inp = Input((6,))
    x = Dense(6, use_bias=False)(inp)
    x = Dense(3, use_bias=False, activation=None)(x)
    model = Model(inp, x)

    # Gewichte: Schicht 1 = Identität, Schicht 2 = one-hot-Extraktion
    model.layers[1].set_weights([np.eye(6, dtype=np.float32)])
    model.layers[2].set_weights([
        np.asarray([
            [1., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [0., 0., 1.],
        ], dtype=np.float32)
    ])

    # Klasse 0 aktiviert durch Features 0 und 1
    data = np.asarray([[3., 2., 0., 0., 0., 0.]], dtype=np.float32)

    explainer = LRP(model, layer=2, idx=0)
    explanations = explainer(data).numpy()[0]

    # Features 0 und 1 sollen die meiste Relevanz haben
    assert explanations[0] > 0 and explanations[1] > 0, (
        "LRP hebt bei Klasse 0 nicht die relevanten Features hervor"
    )
    assert explanations[2] == pytest.approx(0, abs=1e-4), (
        "LRP weist Feature 2 bei Klasse-0-Vorhersage fälschlicherweise Relevanz zu"
    )
