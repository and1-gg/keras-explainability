"""LRP with a user-provided reference value (Letzgus et al. 2022, arXiv:2112.11407).

Without ``reference_value`` the existing LRP path must stay unchanged.
With ``reference_value`` the top of the net is restructured so that LRP
explains ``g(x) = f(x) - y_ref``.
"""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    Conv2D,
    Dense,
    GlobalAveragePooling2D,
    Input,
    ReLU,
)

from explainability import LRP
from explainability.model.utils.reference import (
    find_linear_dense_head,
    output_coefficients,
    restructure_for_reference,
)


def _make_dense_regressor(n_in: int = 4, n_hidden: int = 6) -> Model:
    inp = Input((n_in,))
    x = Dense(n_hidden, activation=None, name='pre')(inp)
    x = Activation('relu', name='relu')(x)
    x = Dense(1, activation=None, name='out')(x)
    model = Model(inp, x, name='dense_reg')
    # Positive, well-conditioned weights so ReLU units are on and flooding
    # can reach references between 0 and f(x).
    w_pre = np.zeros((n_in, n_hidden), dtype=np.float32)
    for j in range(n_hidden):
        w_pre[j % n_in, j] = 1.0
    model.layers[1].set_weights([
        w_pre,
        np.zeros((n_hidden,), dtype=np.float32),
    ])
    w_out = np.linspace(0.4, 1.0, n_hidden, dtype=np.float32).reshape(-1, 1)
    model.layers[3].set_weights([
        w_out,
        np.zeros((1,), dtype=np.float32),
    ])
    return model


def _make_conv_gap_regressor() -> Model:
    inp = Input((6, 6, 1))
    x = Conv2D(3, (3, 3), padding='same', activation=None, name='pre')(inp)
    x = Activation('relu', name='relu')(x)
    x = GlobalAveragePooling2D(name='gap')(x)
    x = Dense(1, activation=None, name='out')(x)
    return Model(inp, x, name='conv_reg')


def _make_clipped_regressor() -> Model:
    inp = Input((4,))
    x = Dense(6, activation=None, name='pre')(inp)
    x = Activation('relu', name='relu')(x)
    x = Dense(1, activation=None, name='predictions')(x)
    x = ReLU(max_value=10.0, name='restrict_relu')(x)
    x = Add(name='restrict_add')([x, tf.constant([[1.0]], dtype=tf.float32)])
    return Model(inp, x, name='clipped_reg')


def test_without_reference_value_has_no_restructured_model():
    model = _make_dense_regressor()
    explainer = LRP(model, layer=3, idx=0)
    assert explainer.reference_value is None
    assert explainer.restructured_model is None
    assert explainer.reference_activations_model is None


def test_restructured_output_equals_f_minus_yref():
    model = _make_dense_regressor()
    data = np.asarray([[0.4, 0.2, 0.8, 0.1]], dtype=np.float32)
    y = model(data).numpy()
    y_ref = 0.4 * float(np.squeeze(y))
    restructured, _, _ = restructure_for_reference(
        model, idx=0, reference_value=y_ref
    )
    g = restructured(data).numpy()
    np.testing.assert_allclose(g, y - y_ref, atol=1e-5, rtol=1e-5)


def test_lrp_reference_conservation_dense():
    model = _make_dense_regressor()
    data = np.asarray([[0.5, 0.1, 0.3, 0.2]], dtype=np.float32)
    y = float(np.squeeze(model(data).numpy()))
    y_ref = 0.3 * y
    explainer = LRP(model, layer=3, idx=0, reference_value=y_ref)
    R = np.asarray(explainer(data).numpy(), dtype=np.float64)
    np.testing.assert_allclose(np.sum(R), y - y_ref, atol=5e-4, rtol=1e-4)


def test_flooding_matches_reference_value():
    model = _make_dense_regressor()
    data = np.asarray([[0.7, 0.1, 0.4, 0.2]], dtype=np.float32)
    y = float(np.squeeze(model(data).numpy()))
    y_ref = 0.5 * y
    explainer = LRP(model, layer=3, idx=0, reference_value=y_ref)
    a_ref = np.asarray(
        explainer.reference_activations_model(data).numpy(), dtype=np.float64
    )
    c = output_coefficients(model, 0).astype(np.float64)
    np.testing.assert_allclose(np.sum(a_ref * c), y_ref, atol=1e-4, rtol=1e-4)


def test_conv_gap_restructured_output_and_lrp_shape():
    tf.keras.utils.set_random_seed(1)
    model = _make_conv_gap_regressor()
    data = np.abs(
        np.random.default_rng(1).normal(size=(1, 6, 6, 1))
    ).astype(np.float32)
    y = model(data).numpy()
    y_ref = 0.3 * float(np.squeeze(y))
    restructured, _, _ = restructure_for_reference(
        model, idx=0, reference_value=y_ref, asymmetry=0.25
    )
    np.testing.assert_allclose(
        restructured(data).numpy(), y - y_ref, atol=1e-4, rtol=1e-4
    )
    explainer = LRP(
        model, layer=4, idx=0, reference_value=y_ref, reference_asymmetry=0.25
    )
    R = np.asarray(explainer(data).numpy())
    assert R.shape == data.shape


def test_find_linear_dense_head_skips_range_clip():
    model = _make_clipped_regressor()
    idx = find_linear_dense_head(model)
    assert model.layers[idx].name == 'predictions'


def test_reference_lrp_accepts_clipped_output_layer():
    model = _make_clipped_regressor()
    w_pre = np.zeros((4, 6), dtype=np.float32)
    w_pre[:, :4] = np.eye(4, dtype=np.float32)
    w_pre[:, 4:] = np.eye(4, dtype=np.float32)[:, :2]
    model.get_layer('pre').set_weights([
        w_pre,
        np.zeros((6,), dtype=np.float32),
    ])
    model.get_layer('predictions').set_weights([
        np.ones((6, 1), dtype=np.float32),
        np.zeros((1,), dtype=np.float32),
    ])
    data = np.asarray([[0.4, 0.3, 0.2, 0.1]], dtype=np.float32)
    dense_idx = find_linear_dense_head(model)
    y_dense = float(np.squeeze(
        Model(model.input, model.layers[dense_idx].output)(data)
    ))
    y_ref = 0.4 * y_dense
    explainer = LRP(
        model, layer=len(model.layers) - 1, idx=0, reference_value=y_ref
    )
    R = np.asarray(explainer(data).numpy(), dtype=np.float64)
    np.testing.assert_allclose(np.sum(R), y_dense - y_ref, atol=5e-4, rtol=1e-4)
    g = explainer.restructured_model(data).numpy()
    np.testing.assert_allclose(g, np.asarray([[y_dense - y_ref]]), atol=1e-5)


def test_classic_and_reference_lrp_differ_for_nonzero_yref():
    model = _make_dense_regressor()
    data = np.asarray([[0.9, 0.2, 0.4, 0.3]], dtype=np.float32)
    classic = np.asarray(LRP(model, layer=3, idx=0)(data).numpy())
    y_ref = float(np.squeeze(model(data).numpy())) * 0.4
    paper = np.asarray(
        LRP(model, layer=3, idx=0, reference_value=y_ref)(data).numpy()
    )
    assert not np.allclose(classic, paper, atol=1e-5)
