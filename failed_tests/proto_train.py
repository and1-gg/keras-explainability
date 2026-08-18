"""Prototyp: Phantome (korrigiert) + SFCN-artiges CNN trainieren + LRP-Metriken."""
import os
import time
import numpy as np
import tensorflow as tf

for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

SIZE = 64
LBL_BG, LBL_CORTEX, LBL_WM, LBL_VENT, LBL_THAL = 0, 1, 2, 3, 4

_grid = np.stack(np.meshgrid(*[np.arange(SIZE)] * 3, indexing='ij'), axis=-1).astype(np.float32)


def _ellipsoid(center, radii):
    d = (_grid - np.asarray(center, np.float32)) / np.asarray(radii, np.float32)
    return (d ** 2).sum(-1) <= 1.0


def make_phantom(rng):
    label = np.full((SIZE,) * 3, LBL_BG, np.uint8)
    c = np.array([SIZE / 2, SIZE / 2, SIZE / 2]) + rng.uniform(-2, 2, 3)

    head_r = rng.uniform(24, 28, 3)
    label[_ellipsoid(c, head_r)] = LBL_CORTEX
    label[_ellipsoid(c, head_r - rng.uniform(2.5, 5.0))] = LBL_WM

    vent_r = rng.uniform(3.0, 6.5)
    for sign in (-1, 1):
        label[_ellipsoid(c + [sign * 5.0, 12.0, 2.0],
                         [vent_r * 0.7, vent_r, vent_r * 1.3])] = LBL_VENT

    thal_r = rng.uniform(3.0, 7.0)
    for sign in (-1, 1):
        label[_ellipsoid(c + [sign * (thal_r + 0.8), -6.0, 0],
                         [thal_r, thal_r * 1.15, thal_r * 0.9])] = LBL_THAL

    base = {LBL_CORTEX: 0.45, LBL_WM: 0.80, LBL_VENT: 0.06, LBL_THAL: 0.62}
    vol = np.zeros((SIZE,) * 3, np.float32)
    for lbl, val in base.items():
        vol[label == lbl] = val
    vol *= rng.uniform(0.9, 1.1)
    noise = rng.normal(0, rng.uniform(0.01, 0.04), vol.shape).astype(np.float32)
    vol = np.clip(vol + noise * (label != LBL_BG), 0, 1)
    return vol, label


def make_dataset(n, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, SIZE, SIZE, SIZE), np.float32)
    L = np.zeros((n, SIZE, SIZE, SIZE), np.uint8)
    for i in range(n):
        X[i], L[i] = make_phantom(rng)
    y = (L == LBL_THAL).sum((1, 2, 3)).astype(np.float32) / 100.
    return X, L, y


if __name__ == '__main__':
    t = time.time()
    X, L, y = make_dataset(600, 42)
    print('400 phantome in %.1fs' % (time.time() - t))
    n_brain = (L != LBL_BG).sum((1, 2, 3))
    n_vent = (L == LBL_VENT).sum((1, 2, 3))
    n_cor = (L == LBL_CORTEX).sum((1, 2, 3))
    print('y  %.2f..%.2f  mean %.2f  std %.2f' % (y.min(), y.max(), y.mean(), y.std()))
    print('corr(y, brain)=%.3f  corr(y, vent)=%.3f  corr(y, cortex)=%.3f' %
          (np.corrcoef(y, n_brain)[0, 1], np.corrcoef(y, n_vent)[0, 1],
           np.corrcoef(y, n_cor)[0, 1]))
    print('corr(y, mean intensity)=%.3f' % np.corrcoef(y, X.mean((1, 2, 3)))[0, 1])
    print('Baseline MAE (Mittelwert) = %.3f   Var = %.2f' % (np.abs(y - y.mean()).mean(), y.var()))
    
    ntr, nva = 420, 90
    tr = slice(0, ntr); va = slice(ntr, ntr + nva); te = slice(ntr + nva, None)
    
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Activation, BatchNormalization, Conv3D, Dense, \
                                        GlobalAveragePooling3D, Input, MaxPooling3D
    from tensorflow.keras.optimizers import Adam
    
    depths = [32, 64, 128, 256]
    inputs = Input((SIZE, SIZE, SIZE))
    z = tf.keras.layers.Reshape((SIZE, SIZE, SIZE, 1))(inputs)
    for d in depths:
        z = Conv3D(d, 3, padding='SAME', activation=None)(z)
        z = BatchNormalization()(z)
        z = Activation('relu')(z)
        z = MaxPooling3D(2)(z)
    z = Conv3D(64, 1, padding='SAME', activation=None)(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)
    z = GlobalAveragePooling3D()(z)
    z = Dense(1, activation=None)(z)
    model = Model(inputs, z)
    print('params', model.count_params())
    
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    CB = [ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=8, min_lr=1e-5),
          EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)]
    model.compile(loss='mse', optimizer=Adam(1e-3), metrics=['mae'])
    t = time.time()
    h = model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), batch_size=16, epochs=60, verbose=0, callbacks=CB)
    print('train %.1fs, epochs %d' % (time.time() - t, len(h.history['loss'])))
    print('last val_mae %.3f  best val_loss %.2f' % (h.history['val_mae'][-1], min(h.history['val_loss'])))
    p = model.predict(X[te], verbose=0).ravel()
    print('test MAE %.3f  (baseline %.3f)  corr %.3f' %
          (np.abs(p - y[te]).mean(), np.abs(y[te] - y[tr].mean()).mean(),
           np.corrcoef(p, y[te])[0, 1]))
    model.save('failed_tests/proto_phantom_model.keras')
