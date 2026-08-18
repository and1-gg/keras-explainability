"""Prototyp: synthetische Thalamus-Phantome + SFCN-Training auf 64^3."""
import os
import time
import numpy as np
import tensorflow as tf

SIZE = 64
LBL_BG, LBL_CORTEX, LBL_WM, LBL_VENT, LBL_THAL = 0, 1, 2, 3, 4

_grid = np.stack(np.meshgrid(*[np.arange(SIZE)] * 3, indexing='ij'), axis=-1).astype(np.float32)


def _ellipsoid(center, radii):
    d = (_grid - np.asarray(center, np.float32)) / np.asarray(radii, np.float32)
    return (d ** 2).sum(-1) <= 1.0


def make_phantom(rng):
    label = np.full((SIZE,) * 3, LBL_BG, np.uint8)
    c = np.array([SIZE / 2, SIZE / 2, SIZE / 2]) + rng.uniform(-2, 2, 3)

    head_r = rng.uniform(24, 28, 3) * rng.uniform(0.95, 1.05)
    head = _ellipsoid(c, head_r)
    label[head] = LBL_CORTEX

    thickness = rng.uniform(2.5, 5.0)
    label[_ellipsoid(c, head_r - thickness)] = LBL_WM

    vent_r = rng.uniform(3.0, 6.5)
    for sign in (-1, 1):
        label[_ellipsoid(c + [sign * 4.5, 2.0, 0], [vent_r * 0.7, vent_r, vent_r * 1.4])] = LBL_VENT

    # Zielgroesse: Radius des Thalamus, unabhaengig von allem anderen gezogen
    thal_r = rng.uniform(3.0, 7.0)
    for sign in (-1, 1):
        label[_ellipsoid(c + [sign * (thal_r + 0.6), -2.0, 0],
                         [thal_r, thal_r * 1.15, thal_r * 0.9])] = LBL_THAL

    base = {LBL_BG: 0.0, LBL_CORTEX: 0.45, LBL_WM: 0.80, LBL_VENT: 0.06, LBL_THAL: 0.62}
    vol = np.zeros((SIZE,) * 3, np.float32)
    for lbl, val in base.items():
        if lbl == LBL_BG:
            continue
        vol[label == lbl] = val
    gain = rng.uniform(0.9, 1.1)
    vol *= gain
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
    rng = np.random.default_rng(0)
    vols, labs = [], []
    for _ in range(20):
        v, l = make_phantom(rng)
        vols.append(v)
        labs.append(l)
    print('20 phantoms in %.1fs' % (time.time() - t))
    labs = np.asarray(labs)
    y = (labs == LBL_THAL).sum((1, 2, 3)) / 100.
    n_brain = (labs != LBL_BG).sum((1, 2, 3))
    n_vent = (labs == LBL_VENT).sum((1, 2, 3))
    print('y (thal vol /100):', np.round(y, 2))
    print('brain:', n_brain.min(), n_brain.max())
    print('corr(y, brain) = %.3f' % np.corrcoef(y, n_brain)[0, 1])
    print('corr(y, vent)  = %.3f' % np.corrcoef(y, n_vent)[0, 1])
    print('thal share of brain: %.2f%% - %.2f%%' %
          (100 * (y * 100 / n_brain).min(), 100 * (y * 100 / n_brain).max()))
