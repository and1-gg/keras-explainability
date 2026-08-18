"""Prototyp fuer das Thalamus-XAI-Notebook. Nur zum Ermitteln echter Zahlen."""
import os
import numpy as np
import nibabel as nib
import tensorflow as tf

from nibabel.processing import resample_from_to

REPO = '/home/and1/git-repos/keras-explainability'
IMG_DIR = os.path.join(REPO, 'data/mri/ixi/cropped/images')
FS_DIR = os.path.join(REPO, 'data/mri/ixi/fastsurfer')
W = os.path.join(REPO, 'output/pyment/models/regression_sfcn_reg_2025_weights.h5')

THALAMUS_LABELS = (10, 49)

ids = sorted(p[:-len('.nii.gz')] for p in os.listdir(IMG_DIR) if p.endswith('.nii.gz'))
print('ids', ids)


def load(subject):
    img = nib.load(os.path.join(IMG_DIR, f'{subject}.nii.gz'))
    aseg = nib.load(os.path.join(FS_DIR, subject, 'mri', 'aparc.DKTatlas+aseg.deep.mgz'))
    seg = resample_from_to(aseg, img, order=0).get_fdata()
    return img.get_fdata(), seg


def shuffle_outside(volume, thal, rng):
    out = volume.copy()
    sel = (~thal) & (volume != 0)
    vals = out[sel]
    rng.shuffle(vals)
    out[sel] = vals
    return out


def metrics(R, thal, brain):
    """Ground-truth-Metriken fuer eine Relevanzkarte."""
    Rp = np.maximum(R, 0)
    n_t, n_b = int(thal.sum()), int(brain.sum())
    mass = Rp[thal].sum() / Rp[brain].sum()
    share = n_t / n_b
    dens_t = Rp[thal].sum() / n_t
    dens_b = Rp[brain].sum() / n_b
    flat = np.where(brain, np.abs(R), -np.inf)
    top = np.argsort(flat.ravel())[::-1][:n_t]
    prec = thal.ravel()[top].mean()
    hit = bool(thal.ravel()[int(np.argmax(flat))])
    return dict(mass=mass, share=share, ratio=mass / share,
                dens_t=dens_t, dens_b=dens_b, prec=prec, hit=hit)


# ------------------------------------------------------------------ Masken
stats = []
for s in ids:
    vol, seg = load(s)
    thal = np.isin(seg, THALAMUS_LABELS)
    brain = vol != 0
    stats.append((s, int(thal.sum()), int(brain.sum()), 100 * thal.sum() / brain.sum(),
                  int((thal & ~brain).sum())))
for row in stats:
    print('%-10s thal=%6d brain=%8d  %.2f%%  thal-outside-brain=%d' % row)

# ------------------------------------------------------------------ Modell (CPU)
from pyment.models import RegressionSFCN
from explainability import LRP, LRPStrategy

with tf.device('/CPU:0'):
    model = RegressionSFCN(weights=W)
    strategy = LRPStrategy(layers=[{'flat': True}, {'flat': True},
                                   {'alpha': 2, 'beta': 1}, {'alpha': 2, 'beta': 1},
                                   {'alpha': 2, 'beta': 1}, {'alpha': 2, 'beta': 1},
                                   {'epsilon': 0.25}])
    lrp = LRP(model, layer=len(model.layers) - 1, idx=0, strategy=strategy)

    rng = np.random.default_rng(42)
    print('\n%-10s %-9s %7s %7s | %6s %6s %6s %6s %5s' %
          ('id', 'variant', 'pred', 'sumR', 'mass', 'share', 'ratio', 'prec', 'hit'))
    for s in ids[:4]:
        vol, seg = load(s)
        thal = np.isin(seg, THALAMUS_LABELS)
        brain = vol != 0
        variants = {'original': vol, 'shuffled': shuffle_outside(vol, thal, rng)}
        for name, v in variants.items():
            X = (v / 255.)[None].astype('float32')
            pred = float(model.predict(X, verbose=0).ravel()[0])
            R = lrp.predict(X, verbose=0)[0] * brain
            m = metrics(R, thal, brain)
            print('%-10s %-9s %7.2f %7.3f | %6.4f %6.4f %6.2f %6.4f %5s' %
                  (s, name, pred, R.sum(), m['mass'], m['share'], m['ratio'],
                   m['prec'], m['hit']))
