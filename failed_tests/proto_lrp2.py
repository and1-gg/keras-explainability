"""Prototyp 2: Kompartiment-Aufschluesselung + dilatierte Maske + Distanzprofil."""
import numpy as np
import tensorflow as tf
from scipy import ndimage

for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

from tensorflow.keras.models import load_model, clone_model
from explainability import LRP, LRPStrategy
from proto_train import make_dataset, LBL_BG, LBL_CORTEX, LBL_WM, LBL_VENT, LBL_THAL

NAMES = {LBL_CORTEX: 'Kortex', LBL_WM: 'Weisse Substanz',
         LBL_VENT: 'Ventrikel', LBL_THAL: 'Thalamus'}

model = load_model('failed_tests/proto_phantom_model.keras')
X, L, y = make_dataset(12, 7)

strategy = LRPStrategy(layers=[{'flat': True}] + [{'alpha': 2, 'beta': 1}] * 4 + [{'epsilon': 0.25}])
lrp = LRP(model, layer=len(model.layers) - 1, idx=0, strategy=strategy)

comp_mass, comp_share, dens = {k: [] for k in NAMES}, {k: [] for k in NAMES}, {k: [] for k in NAMES}
dil_mass = {2: [], 4: [], 6: []}
prof = []
hits, precs, masses = [], [], []

for i in range(12):
    R = lrp.predict(X[i:i + 1], verbose=0)[0]
    brain = L[i] != LBL_BG
    R = R * brain
    Rp = np.maximum(R, 0)
    tot = Rp[brain].sum()
    thal = L[i] == LBL_THAL
    for k in NAMES:
        m = L[i] == k
        comp_mass[k].append(Rp[m].sum() / tot)
        comp_share[k].append(m.sum() / brain.sum())
        dens[k].append(Rp[m].mean())
    dist = ndimage.distance_transform_edt(~thal)
    for r in dil_mass:
        dil_mass[r].append(Rp[(dist <= r) & brain].sum() / tot)
    edges = [0, 1, 2, 4, 8, 16, 999]
    prof.append([Rp[(dist > a) & (dist <= b) & brain].sum() / tot
                 for a, b in zip(edges[:-1], edges[1:])])
    n_t = int(thal.sum())
    scores = np.where(brain, np.abs(R), -np.inf).ravel()
    top = np.argpartition(-scores, n_t)[:n_t]
    precs.append(thal.ravel()[top].mean())
    hits.append(bool(thal.ravel()[int(np.argmax(scores))]))
    masses.append(Rp[thal].sum() / tot)

print('\n%-16s %8s %8s %8s' % ('Kompartiment', 'R-Anteil', 'Vol-Ant', 'Ratio'))
for k in NAMES:
    m, s = np.mean(comp_mass[k]), np.mean(comp_share[k])
    print('%-16s %7.3f  %7.3f  %7.2f' % (NAMES[k], m, s, m / s))

print('\nDilatierte Thalamusmaske (Relevanzanteil):')
print('  exakt      %.3f' % np.mean(masses))
for r in sorted(dil_mass):
    print('  +%d Voxel   %.3f' % (r, np.mean(dil_mass[r])))

print('\nRelevanz nach Abstand zum Thalamus (mm):')
labels = ['0 (innen)', '0-1', '1-2', '2-4', '4-8', '8-16', '>16']
p = np.mean(prof, axis=0)
for lab, v in zip(labels, p):
    print('  %-10s %.3f' % (lab, v))

print('\nPointing game: %.2f   Top-k-Praezision: %.3f   Masse: %.3f' %
      (np.mean(hits), np.mean(precs), np.mean(masses)))
print('sum profile', p.sum())
