"""Prototyp: LRP auf dem Phantom-Modell + Ground-Truth-Metriken."""
import numpy as np
import tensorflow as tf

for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

from tensorflow.keras.models import load_model, clone_model
from explainability import LRP, LRPStrategy
from proto_train import make_dataset, LBL_THAL, LBL_BG, LBL_VENT, LBL_WM, LBL_CORTEX  # noqa

model = load_model('failed_tests/proto_phantom_model.keras')
X, L, y = make_dataset(24, 7)
print('test y', np.round(y[:8], 2))
print('pred  ', np.round(model.predict(X[:8], verbose=0).ravel(), 2))


def metrics(R, target, brain):
    Rp = np.maximum(R, 0)
    tot = Rp[brain].sum()
    n_t, n_b = int(target.sum()), int(brain.sum())
    mass = Rp[target].sum() / tot if tot > 0 else np.nan
    share = n_t / n_b
    scores = np.where(brain, np.abs(R), -np.inf).ravel()
    top = np.argpartition(-scores, n_t)[:n_t]
    return dict(mass=mass, share=share, ratio=mass / share,
                prec=float(target.ravel()[top].mean()),
                hit=bool(target.ravel()[int(np.argmax(scores))]))


for name, strategy in [
    ('flat+ab',  LRPStrategy(layers=[{'flat': True}, {'alpha': 2, 'beta': 1},
                                     {'alpha': 2, 'beta': 1}, {'alpha': 2, 'beta': 1},
                                     {'alpha': 2, 'beta': 1}, {'epsilon': 0.25}])),
    ('all-ab',   LRPStrategy(layers=[{'alpha': 2, 'beta': 1}] * 5 + [{'epsilon': 0.25}])),
    ('all-eps',  LRPStrategy(layers=[{'epsilon': 0.25}] * 6)),
]:
    lrp = LRP(clone_model(model), layer=len(model.layers) - 1, idx=0, strategy=strategy)
    lrp.set_weights([]) if False else None
    # clone_model verliert Gewichte -> neu setzen ueber ein frisches LRP auf dem Original
    lrp = LRP(model, layer=len(model.layers) - 1, idx=0, strategy=strategy)
    rows = []
    for i in range(8):
        R = lrp.predict(X[i:i + 1], verbose=0)[0]
        brain = L[i] != LBL_BG
        R = R * brain
        rows.append(metrics(R, L[i] == LBL_THAL, brain))
        if i == 0:
            print(f'  [{name}] sumR={R.sum():.3f} pred={float(model.predict(X[i:i+1], verbose=0)):.2f} y={y[i]:.2f}')
    agg = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
    print('%-8s mass=%.3f share=%.3f ratio=%5.1f prec=%.3f hit=%.2f' %
          (name, agg['mass'], agg['share'], agg['ratio'], agg['prec'], agg['hit']))

# Kontrolle: randomisierte Gewichte
rnd = clone_model(model)
lrp_rnd = LRP(rnd, layer=len(model.layers) - 1, idx=0,
              strategy=LRPStrategy(layers=[{'flat': True}] + [{'alpha': 2, 'beta': 1}] * 4 + [{'epsilon': 0.25}]))
rows = []
for i in range(8):
    R = lrp_rnd.predict(X[i:i + 1], verbose=0)[0] * (L[i] != LBL_BG)
    rows.append(metrics(R, L[i] == LBL_THAL, L[i] != LBL_BG))
agg = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
print('%-8s mass=%.3f share=%.3f ratio=%5.1f prec=%.3f hit=%.2f' %
      ('random', agg['mass'], agg['share'], agg['ratio'], agg['prec'], agg['hit']))
