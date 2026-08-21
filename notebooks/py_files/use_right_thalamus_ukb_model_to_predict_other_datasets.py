# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: notebooks/ipynb_files//ipynb,notebooks/py_files//py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # UKB Right-Thalamus-Modell → Vorhersage auf anderen Datensätzen
#
# Lädt ein auf UKB trainiertes `Right-Whole_thalamus`-Modell und bewertet es auf
# einem ausgewählten Datensatz unter `/mnt/ceph/data` (jeweils die ersten
# `N_SUBJECTS` Subjekte).

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

# Inline-Backend *vor* dem pyplot-Import setzen (sonst bleibt oft Agg aktiv
# und plt.show() zeigt im Notebook nichts, obwohl savefig die PNG schreibt).
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display
from omegaconf import OmegaConf, open_dict
from scipy.stats import pearsonr

# %% [markdown]
# ## Konfiguration
#
# `DATASET` wählen: `ixi`, `ds003114_OvarianHormones` oder `ds004711_AgeRisk`.

# %%
RUN_DIR = Path(
    "~/data/nn-trainings/mri/Right-Whole_thalamus/"
    "training_run_21h19m18s_20aug2026"
    # "training_run_07h44m40s_01apr2026"
).expanduser().resolve()

DATA_ROOT = Path("/mnt/ceph/data")
DATASETS = {
    "ixi": DATA_ROOT / "ixi",
    "ds003114_OvarianHormones": DATA_ROOT / "ds003114_OvarianHormones",
    "ds004711_AgeRisk": DATA_ROOT / "ds004711_AgeRisk",
}

DATASET = "ixi"  # <- hier umschalten
N_SUBJECTS = 50
PRED_BATCH_SIZE = 8

MODEL_PATH = RUN_DIR / "model.keras"
CONFIG_PATH = RUN_DIR / "config.yaml"

if DATASET not in DATASETS:
    raise ValueError(f"Unbekanntes DATASET={DATASET!r}. Erlaubt: {list(DATASETS)}")
if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(CONFIG_PATH)

print("RUN_DIR: ", RUN_DIR)
print("DATASET:", DATASET, "→", DATASETS[DATASET])

# %% [markdown]
# ## Pfade / Imports (keras-explainability + pybrainmetrics)

# %%
def _first_existing_dir(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_dir():
            return p.resolve()
    return None


def find_keras_xai_root() -> Path:
    p = Path.cwd().resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "explainability").is_dir() and (
            (candidate / "pyproject.toml").exists() or (candidate / "pixi.toml").exists()
        ):
            return candidate
    env = os.environ.get("KERAS_XAI_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    raise FileNotFoundError(
        "keras-explainability-Root nicht gefunden. "
        "Notebook aus dem Repo starten oder KERAS_XAI_ROOT setzen."
    )


def find_pybrainmetrics_src() -> Path:
    env = os.environ.get("PYBRAINMETRICS_SRC")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            Path("~/git-repos/pyment-and1/src").expanduser(),
            Path("~/git-repos/pyment-public/src").expanduser(),
            Path("/mnt/users/andreasre/git-repos/pyment-and1/src"),
            Path("/mnt/users/andreasre/git-repos/pyment-public/src"),
        ]
    )
    src = _first_existing_dir(candidates)
    if src is None or not (src / "pybrainmetrics").is_dir():
        raise ModuleNotFoundError(
            "pybrainmetrics nicht gefunden. PYBRAINMETRICS_SRC setzen oder "
            "pyment-and1/pyment-public klonen.\n"
            f"Geprüft: {candidates}"
        )
    return src


keras_xai_root = find_keras_xai_root()
pybm_src = find_pybrainmetrics_src()
for p in (keras_xai_root, pybm_src):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from pybrainmetrics.data.dataset import build_prediction_csv
from pybrainmetrics.modeling.predict import run_predictions
from pybrainmetrics.modeling.train import build_model

print("keras-xai:     ", keras_xai_root)
print("pybrainmetrics:", pybm_src)
print("TensorFlow:    ", tf.__version__)

# %% [markdown]
# ## Label-Tabelle laden (erste `N_SUBJECTS` Subjekte)

# %%
def _normalize_label_columns(df: pd.DataFrame, pred_var: str) -> pd.DataFrame:
    df = df.copy()
    if "filepath" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path": "filepath"})
    if "participant_id" not in df.columns and "subject-id" in df.columns:
        df["participant_id"] = df["subject-id"]
    if "subject-id" not in df.columns and "participant_id" in df.columns:
        df["subject-id"] = df["participant_id"]
    if "Subject" in df.columns and "participant_id" not in df.columns:
        df["participant_id"] = df["Subject"]
        df["subject-id"] = df["Subject"]
    missing = [c for c in ("filepath", "participant_id", pred_var) if c not in df.columns]
    if missing:
        raise ValueError(f"Spalten fehlen: {missing}. Vorhanden: {list(df.columns)}")
    return df


def load_dataset_labels(dataset_dir: Path, pred_var: str, n_subjects: int) -> pd.DataFrame:
    """Bevorzugt subjects_dl_input.tsv; sonst Aufbau aus FreeSurfer-Stats + cropped.nii.gz."""
    for name in ("subjects_dl_input.tsv", "participants_dl_input.tsv"):
        cand = dataset_dir / name
        if cand.is_file():
            df = pd.read_csv(cand, sep=None, engine="python")
            df = _normalize_label_columns(df, pred_var)
            df = df.dropna(subset=[pred_var, "filepath"]).head(int(n_subjects))
            print(f"Quelle: {cand}, n={len(df)}")
            return df

    vol_path = dataset_dir / "T1stats" / "ThalamicNuclei.volumes.txt_concat.stats"
    if not vol_path.is_file():
        raise FileNotFoundError(
            f"Keine Label-TSV und keine Thalamus-Stats unter {dataset_dir}"
        )
    vols = pd.read_csv(vol_path, sep=r"\s+")
    rows = []
    for _, row in vols.iterrows():
        sid = str(row["Subject"])
        cropped = dataset_dir / "recon" / sid / "mri" / "cropped.nii.gz"
        if not cropped.is_file():
            continue
        rows.append(
            {
                "filepath": str(cropped),
                "path": str(cropped),
                "subject-id": sid,
                "participant_id": sid,
                "Left-Whole_thalamus": float(row["Left-Whole_thalamus"]),
                "Right-Whole_thalamus": float(row["Right-Whole_thalamus"]),
            }
        )
    if not rows:
        raise FileNotFoundError(f"Keine cropped.nii.gz zu Stats in {dataset_dir}")
    df = pd.DataFrame(rows)
    df = _normalize_label_columns(df, pred_var)
    df = df.dropna(subset=[pred_var, "filepath"]).head(int(n_subjects))
    print(f"Quelle: {vol_path} (+ cropped.nii.gz), n={len(df)}")
    return df


cfg = OmegaConf.load(CONFIG_PATH)
pred_var = cfg.data.prediction_variable

df = load_dataset_labels(DATASETS[DATASET], pred_var, N_SUBJECTS)
labels_tsv = RUN_DIR / f"{DATASET}_predict_labels_n{len(df)}.tsv"
df.to_csv(labels_tsv, sep="\t", index=False)

with open_dict(cfg):
    cfg.paths.csv_dir = str(RUN_DIR)
    cfg.data.predict_labels_file = str(labels_tsv)
    if "prediction" not in cfg.training:
        cfg.training.prediction = {}
    cfg.training.prediction.batch_size = int(PRED_BATCH_SIZE)

print("pred_var:", pred_var)
print("labels:  ", labels_tsv)

# %% [markdown]
# ## Modell bauen und Gewichte laden

# %%
model = build_model(cfg)

w_before = model.get_weights()[0].copy()
status = model.load_weights(str(MODEL_PATH))
w_after = model.get_weights()[0]
delta = float(np.mean(np.abs(w_after - w_before)))
print(f"load_weights status: {status}")
print(f"Gewichtsdelta (sollte >> 0): {delta:.6g}")
if delta < 1e-9:
    raise RuntimeError("Gewichte wurden nicht geladen — Layer-Namen prüfen.")

z = np.zeros((1, 167, 212, 160, 1), dtype="float32")
r = np.random.randn(1, 167, 212, 160, 1).astype("float32") * 50
p_z = float(model.predict(z, verbose=0).squeeze())
p_r = float(model.predict(r, verbose=0).squeeze())
print(f"pred zeros:  {p_z:.1f}")
print(f"pred random: {p_r:.1f}")
if abs(p_z - p_r) < 1.0:
    raise RuntimeError("Modell ignoriert Input — Gewichte/Architektur passen nicht.")

# %% [markdown]
# ## Vorhersagen

# %%
predict_csv = build_prediction_csv(cfg)
print("predict_csv:", predict_csv)

preds_df = run_predictions(model, predict_csv, cfg)

out_tsv = RUN_DIR / f"participants_predict_{DATASET}_n{len(preds_df)}.tsv"
preds_df.to_csv(out_tsv, sep="\t", index=False)
print(preds_df.head())
print(
    "prediction min/max/mean:",
    float(preds_df["prediction"].min()),
    float(preds_df["prediction"].max()),
    float(preds_df["prediction"].mean()),
)
print("geschrieben:", out_tsv)

# %% [markdown]
# ## Korrelation True vs. Pred

# %%
y_true = preds_df[pred_var].astype(float).to_numpy()
y_pred = preds_df["prediction"].astype(float).to_numpy()

r_val, _ = pearsonr(y_true, y_pred)
mae = float(np.mean(np.abs(y_true - y_pred)))
print(f"{DATASET} n={len(preds_df)}   Pearson r={r_val:.4f}   MAE={mae:.1f}")

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="none")
lo = float(min(y_true.min(), y_pred.min()))
hi = float(max(y_true.max(), y_pred.max()))
ax.plot([lo, hi], [lo, hi], "k--", lw=1)
ax.set_xlabel(f"true {pred_var}")
ax.set_ylabel("prediction")
ax.set_title(f"{DATASET}  r={r_val:.3f}  MAE={mae:.1f}")
ax.set_aspect("equal", adjustable="box")
fig.tight_layout()
scatter_path = RUN_DIR / f"scatter_true_vs_pred_{DATASET}_n{len(preds_df)}.png"
fig.savefig(scatter_path, dpi=120)
print("Scatter:", scatter_path)
display(fig)  # zuverlässiger als plt.show() unter Agg / in VS Code–Jupyter
plt.close(fig)
