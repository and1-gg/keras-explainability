# ---
# jupyter:
#   jupytext:
#     formats: notebooks/ipynb_files//ipynb,notebooks/py_files//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: py-uv_keras-xai (uv)
#     language: python
#     name: py-uv_keras-xai
# ---

# %% [markdown]
# # LRP-Heatmaps: drei Right-Thalamus-UKB-Modelle
#
# Vergleich von drei auf UKB trainierten `Right-Whole_thalamus`-CNNs:
#
# 1. **unveränderte** MRI-Scans (`cropped.nii.gz`)
# 2. **gejitterte** Scans (rechter Thalamus erhalten, Rest permutiert)
# 3. **all-zero** außer rechtem Thalamus
#
# Ablauf: Vorhersagen für `N_SUBJ_PRED` Holdout-Subjects → Scatter/MAE/r →
# für Subject `IDX_PRED` Input-Schnitte + neu berechnete unnormierte LRP-Heatmaps →
# Gruppen-LRP über die ersten `N_GROUP` Subjects je Holdout-TSV (Sum-Norm) →
# CPU- vs. GPU-Vergleich der LRP-Heatmaps (Abschnitt M) →
# Folgeanalysen zur Device-Diskrepanz (Abschnitt N).

# %% [markdown]
# ## A. Imports

# %%
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# %matplotlib inline

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display
from matplotlib.patches import Patch
from omegaconf import OmegaConf, open_dict
from scipy.stats import pearsonr
from tqdm import tqdm


# %% [markdown]
# ## B. Konfiguration
#
# - `N_SUBJ_PRED`: Anzahl Holdout-Subjects für Vorhersage / Scatter (erste Zeilen
#   der jeweiligen `predict.tsv`, Subject-IDs über alle drei Varianten gematcht).
# - `IDX_PRED`: Index des Subjects für die 2×3-Figur; muss `0 <= IDX_PRED < N_SUBJ_PRED`
#   erfüllen.
# - `N_GROUP`: Anzahl Subjects je Holdout-TSV für die Gruppen-LRP (Sum-Norm,
#   Abschnitte K–L); jeweils die **ersten** `N_GROUP` Zeilen der jeweiligen TSV.

# %%
N_SUBJ_PRED = 3
IDX_PRED = 1
N_GROUP = 50

PRED_BATCH_SIZE = 8
SAGITTAL_X = 70
SHOW_PLOTS_INLINE = True
COLOR_RIGHT = (0.15, 0.65, 0.25, 0.50)  # grün, alpha=0.5

if not (0 <= int(IDX_PRED) < int(N_SUBJ_PRED)):
    raise ValueError(
        f"IDX_PRED={IDX_PRED} muss kleiner als N_SUBJ_PRED={N_SUBJ_PRED} sein "
        f"(gültig: 0 .. {N_SUBJ_PRED - 1})."
    )
if int(N_GROUP) < 1:
    raise ValueError(f"N_GROUP muss >= 1 sein, got {N_GROUP}.")

# (1) unveränderte MRI
RUN_DIR_NORMAL = Path(
    "/mnt/ceph2/dl_project/data/nn-trainings/mri/Right-Whole_thalamus/"
    "training_run_21h19m18s_20aug2026"
).resolve()
PREDICT_TSV_NORMAL = Path(
    "/mnt/users/andreasre/git-repos/pyment-and1/training_runs/input_files/mri/"
    "right_whole_thalamus/predict.tsv"
).resolve()
MASK_TMPL_NORMAL = (
    "/mnt/ceph2/dl_project/data/mri-scans/not_altered/ukb/recon/"
    "{sid}/mri/aseg_mni152_right_thalamus_cropped.nii.gz"
)

# (2) gejittert (rechter Thalamus erhalten)
RUN_DIR_JITTER = Path(
    "/mnt/ceph2/dl_project/data/nn-trainings/mri/Right-Whole_thalamus/"
    "training_run_05h09m52s_04sep2026"
).resolve()
PREDICT_TSV_JITTER = Path(
    "/mnt/users/andreasre/git-repos/pyment-and1/training_runs/input_files/mri/"
    "right_whole_thalamus/jittered_data/predict.tsv"
).resolve()
MASK_TMPL_JITTER = (
    "/mnt/ceph2/dl_project/data/mri-scans/jittered_data/ukb/recon/"
    "{sid}/mri/aseg_mni152_right_thalamus_cropped.nii.gz"
)

# (3) all-zero außer rechtem Thalamus
RUN_DIR_ALL_ZERO = Path(
    "/mnt/ceph2/dl_project/data/nn-trainings/mri/Right-Whole_thalamus/"
    "training_run_21h19m09s_09sep2026"
).resolve()
PREDICT_TSV_ALL_ZERO = Path(
    "/mnt/users/andreasre/git-repos/pyment-and1/training_runs/input_files/mri/"
    "right_whole_thalamus/all_zero_except_right_thalamus/predict.tsv"
).resolve()
MASK_TMPL_ALL_ZERO = (
    "/mnt/ceph2/dl_project/data/mri-scans/only_brain_regions/ukb/recon/"
    "{sid}/mri/aseg_mni152_right_thalamus_cropped.nii.gz"
)

DATASETS = {
    "normal": {
        "label": "(1) unverändert",
        "run_dir": RUN_DIR_NORMAL,
        "predict_tsv": PREDICT_TSV_NORMAL,
        "mask_tmpl": MASK_TMPL_NORMAL,
        "mask_fallbacks": [
            "/mnt/ceph2/dl_project/data/mri-scans/not_altered/ukb/recon/"
            "{sid}/mri/aseg_mni152_right_thalamus.nii.gz",
        ],
    },
    "jittered": {
        "label": "(2) gejittert",
        "run_dir": RUN_DIR_JITTER,
        "predict_tsv": PREDICT_TSV_JITTER,
        "mask_tmpl": MASK_TMPL_JITTER,
        "mask_fallbacks": [],
    },
    "all_zero": {
        "label": "(3) all-zero außer rechtem Thalamus",
        "run_dir": RUN_DIR_ALL_ZERO,
        "predict_tsv": PREDICT_TSV_ALL_ZERO,
        "mask_tmpl": MASK_TMPL_ALL_ZERO,
        "mask_fallbacks": [
            "/mnt/ceph2/dl_project/data/mri-scans/only_brain_regions/ukb/recon/"
            "{sid}/mri/aseg_mni152_right_thalamus.nii.gz",
            "/mnt/ceph2/dl_project/data/mri-scans/only_brain_regions/right-thalamus/"
            "ukb/recon/{sid}/mri/aseg_mni152_right_thalamus_cropped.nii.gz",
        ],
    },
}

for key, meta in DATASETS.items():
    model_path = meta["run_dir"] / "model.keras"
    cfg_path = meta["run_dir"] / "config.yaml"
    for p, name in (
        (model_path, f"{key} model.keras"),
        (cfg_path, f"{key} config.yaml"),
        (meta["predict_tsv"], f"{key} predict.tsv"),
    ):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{name} fehlt: {p}")

print(f"N_SUBJ_PRED = {N_SUBJ_PRED}")
print(f"IDX_PRED    = {IDX_PRED}")
print(f"N_GROUP     = {N_GROUP}")
for key, meta in DATASETS.items():
    print(f"[{key}] run={meta['run_dir'].name}")
    print(f"        tsv={meta['predict_tsv']}")


# %% [markdown]
# ## C. Repo-Pfade (`keras-explainability` + `pyment-and1`)
#
# Zur Laufzeit wird Code aus beiden Repos benötigt. `sys.path` wird um die
# keras-explainability-Root und `pyment-and1/src` (→ `pybrainmetrics`) ergänzt.

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
            Path("/mnt/users/andreasre/git-repos/pyment-and1/src"),
            Path("~/git-repos/pyment-and1/src").expanduser(),
            Path("~/git-repos/pyment-public/src").expanduser(),
            Path("/mnt/users/andreasre/git-repos/pyment-public/src"),
        ]
    )
    src = _first_existing_dir(candidates)
    if src is None or not (src / "pybrainmetrics").is_dir():
        raise ModuleNotFoundError(
            "pybrainmetrics nicht gefunden (pyment-and1/src). "
            f"Geprüft: {candidates}"
        )
    return src


keras_xai_root = find_keras_xai_root()
pybm_src = find_pybrainmetrics_src()
for p in (keras_xai_root, pybm_src):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from pybrainmetrics.data.dataset import (  # noqa: E402
    _load_single_volume,
    _load_single_volume_native,
)
from pybrainmetrics.modeling.train import _build_single_device_model  # noqa: E402
from explainability import LRP, LRPStrategy  # noqa: E402

print("keras-xai:     ", keras_xai_root)
print("pybrainmetrics:", pybm_src)
print("sys.path[0:2]: ", sys.path[:2])
print("TensorFlow:    ", tf.__version__)


# %% [markdown]
# ## D. Labels laden und Subject-Pool festlegen
#
# Die drei `predict.tsv` enthalten dieselben Subject-IDs in unterschiedlicher
# Reihenfolge. Für den Vergleich werden die ersten `N_SUBJ_PRED` IDs aus der
# **normalen** Holdout-TSV genommen und in den anderen TSVs nachgeschlagen.

# %%
PRED_VAR = "Right-Whole_thalamus"


def _normalize_label_columns(df: pd.DataFrame, pred_var: str) -> pd.DataFrame:
    df = df.copy()
    if "filepath" not in df.columns and "path" in df.columns:
        df = df.rename(columns={"path": "filepath"})
    if "participant_id" not in df.columns and "subject-id" in df.columns:
        df["participant_id"] = df["subject-id"]
    if "subject-id" not in df.columns and "participant_id" in df.columns:
        df["subject-id"] = df["participant_id"]
    missing = [c for c in ("filepath", "participant_id", pred_var) if c not in df.columns]
    if missing:
        raise ValueError(f"Spalten fehlen: {missing}. Vorhanden: {list(df.columns)}")
    return df


def load_predict_tsv(tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep=None, engine="python")
    return _normalize_label_columns(df, PRED_VAR)


def tsv_lookup_by_subject(df: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        out[str(row["participant_id"])] = row
    return out


labels_full = {key: load_predict_tsv(meta["predict_tsv"]) for key, meta in DATASETS.items()}
lookup = {key: tsv_lookup_by_subject(df) for key, df in labels_full.items()}

subject_ids = (
    labels_full["normal"]["participant_id"].astype(str).head(int(N_SUBJ_PRED)).tolist()
)
missing_ids = [
    sid
    for sid in subject_ids
    if not all(sid in lookup[k] for k in DATASETS)
]
if missing_ids:
    raise RuntimeError(f"Subject-IDs fehlen in mind. einer TSV: {missing_ids}")

subject_id_plot = subject_ids[int(IDX_PRED)]
print(f"Subject-Pool (n={len(subject_ids)}): {subject_ids}")
print(f"Plot-Subject IDX_PRED={IDX_PRED}: {subject_id_plot}")


# %% [markdown]
# ## E. Modelle, Volume-Loader und LRP-Strategie
#
# Pro Dataset-Variante wird das zugehörige `model.keras` geladen. LRP-Composite
# für SFCN: zwei `flat`-Schichten, vier αβ-Schichten, ε am Dense-Ausgang.
# Heatmaps bleiben **unnormiert** und werden immer neu berechnet (kein Laden).

# %%
def load_keras_model(run_dir: Path):
    cfg = OmegaConf.load(run_dir / "config.yaml")
    with open_dict(cfg):
        cfg.paths.csv_dir = str(run_dir)
        if "prediction" not in cfg.training:
            cfg.training.prediction = {}
        cfg.training.prediction.batch_size = int(PRED_BATCH_SIZE)
    model = _build_single_device_model(cfg)
    w0 = model.get_weights()[0].copy()
    model.load_weights(str(run_dir / "model.keras"))
    delta = float(np.mean(np.abs(model.get_weights()[0] - w0)))
    if delta < 1e-9:
        raise RuntimeError(f"Gewichte nicht geladen: {run_dir}")
    return model, cfg, delta


def make_load_volume(cfg):
    norm = float(cfg.preprocessing.normalization_factor)
    loader = str(getattr(cfg.data, "loader", "nifti-nibabel")).lower()

    def _load(path: str) -> np.ndarray:
        if loader == "nifti-native":
            vol = _load_single_volume_native(path, norm)
        else:
            vol = _load_single_volume(path, norm)
        if vol.ndim == 3:
            vol = np.expand_dims(vol, axis=-1)
        return vol.astype(np.float32)

    return _load


# SFCN: 5× MaxPool3D + 1× GlobalAvgPool. Flat-Pooling erhält ΣR geräteunabhängig
# (WTA über MaxPool3DGrad inflatiert auf CPU bei Ties; redistribute verwirft Nullfenster).
N_POOLING_LAYERS = 6
LRP_STRATEGY = LRPStrategy(
    layers=[
        {"flat": True},
        {"flat": True},
        {"alpha": 2, "beta": 1},
        {"alpha": 2, "beta": 1},
        {"alpha": 2, "beta": 1},
        {"alpha": 2, "beta": 1},
        {"epsilon": 0.25},
    ],
    pooling=[{"strategy": "flat"}] * N_POOLING_LAYERS,
)

models: dict[str, object] = {}
load_volume_fns: dict[str, object] = {}
for key, meta in DATASETS.items():
    model, cfg, delta = load_keras_model(meta["run_dir"])
    models[key] = model
    load_volume_fns[key] = make_load_volume(cfg)
    print(f"[{key}] geladen  Δw={delta:.3g}  loader={cfg.data.loader}")


# %% [markdown]
# ## F. Vorhersagen für `N_SUBJ_PRED` Subjects
#
# Jedes Modell prädiziert die Volumes seiner eigenen Datenvariante für denselben
# Subject-Pool. Anschließend Scatter (3 Panels) mit MAE und Pearson-r sowie eine
# Tabelle true vs. predicted.

# %%
def predict_subjects(dataset_key: str, sids: list[str]) -> pd.DataFrame:
    model = models[dataset_key]
    load_vol = load_volume_fns[dataset_key]
    rows: list[dict[str, object]] = []
    for sid in sids:
        row = lookup[dataset_key][sid]
        path = Path(str(row["filepath"]))
        y_true = float(row[PRED_VAR])
        if not path.is_file():
            raise FileNotFoundError(f"[{dataset_key}/{sid}] Volume fehlt: {path}")
        vol = load_vol(str(path))
        y_pred = float(np.squeeze(model.predict(np.expand_dims(vol, 0), verbose=0)))
        rows.append(
            {
                "dataset": dataset_key,
                "subject_id": sid,
                "filepath": str(path),
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
    return pd.DataFrame(rows)


preds_by_dataset = {
    key: predict_subjects(key, subject_ids) for key in DATASETS
}

metrics_rows: list[dict[str, object]] = []
for key, df in preds_by_dataset.items():
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    r_val = float(pearsonr(y_true, y_pred)[0]) if len(df) >= 2 else float("nan")
    mae = float(np.mean(np.abs(y_true - y_pred)))
    metrics_rows.append(
        {
            "dataset": key,
            "label": DATASETS[key]["label"],
            "n": len(df),
            "MAE": mae,
            "pearson_r": r_val,
        }
    )
    print(
        f"[{key}] n={len(df)}  MAE={mae:.1f}  r={r_val:.3f}"
    )

metrics_df = pd.DataFrame(metrics_rows)
display(metrics_df.round(3))

# Tabelle: true / predicted für alle drei Datasets
table_parts = []
for key, df in preds_by_dataset.items():
    part = df[["subject_id", "y_true", "y_pred"]].copy()
    part = part.rename(
        columns={
            "y_true": f"y_true_{key}",
            "y_pred": f"y_pred_{key}",
        }
    )
    table_parts.append(part.set_index("subject_id"))
table_df = pd.concat(table_parts, axis=1).reset_index()
print("\nTrue- vs. Predicted-Volumen (rechter Thalamus):")
display(table_df.round(1))


# %% [markdown]
# ## G. Scatter-Plot (3 Panels)
#
# Ein Panel je Datenvariante / Modell. Diagonale = ideale Vorhersage.

# %%
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
fig.suptitle(
    f"True vs. Predicted  ·  Right-Whole_thalamus  ·  n={N_SUBJ_PRED}",
    fontsize=12,
)

for ax, key in zip(axes, DATASETS):
    df = preds_by_dataset[key]
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    mae = float(metrics_df.loc[metrics_df["dataset"] == key, "MAE"].iloc[0])
    r_val = float(metrics_df.loc[metrics_df["dataset"] == key, "pearson_r"].iloc[0])
    ax.scatter(y_true, y_pred, alpha=0.85, s=55)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo + 1e-6)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
    ax.set_xlabel("true volume")
    ax.set_ylabel("predicted volume")
    ax.set_title(f"{DATASETS[key]['label']}\nMAE={mae:.1f}  r={r_val:.3f}")
    ax.set_aspect("equal", adjustable="box")
    # IDX_PRED hervorheben
    ax.scatter(
        [y_true[IDX_PRED]],
        [y_pred[IDX_PRED]],
        s=120,
        facecolors="none",
        edgecolors="C3",
        linewidths=2,
        label=f"IDX_PRED={IDX_PRED}",
    )
    ax.legend(loc="best", fontsize=8)

fig.tight_layout()
if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)


# %% [markdown]
# ## H. Input-Volumes und Thalamus-Maske für `IDX_PRED`
#
# Sagittaler Schnitt `x=70`. Maskenpfade wie spezifiziert; falls
# `…_cropped.nii.gz` fehlt, Fallback auf gleich große Alternativdateien.

# %%
def resolve_mask_path(dataset_key: str, sid: str) -> Path:
    meta = DATASETS[dataset_key]
    candidates = [Path(meta["mask_tmpl"].format(sid=sid))]
    candidates.extend(Path(t.format(sid=sid)) for t in meta["mask_fallbacks"])
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Keine Thalamus-Maske für [{dataset_key}/{sid}]. Geprüft:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32).squeeze()


def rgba_mask(mask_slc: np.ndarray, rgba: tuple[float, ...]) -> np.ndarray:
    out = np.zeros((*mask_slc.shape, 4), dtype=np.float32)
    out[mask_slc] = rgba
    return out


def sagittal_slc(vol: np.ndarray, cx: int) -> np.ndarray:
    return np.rot90(vol[cx])


plot_paths: dict[str, Path] = {}
plot_masks: dict[str, Path] = {}
plot_vols: dict[str, np.ndarray] = {}
plot_right: dict[str, np.ndarray] = {}
plot_true: dict[str, float] = {}
plot_pred: dict[str, float] = {}

for key in DATASETS:
    row = lookup[key][subject_id_plot]
    vol_path = Path(str(row["filepath"]))
    mask_path = resolve_mask_path(key, subject_id_plot)
    if not vol_path.is_file():
        raise FileNotFoundError(vol_path)
    vol = load_nifti(vol_path)
    mask = load_nifti(mask_path)
    if vol.shape != mask.shape:
        raise ValueError(
            f"Shape-Mismatch [{key}/{subject_id_plot}]: "
            f"vol {vol.shape} vs mask {mask.shape}"
        )
    pred_row = preds_by_dataset[key].set_index("subject_id").loc[subject_id_plot]
    plot_paths[key] = vol_path
    plot_masks[key] = mask_path
    plot_vols[key] = vol
    plot_right[key] = mask > 0
    plot_true[key] = float(pred_row["y_true"])
    plot_pred[key] = float(pred_row["y_pred"])
    print(f"[{key}] volume: {vol_path}")
    print(f"        mask:   {mask_path}")
    print(
        f"        true={plot_true[key]:.1f}  pred={plot_pred[key]:.1f}  "
        f"shape={vol.shape}"
    )

cx = int(np.clip(SAGITTAL_X, 0, next(iter(plot_vols.values())).shape[0] - 1))
print(f"sagittal x = {cx}")


# %% [markdown]
# ## I. LRP-Heatmaps neu berechnen (unnormiert)
#
# Für das ausgewählte Subject werden die drei Heatmaps **immer neu** berechnet
# (kein Einlesen gespeicherter NIfTIs). Anschließend Dauer aller drei Läufe.

# %%
heatmaps: dict[str, np.ndarray] = {}
t0 = time.perf_counter()
for key in DATASETS:
    model = models[key]
    load_vol = load_volume_fns[key]
    lrp = LRP(
        model,
        layer=len(model.layers) - 1,
        idx=0,
        strategy=LRP_STRATEGY,
    )
    vol = load_vol(str(plot_paths[key]))
    R = lrp(np.expand_dims(vol, 0))[0].numpy()
    heatmaps[key] = np.asarray(R, dtype=np.float32).squeeze()
    print(
        f"[{key}] LRP fertig  shape={heatmaps[key].shape}  "
        f"|R|_max={float(np.nanmax(np.abs(heatmaps[key]))):.4g}  "
        f"ΣR={float(np.sum(heatmaps[key])):.4g}"
    )
elapsed_s = time.perf_counter() - t0
print(
    f"\nDauer Berechnung aller 3 Heatmaps: {elapsed_s:.2f} s "
    f"({elapsed_s / 60.0:.2f} min)"
)


# %% [markdown]
# ## J. Figur 2×3: Inputs (oben) + LRP-Heatmaps (unten)
#
# - Obere Reihe: Input-Intensität, sagittal `x=70`, rechte Thalamus-Maske in Grün
#   (`alpha=0.5`), eigene Colorbar je Panel.
# - Untere Reihe: unnormierte LRP-Relevanzen, Colorbar je Panel mit
#   `vmin/vmax = ±P99.5(|R|)` (wie Abschnitt M) — Spitzen werden gekappt, die
#   Karten sind zwischen den Varianten besser vergleichbar. `|R|_max` bleibt
#   im Titel als echte Peak-Größe.
# - In jedem Panel: Subject-ID, wahres und prädiziertes Volumen.
#
# **Anteil im / außerhalb des rechten Thalamus** (unnormiertes LRP;
# Gesamtrelevanz $\sum_i R_i \approx$ vorhergesagtes Volumen wegen Relevanzerhaltung).
# Außerhalb: Summe über alle Voxel **nicht** in der rechten Thalamus-Maske:
#
# $$
# \%R_{\mathrm{Thal}}
# = 100 \cdot
# \frac{\sum_{i \in \mathrm{Thalamus}} R_i}{\sum_i R_i}
# \qquad
# \%R_{\mathrm{out}}
# = 100 \cdot
# \frac{\sum_{i \notin \mathrm{Thalamus}} R_i}{\sum_i R_i}
# $$

# %%
fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.6))
fig.suptitle(
    (
        f"Subject {subject_id_plot}  ·  IDX_PRED={IDX_PRED}/{N_SUBJ_PRED}  ·  "
        f"sagittal x={cx}\n"
        + r"$\%R_{\mathrm{Thal}}="
        r"100\cdot"
        r"(\sum_{i\in\mathrm{Thal}} R_i)/(\sum_i R_i)$"
        r"$\quad"
        r"\%R_{\mathrm{out}}="
        r"100\cdot"
        r"(\sum_{i\notin\mathrm{Thal}} R_i)/(\sum_i R_i)$"
        r"$\quad(\sum_i R_i\approx$ Pred.-Volumen, unnormiert$)$"
    ),
    fontsize=11,
)

keys = list(DATASETS.keys())
im_vols = []
for col, key in enumerate(keys):
    ax = axes[0, col]
    vol = plot_vols[key]
    right = plot_right[key]
    pos = vol[vol > 0]
    vmax_i = float(np.percentile(pos, 99.5)) if pos.size else 1.0
    r_slc = sagittal_slc(right.astype(np.float32), cx) > 0
    im = ax.imshow(sagittal_slc(vol, cx), cmap="gray", vmin=0.0, vmax=vmax_i)
    ax.imshow(rgba_mask(r_slc, COLOR_RIGHT), interpolation="nearest")
    ax.set_title(
        f"{DATASETS[key]['label']}\n"
        f"{subject_id_plot}\n"
        f"true={plot_true[key]:.0f}  pred={plot_pred[key]:.0f}",
        fontsize=9,
    )
    ax.axis("off")
    im_vols.append(im)

im_lrps = []
for col, key in enumerate(keys):
    ax = axes[1, col]
    heat = heatmaps[key]
    right = plot_right[key]
    sum_R = float(np.sum(heat))
    sum_R_thal = float(np.sum(heat[right]))
    # Explizit: alle Relevanzen außerhalb der rechten Thalamus-Maske aufaddieren
    sum_R_out = float(np.sum(heat[~right]))
    if abs(sum_R) > 0:
        pct_thal = 100.0 * sum_R_thal / sum_R
        pct_out = 100.0 * sum_R_out / sum_R
    else:
        pct_thal = float("nan")
        pct_out = float("nan")
    abs_max = float(np.nanmax(np.abs(heat))) or 1.0
    # Wie Abschnitt M: robuste Skala über Perzentil, nicht Peak (sonst wirkt
    # „unverändert“ flau und „gejittert“ übersteuert).
    vmax_r = float(np.nanpercentile(np.abs(heat), 99.5)) or abs_max
    r_slc = sagittal_slc(right.astype(np.float32), cx) > 0
    im = ax.imshow(
        sagittal_slc(heat, cx),
        cmap="RdBu_r",
        vmin=-vmax_r,
        vmax=vmax_r,
    )
    ax.imshow(rgba_mask(r_slc, COLOR_RIGHT), interpolation="nearest")
    ax.set_title(
        f"LRP unnormiert · {DATASETS[key]['label']}\n"
        f"{subject_id_plot}\n"
        f"true={plot_true[key]:.0f}  pred={plot_pred[key]:.0f}  "
        f"|R|_max={abs_max:.3g}\n"
        f"%Thalamus = {pct_thal:.1f}%  "
        f"(ΣR_thal={sum_R_thal:.1f} / ΣR={sum_R:.1f})\n"
        f"%Outside = {pct_out:.1f}%  "
        f"(ΣR_out={sum_R_out:.1f} / ΣR={sum_R:.1f})",
        fontsize=9,
    )
    ax.axis("off")
    im_lrps.append(im)
    print(
        f"[J/{key}] %Thalamus={pct_thal:.2f}%  %Outside={pct_out:.2f}%  "
        f"ΣR_thal={sum_R_thal:.4g}  ΣR_out={sum_R_out:.4g}  "
        f"ΣR={sum_R:.4g}  pred={plot_pred[key]:.1f}"
    )

for col, im in enumerate(im_vols):
    cbar = fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
    cbar.set_label("Intensity")
for col, im in enumerate(im_lrps):
    cbar = fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
    cbar.set_label("LRP relevance")

fig.legend(
    handles=[Patch(facecolor=COLOR_RIGHT, edgecolor="none", label="rechter Thalamus")],
    loc="lower center",
    ncol=1,
    frameon=False,
    fontsize=9,
)
fig.tight_layout(rect=[0, 0.05, 1, 0.88])

if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)

print(
    f"Heatmap-Berechnung (Abschnitt I): {elapsed_s:.2f} s für alle 3 Modelle "
    f"(Subject {subject_id_plot})."
)


# %% [markdown]
# ## K. Gruppen-LRP für `N_GROUP` Subjects (Sum-Norm)
#
# Pro Datenvariante die **ersten `N_GROUP` Zeilen** der jeweiligen Holdout-`predict.tsv`
# (unabhängig voneinander; Reihenfolge kann zwischen den TSVs differieren).
#
# Für jedes Subject $s$:
#
# $$
# R^{(s)}_{\mathrm{norm},i}
# = \frac{\bigl|R^{(s)}_i\bigr|}{\sum_j \bigl|R^{(s)}_j\bigr|}
# \qquad\Rightarrow\qquad
# \sum_i R^{(s)}_{\mathrm{norm},i} = 1
# $$
#
# Gruppenheatmap:
#
# $$
# H = \sum_{s=1}^{N_{\mathrm{GROUP}}} R^{(s)}_{\mathrm{norm}}
# $$
#
# Gruppenmaske: Summe der `aseg_mni152_right_thalamus_cropped.nii.gz`, danach
# binär halten — Voxel mit Wert 0 bleiben 0, Voxel $> 1$ werden auf 1 gesetzt.

# %%
def normalize_lrp_sum_abs(R: np.ndarray) -> np.ndarray:
    """R_norm_i = |R_i| / sum(|R|); Summe über Voxel = 1."""
    abs_r = np.abs(np.asarray(R, dtype=np.float64))
    denom = float(np.sum(abs_r))
    if denom <= 0.0:
        raise RuntimeError("LRP-Summe |R| ist 0 — Normierung nicht möglich.")
    return (abs_r / denom).astype(np.float64)


def binarize_summed_mask(mask_sum: np.ndarray) -> np.ndarray:
    """0 bleibt 0; Werte > 1 → 1 (binäre Vereinigung der Subject-Masken)."""
    out = np.asarray(mask_sum, dtype=np.float32).copy()
    out[out > 1.0] = 1.0
    return out


group_heatmaps: dict[str, np.ndarray] = {}
group_masks: dict[str, np.ndarray] = {}
group_meta: dict[str, dict[str, object]] = {}

t0_group = time.perf_counter()
for key in DATASETS:
    df_group = labels_full[key].head(int(N_GROUP))
    if len(df_group) < int(N_GROUP):
        raise RuntimeError(
            f"[{key}] TSV hat nur {len(df_group)} Zeilen, N_GROUP={N_GROUP}."
        )

    model = models[key]
    load_vol = load_volume_fns[key]
    lrp = LRP(
        model,
        layer=len(model.layers) - 1,
        idx=0,
        strategy=LRP_STRATEGY,
    )

    heat_acc: np.ndarray | None = None
    mask_acc: np.ndarray | None = None
    n_ok = 0
    first_sids: list[str] = []

    for _, row in tqdm(
        df_group.iterrows(),
        total=len(df_group),
        desc=f"group-LRP [{key}]",
    ):
        sid = str(row["participant_id"])
        vol_path = Path(str(row["filepath"]))
        if not vol_path.is_file():
            raise FileNotFoundError(f"[{key}/{sid}] Volume fehlt: {vol_path}")
        mask_path = resolve_mask_path(key, sid)

        vol = load_vol(str(vol_path))
        R = lrp(np.expand_dims(vol, 0))[0].numpy()
        R_norm = normalize_lrp_sum_abs(R)

        mask = load_nifti(mask_path)
        mask_bin = (mask > 0).astype(np.float32)

        if heat_acc is None:
            heat_acc = np.zeros_like(R_norm, dtype=np.float64)
            mask_acc = np.zeros(mask_bin.shape, dtype=np.float64)
        if R_norm.shape != heat_acc.shape:
            raise ValueError(
                f"[{key}/{sid}] Heatmap-Shape {R_norm.shape} ≠ {heat_acc.shape}"
            )
        if mask_bin.shape != mask_acc.shape:
            raise ValueError(
                f"[{key}/{sid}] Mask-Shape {mask_bin.shape} ≠ {mask_acc.shape}"
            )

        heat_acc += R_norm
        mask_acc += mask_bin
        n_ok += 1
        if len(first_sids) < 3:
            first_sids.append(sid)

    assert heat_acc is not None and mask_acc is not None
    group_mask = binarize_summed_mask(mask_acc)
    sum_H = float(np.sum(heat_acc))
    sum_H_thal = float(np.sum(heat_acc[group_mask > 0]))
    # ΣH ≈ N_GROUP; Anteil = Summe in Thalamus-Maske / Gesamtrelevanz
    pct_thal = 100.0 * sum_H_thal / sum_H if sum_H > 0 else float("nan")

    group_heatmaps[key] = heat_acc.astype(np.float32)
    group_masks[key] = group_mask
    group_meta[key] = {
        "n": n_ok,
        "sum_H": sum_H,
        "sum_H_thal": sum_H_thal,
        "pct_thal": pct_thal,
        "max_H": float(np.max(heat_acc)),
        "mask_voxels": int(np.sum(group_mask > 0)),
        "first_sids": first_sids,
    }
    print(
        f"[{key}] n={n_ok}  ΣH={sum_H:.4g} (≈ N_GROUP={N_GROUP})  "
        f"ΣH_thal={sum_H_thal:.4g}  "
        f"%Thalamus={pct_thal:.2f}%  "
        f"max(H)={group_meta[key]['max_H']:.4g}  "
        f"Masken-Voxel={group_meta[key]['mask_voxels']}  "
        f"erste IDs={first_sids}"
    )

elapsed_group_s = time.perf_counter() - t0_group
print(
    f"\nDauer Gruppen-LRP (3× N_GROUP={N_GROUP}): "
    f"{elapsed_group_s:.1f} s ({elapsed_group_s / 60.0:.2f} min)"
)


# %% [markdown]
# ## L. Figur: drei Gruppen-LRP-Karten
#
# Ein Panel je Datenvariante, sagittal `x=70`. Overlay der binären Gruppen-Thalamus-
# Maske in Grün (`alpha=0.5`). Eigene Colorbar je Panel (Werte ≥ 0 wegen Sum-Norm).
#
# **Anteil im rechten Thalamus** (bezogen auf die Gesamtrelevanz $\sum H \approx N_{\mathrm{GROUP}}$):
#
# $$
# \%R_{\mathrm{Thal}}
# = 100 \cdot
# \frac{\sum_{i \in \mathrm{Thalamus}} H_i}{\sum_i H_i}
# $$

# %%
fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4))
fig.suptitle(
    (
        f"Gruppen-LRP (Sum-Norm)  ·  $N_{{\\mathrm{{GROUP}}}}={N_GROUP}$"
        f"  ·  sagittal $x={SAGITTAL_X}$\n"
        + r"$R^{(s)}_{\mathrm{norm},i}=\dfrac{|R^{(s)}_i|}{\sum_j |R^{(s)}_j|}"
        r"\;\Rightarrow\;"
        r"\sum_i R^{(s)}_{\mathrm{norm},i}=1$"
        "\n"
        r"$H=\sum_{s=1}^{N_{\mathrm{GROUP}}} R^{(s)}_{\mathrm{norm}}$"
        r"$\qquad"
        r"\%R_{\mathrm{Thal}}="
        r"100\cdot"
        r"(\sum_{i\in\mathrm{Thal}} H_i)/(\sum_i H_i)$"
    ),
    fontsize=11,
)

im_groups = []
for ax, key in zip(axes, DATASETS):
    H = group_heatmaps[key]
    M = group_masks[key] > 0
    cx_g = int(np.clip(SAGITTAL_X, 0, H.shape[0] - 1))
    vmax_h = float(np.nanmax(H)) or 1.0
    r_slc = sagittal_slc(M.astype(np.float32), cx_g) > 0
    im = ax.imshow(
        sagittal_slc(H, cx_g),
        cmap="hot",
        vmin=0.0,
        vmax=vmax_h,
    )
    ax.imshow(rgba_mask(r_slc, COLOR_RIGHT), interpolation="nearest")
    meta = group_meta[key]
    ax.set_title(
        f"{DATASETS[key]['label']}\n"
        f"n={meta['n']}  ΣH={meta['sum_H']:.2f}  max={meta['max_H']:.3g}\n"
        f"%Thalamus = {meta['pct_thal']:.1f}%  "
        f"(ΣH_thal={meta['sum_H_thal']:.2f})",
        fontsize=10,
    )
    ax.axis("off")
    im_groups.append(im)

for ax, im in zip(axes, im_groups):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\sum_s R^{(s)}_{\mathrm{norm}}$")

fig.legend(
    handles=[
        Patch(
            facecolor=COLOR_RIGHT,
            edgecolor="none",
            label="Gruppen-Thalamus-Maske (binär)",
        )
    ],
    loc="lower center",
    ncol=1,
    frameon=False,
    fontsize=9,
)
fig.tight_layout(rect=[0, 0.08, 1, 0.82])

if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)

print(
    f"Gruppenfigur fertig (N_GROUP={N_GROUP}, Berechnung {elapsed_group_s:.1f} s)."
)


# %% [markdown]
# ## M. CPU vs. GPU: LRP-Heatmaps für `IDX_PRED`
#
# Für dasselbe Subject wie in Abschnitt I werden die unnormierten LRP-Karten
# **einmal auf CPU und einmal auf GPU** berechnet (`training=False`). Ziel: prüfen,
# ob Device/dtype die Heatmaps verzerren.
#
# ### Was rauskommen sollte
#
# | Check | Erwartung |
# |---|---|
# | `input.dtype` / `R.dtype` | auf CPU und GPU gleich, typisch `float32` |
# | GPU-Heatmap | Relevanz lokal am Gewebe / Thalamus; Hintergrund ≈ 0 (weiß bei RdBu) |
# | CPU-Heatmap | oft **ähnlich** zur GPU; bei Instabilität (Division `R/z` an Padding/Nullen) **Rahmen-/Hintergrund-Artefakte**, große `\|R\|_max`, ggf. NaN/Inf |
# | `max\|R_cpu − R_gpu\|` | bei stabilem Lauf klein relativ zu `\|R\|_max`; bei wildem CPU-Lauf groß |
# | `ΣR` | grob in der Größenordnung der Modellvorhersage (Relevanzerhaltung), sofern keine Explosion |
#
# **Interpretation:** Gleiche dtypes + ähnliche Karten → Device egal. Wilde CPU-Rahmen
# bei sauberer GPU → numerische Instabilität (nicht „falsches Modell“); für Plots
# die GPU-Variante bevorzugen. Fehlt eine GPU, wird nur CPU gerechnet und gemeldet.

# %%
def _lrp_on_device(
    model,
    vol: np.ndarray,
    device: str,
) -> tuple[np.ndarray, str, str]:
    """LRP auf `device` mit training=False; liefert (R, R.dtype, device_str)."""
    x = np.expand_dims(vol, 0).astype(np.float32)
    with tf.device(device):
        lrp = LRP(
            model,
            layer=len(model.layers) - 1,
            idx=0,
            strategy=LRP_STRATEGY,
        )
        R_t = lrp(x, training=False)
        R = np.asarray(R_t[0].numpy(), dtype=np.float32).squeeze()
        dtype_str = str(R_t.dtype)
        # physisches Device des Ergebnis-Tensors (falls verfügbar)
        try:
            dev_str = R_t.device
        except Exception:
            dev_str = device
    return R, dtype_str, str(dev_str)


gpus = tf.config.list_physical_devices("GPU")
has_gpu = len(gpus) > 0
print(f"TensorFlow {tf.__version__}")
print(f"GPUs sichtbar: {gpus if has_gpu else '(keine)'}")
print(f"Vergleich für Subject IDX_PRED={IDX_PRED}: {subject_id_plot}")
print(f"Input-Volumes dtype (erwartet float32):")
for key in DATASETS:
    print(f"  [{key}] {plot_vols[key].dtype}")

devices_to_run: list[tuple[str, str]] = [("CPU", "/CPU:0")]
if has_gpu:
    devices_to_run.append(("GPU", "/GPU:0"))
else:
    print(
        "\nKeine GPU verfügbar — nur CPU-Lauf. "
        "Erwartetes „sauberes“ Referenzbild entfällt."
    )

heatmaps_by_device: dict[str, dict[str, np.ndarray]] = {
    label: {} for label, _ in devices_to_run
}
meta_by_device: dict[str, dict[str, dict[str, object]]] = {
    label: {} for label, _ in devices_to_run
}

t0_dev = time.perf_counter()
for key in DATASETS:
    model = models[key]
    load_vol = load_volume_fns[key]
    vol = load_vol(str(plot_paths[key]))
    for label, device in devices_to_run:
        R, dtype_str, dev_str = _lrp_on_device(model, vol, device)
        heatmaps_by_device[label][key] = R
        n_nan = int(np.isnan(R).sum())
        n_inf = int(np.isinf(R).sum())
        abs_max = float(np.nanmax(np.abs(R))) if R.size else float("nan")
        sum_r = float(np.nansum(R))
        meta_by_device[label][key] = {
            "dtype": dtype_str,
            "device": dev_str,
            "abs_max": abs_max,
            "sum_R": sum_r,
            "n_nan": n_nan,
            "n_inf": n_inf,
        }
        print(
            f"[{label}/{key}] dtype={dtype_str}  device={dev_str}  "
            f"|R|_max={abs_max:.4g}  ΣR={sum_r:.4g}  "
            f"NaN={n_nan}  Inf={n_inf}"
        )
elapsed_dev_s = time.perf_counter() - t0_dev
print(f"\nDauer CPU/GPU-Vergleich: {elapsed_dev_s:.2f} s")

# Numerischer Diff CPU vs. GPU (falls GPU da)
if has_gpu:
    print("\n--- Diff CPU − GPU ---")
    rows_diff: list[dict[str, object]] = []
    for key in DATASETS:
        Rc = heatmaps_by_device["CPU"][key]
        Rg = heatmaps_by_device["GPU"][key]
        diff = Rc.astype(np.float64) - Rg.astype(np.float64)
        max_abs = float(np.nanmax(np.abs(diff)))
        rmse = float(np.sqrt(np.nanmean(diff**2)))
        scale = max(
            float(np.nanmax(np.abs(Rg))),
            float(np.nanmax(np.abs(Rc))),
            1e-12,
        )
        rel = max_abs / scale
        same_dtype = (
            meta_by_device["CPU"][key]["dtype"]
            == meta_by_device["GPU"][key]["dtype"]
        )
        rows_diff.append(
            {
                "dataset": key,
                "same_dtype": same_dtype,
                "max|Δ|": max_abs,
                "RMSE": rmse,
                "max|Δ|/max|R|": rel,
                "CPU_|R|_max": meta_by_device["CPU"][key]["abs_max"],
                "GPU_|R|_max": meta_by_device["GPU"][key]["abs_max"],
                "CPU_NaN": meta_by_device["CPU"][key]["n_nan"],
                "GPU_NaN": meta_by_device["GPU"][key]["n_nan"],
            }
        )
        status = "OK (ähnlich)" if rel < 0.05 and max_abs < 1.0 else (
            "WARNUNG (große Abweichung — typisch bei CPU-Instabilität)"
        )
        print(
            f"[{key}] same_dtype={same_dtype}  max|Δ|={max_abs:.4g}  "
            f"rel={rel:.3g}  → {status}"
        )
    display(pd.DataFrame(rows_diff))
else:
    print("Diff-Tabelle übersprungen (keine GPU).")

# Figur: Zeile 0 = CPU, Zeile 1 = GPU (falls vorhanden)
n_rows = len(devices_to_run)
fig, axes = plt.subplots(
    n_rows, 3, figsize=(14.5, 4.2 * n_rows), squeeze=False
)
fig.suptitle(
    (
        f"CPU vs. GPU LRP  ·  Subject {subject_id_plot}"
        f"  ·  sagittal $x={cx}$  ·  training=False\n"
        "Erwartung: GPU lokal/sauber; CPU ggf. Rahmen-Artefakte bei Instabilität"
    ),
    fontsize=11,
)

for row, (label, _) in enumerate(devices_to_run):
    for col, key in enumerate(DATASETS):
        ax = axes[row, col]
        R = heatmaps_by_device[label][key]
        meta = meta_by_device[label][key]
        vmax = float(np.nanpercentile(np.abs(R), 99.5)) or 1.0
        im = ax.imshow(
            sagittal_slc(R, cx),
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(
            f"{label} · {DATASETS[key]['label']}\n"
            f"dtype={meta['dtype']}  |R|_max={meta['abs_max']:.3g}\n"
            f"ΣR={meta['sum_R']:.3g}  NaN={meta['n_nan']} Inf={meta['n_inf']}",
            fontsize=9,
        )
        ax.axis("off")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("LRP relevance")

fig.tight_layout(rect=[0, 0.0, 1, 0.92])
if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)

print(
    "\nFazit-Checkliste:\n"
    "  1) dtypes CPU/GPU gleich (float32)?\n"
    "  2) GPU-Karte ohne Rahmen/Hintergrund-Rauschen?\n"
    "  3) Wenn CPU wild und GPU sauber → Device-Numerik, LRP auf GPU plotten.\n"
    "  4) Wenn beide wild → Strategie/ε bzw. Division R/z prüfen, nicht nur Device."
)


# %% [markdown]
# ## N. Folgeanalysen zur CPU/GPU-LRP-Diskrepanz
#
# Baut auf Abschnitt M auf (`heatmaps_by_device`, `meta_by_device`, `has_gpu`).
# Ziel: eingrenzen, **wo** und **warum** CPU und GPU auseinanderlaufen.
#
# | Teil | Frage | Erwartung bei eurem Befund |
# |---|---|---|
# | **N.1** | `y_pred` vs. `ΣR`? | GPU: `ΣR ≈ y_pred`; CPU: `ΣR` Größenordnungen daneben |
# | **N.2** | Ab welcher LRP-Schicht springt `ΣR`? | CPU: Sprung früh (Conv/Padding/Pool); GPU: flach |
# | **N.3** | Anteil `\|R\|` in Zero-Padding / außerhalb Maske? | CPU hoch, GPU niedrig |
# | **N.4** | Hilft größeres ε? | CPU näher an GPU → Nenner-Stabilität; sonst tiefer in αβ/`1e-9` |
# | **N.5** | Nur Forward `model(x)` CPU vs. GPU? | fast gleich → Bug im LRP-Backward |
# | **N.6** | Zwischen-Aktivierungen (Conv)? | winzige Δ → LRP verstärkt sie |
# | **N.7** | Praxis | LRP-Plots auf GPU; CPU als instabil dokumentieren |

# %%
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.layers import Lambda
from explainability.layers import PoolingLRPLayer, StandardLRPLayer
from explainability.layers.layer import LRPLayer

if "heatmaps_by_device" not in globals() or "devices_to_run" not in globals():
    raise RuntimeError(
        "Abschnitt N braucht Abschnitt M (heatmaps_by_device / devices_to_run)."
    )

ZERO_EPS = 1e-8  # Voxel mit |x| <= ZERO_EPS gelten als Null/Padding
EPS_LARGE = 10.0  # N.4: ε am Dense gegenüber 0.25 erhöhen


def _device_labels() -> list[str]:
    return [label for label, _ in devices_to_run]


def _predict_on_device(model, vol: np.ndarray, device: str) -> float:
    x = np.expand_dims(vol, 0).astype(np.float32)
    with tf.device(device):
        y = model(x, training=False)
        return float(np.squeeze(y.numpy()))


def _find_backward_start(lrp_model) -> int:
    for i, layer in enumerate(lrp_model.layers):
        if isinstance(layer, Lambda) and "output_mask" in layer.name:
            return i
    for i, layer in enumerate(lrp_model.layers):
        if isinstance(layer, LRPLayer):
            return i
    raise RuntimeError("Kein Rückwärtspfad im LRP-Modell gefunden")


def _layer_rule_tag(layer) -> str:
    if isinstance(layer, StandardLRPLayer):
        parts = []
        if layer.epsilon is not None:
            parts.append(f"ε={layer.epsilon}")
        if layer.gamma is not None:
            parts.append(f"γ={layer.gamma}")
        if layer.alpha is not None:
            parts.append(f"α={layer.alpha},β={layer.beta}")
        if getattr(layer, "b", False):
            parts.append("b")
        if getattr(layer, "flat", False):
            parts.append("flat")
        return ", ".join(parts) if parts else "LRP-0"
    if isinstance(layer, PoolingLRPLayer):
        return str(getattr(layer, "strategy", "?"))
    return "—"


def _collect_layer_relevance(
    model,
    vol: np.ndarray,
    device: str,
    *,
    strategy: LRPStrategy,
) -> pd.DataFrame:
    """ΣR je Mask-/Standard-/Pooling-LRP-Schicht (eine Output-Schicht nach der anderen)."""
    x = np.expand_dims(vol, 0).astype(np.float32)
    with tf.device(device):
        lrp_model = LRP(
            model,
            layer=len(model.layers) - 1,
            idx=0,
            strategy=strategy,
        )
        start = _find_backward_start(lrp_model)
        layers = [
            lyr
            for lyr in lrp_model.layers[start:]
            if isinstance(lyr, (StandardLRPLayer, PoolingLRPLayer))
            or (isinstance(lyr, Lambda) and "output_mask" in lyr.name)
        ]
        rows = []
        for layer in layers:
            probe = KerasModel(lrp_model.input, layer.output)
            R = np.asarray(probe(x, training=False).numpy())
            sum_r = float(np.sum(R))
            rows.append(
                {
                    "name": layer.name,
                    "type": type(layer).__name__,
                    "rule": _layer_rule_tag(layer),
                    "sum_R": sum_r,
                    "abs_max": float(np.nanmax(np.abs(R))) if R.size else float("nan"),
                    "n_nan": int(np.isnan(R).sum()),
                    "n_inf": int(np.isinf(R).sum()),
                }
            )
            del probe, R
    df = pd.DataFrame(rows)
    if len(df):
        r0 = float(df["sum_R"].iloc[0])
        df["delta_sum_R"] = df["sum_R"].diff()
        df["ratio_to_start"] = df["sum_R"] / r0 if r0 else np.nan
    return df


def _frac_abs_r_in_mask(R: np.ndarray, mask: np.ndarray) -> float:
    abs_r = np.abs(R.astype(np.float64))
    total = float(abs_r.sum())
    if total <= 0:
        return float("nan")
    m = mask.astype(bool)
    if m.shape != R.shape:
        raise ValueError(f"Maske {m.shape} != R {R.shape}")
    return float(abs_r[m].sum() / total)


# ---------------------------------------------------------------------------
# N.1  y_pred vs. ΣR
# ---------------------------------------------------------------------------
print("=" * 72)
print("N.1  Vorhersage y_pred vs. Relevanzsumme ΣR (training=False)")
print("=" * 72)

rows_n1: list[dict[str, object]] = []
for key in DATASETS:
    model = models[key]
    load_vol = load_volume_fns[key]
    vol = load_vol(str(plot_paths[key]))
    for label, device in devices_to_run:
        y_pred = _predict_on_device(model, vol, device)
        sum_r = float(meta_by_device[label][key]["sum_R"])
        abs_max = float(meta_by_device[label][key]["abs_max"])
        ratio = sum_r / y_pred if y_pred != 0 else float("nan")
        rows_n1.append(
            {
                "dataset": key,
                "device": label,
                "y_pred": y_pred,
                "sum_R": sum_r,
                "sum_R/y_pred": ratio,
                "|R|_max": abs_max,
            }
        )
        print(
            f"[{label}/{key}] y_pred={y_pred:.4g}  ΣR={sum_r:.4g}  "
            f"ΣR/y_pred={ratio:.4g}"
        )

df_n1 = pd.DataFrame(rows_n1)
display(df_n1)

fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.2 * len(DATASETS), 4.0))
if len(DATASETS) == 1:
    axes = [axes]
for ax, key in zip(axes, DATASETS):
    sub = df_n1[df_n1["dataset"] == key]
    x_pos = np.arange(len(sub))
    ax.bar(x_pos - 0.15, sub["y_pred"], width=0.3, label="y_pred")
    ax.bar(x_pos + 0.15, sub["sum_R"], width=0.3, label="ΣR")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sub["device"].tolist())
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title(DATASETS[key]["label"], fontsize=10)
    ax.set_ylabel("Wert (symlog)")
    ax.legend(fontsize=8)
fig.suptitle(
    f"N.1  y_pred vs. ΣR  ·  Subject {subject_id_plot}\n"
    "Erwartung: GPU ΣR ≈ y_pred; CPU oft um Größenordnungen daneben",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)


# ---------------------------------------------------------------------------
# N.2  Layer-weise ΣR
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("N.2  Layer-weise Relevanzsumme ΣR (Mask + StandardLRP + Pooling)")
print("=" * 72)

layerwise: dict[str, dict[str, pd.DataFrame]] = {
    label: {} for label in _device_labels()
}
t0_lw = time.perf_counter()
for key in DATASETS:
    model = models[key]
    vol = load_volume_fns[key](str(plot_paths[key]))
    for label, device in devices_to_run:
        print(f"  layerwise [{label}/{key}] …")
        df_lw = _collect_layer_relevance(
            model, vol, device, strategy=LRP_STRATEGY
        )
        layerwise[label][key] = df_lw
        jump_idx = None
        if len(df_lw) > 1:
            dlt = df_lw["delta_sum_R"].iloc[1:].abs()
            if len(dlt) and dlt.notna().any():
                jump_idx = int(dlt.idxmax())
        jump = df_lw.loc[jump_idx] if jump_idx is not None else None
        print(
            f"    Start ΣR={df_lw['sum_R'].iloc[0]:.4g}  "
            f"Ende ΣR={df_lw['sum_R'].iloc[-1]:.4g}  "
            f"max|Δ| @ {jump['name'] if jump is not None else '—'} "
            f"({float(jump['delta_sum_R']) if jump is not None else float('nan'):.4g})"
        )
print(f"Dauer N.2: {time.perf_counter() - t0_lw:.1f} s")

n_dev = len(devices_to_run)
fig, axes = plt.subplots(
    n_dev, len(DATASETS), figsize=(4.5 * len(DATASETS), 3.8 * n_dev), squeeze=False
)
for row, (label, _) in enumerate(devices_to_run):
    for col, key in enumerate(DATASETS):
        ax = axes[row, col]
        df_lw = layerwise[label][key]
        ax.plot(df_lw["sum_R"].values, marker="o", ms=3)
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(f"{label} · {DATASETS[key]['label']}", fontsize=9)
        ax.set_xlabel("LRP-Schritt (Mask → Input)")
        ax.set_ylabel("ΣR")
        if len(df_lw):
            ax.axhline(df_lw["sum_R"].iloc[0], color="C1", ls="--", lw=0.8)
fig.suptitle(
    "N.2  ΣR entlang des LRP-Rückwegs\n"
    "Erwartung: GPU flach; CPU mit Sprung (oft früh bei Conv/Padding)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.90])
if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)

# Tabelle: größter |ΔΣR| je Device/Dataset
rows_jump = []
for label in _device_labels():
    for key in DATASETS:
        df_lw = layerwise[label][key]
        if len(df_lw) < 2:
            continue
        i = None
        if len(df_lw) > 1:
            dlt = df_lw["delta_sum_R"].iloc[1:].abs()
            if len(dlt) and dlt.notna().any():
                i = int(dlt.idxmax())
        if i is None:
            continue
        row = df_lw.loc[i]
        rows_jump.append(
            {
                "device": label,
                "dataset": key,
                "layer": row["name"],
                "rule": row["rule"],
                "delta_sum_R": float(row["delta_sum_R"]),
                "sum_R_after": float(row["sum_R"]),
            }
        )
display(pd.DataFrame(rows_jump))


# ---------------------------------------------------------------------------
# N.3  Anteil |R| in Zero-Padding / außerhalb Thalamus
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("N.3  Anteil |R| in Null-Voxel bzw. außerhalb rechter Thalamus-Maske")
print("=" * 72)

rows_n3: list[dict[str, object]] = []
for key in DATASETS:
    vol = plot_vols[key]
    zero_mask = np.abs(vol) <= ZERO_EPS
    if vol.ndim == 4:
        zero_mask = zero_mask[..., 0]
    thal = np.asarray(nib.load(str(plot_masks[key])).get_fdata()) > 0
    thal = thal.squeeze()
    for label in _device_labels():
        R = heatmaps_by_device[label][key]
        if R.ndim == 4:
            R = R[..., 0]
        frac_zero = _frac_abs_r_in_mask(R, zero_mask)
        frac_out_thal = _frac_abs_r_in_mask(R, ~thal)
        frac_thal = _frac_abs_r_in_mask(R, thal)
        rows_n3.append(
            {
                "dataset": key,
                "device": label,
                "pct_|R|_in_zeros": 100.0 * frac_zero,
                "pct_|R|_outside_thal": 100.0 * frac_out_thal,
                "pct_|R|_in_thal": 100.0 * frac_thal,
                "pct_voxels_zero": 100.0 * float(zero_mask.mean()),
            }
        )
        print(
            f"[{label}/{key}] %|R| in zeros={100 * frac_zero:.1f}%  "
            f"%|R| außerhalb Thal={100 * frac_out_thal:.1f}%  "
            f"%|R| in Thal={100 * frac_thal:.1f}%"
        )

df_n3 = pd.DataFrame(rows_n3)
display(df_n3)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
metrics = [
    ("pct_|R|_in_zeros", "%|R| in Null-Voxel"),
    ("pct_|R|_outside_thal", "%|R| außerhalb Thalamus"),
]
for ax, (col, title) in zip(axes, metrics):
    for label in _device_labels():
        sub = df_n3[df_n3["device"] == label]
        ax.plot(
            range(len(sub)),
            sub[col].values,
            marker="o",
            label=label,
        )
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASETS[k]["label"] for k in DATASETS], rotation=15, ha="right")
    ax.set_ylabel("%")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
fig.suptitle(
    f"N.3  Relevanz-Leckage  ·  Subject {subject_id_plot}\n"
    "Erwartung: CPU hoch in Zeros/Rahmen; GPU niedrig (bes. all-zero)",
    fontsize=11,
)
fig.tight_layout(rect=[0, 0, 1, 0.88])
if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)


# ---------------------------------------------------------------------------
# N.4  Größeres ε (Dense-Regel)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print(f"N.4  Stabilisierung: Dense-ε → {EPS_LARGE} (Basis-Strategie hatte ε=0.25)")
print("=" * 72)

LRP_STRATEGY_EPS_LARGE = LRPStrategy(
    layers=[
        {"flat": True},
        {"flat": True},
        {"alpha": 2, "beta": 1},
        {"alpha": 2, "beta": 1},
        {"alpha": 2, "beta": 1},
        {"alpha": 2, "beta": 1},
        {"epsilon": EPS_LARGE},
    ],
    pooling=[{"strategy": "flat"}] * N_POOLING_LAYERS,
)

rows_n4: list[dict[str, object]] = []
heatmaps_eps: dict[str, dict[str, np.ndarray]] = {
    label: {} for label in _device_labels()
}
t0_eps = time.perf_counter()
for key in DATASETS:
    model = models[key]
    vol = load_volume_fns[key](str(plot_paths[key]))
    zero_mask = np.abs(vol.squeeze()) <= ZERO_EPS
    for label, device in devices_to_run:
        x = np.expand_dims(vol, 0).astype(np.float32)
        with tf.device(device):
            lrp_e = LRP(
                model,
                layer=len(model.layers) - 1,
                idx=0,
                strategy=LRP_STRATEGY_EPS_LARGE,
            )
            R_e = np.asarray(
                lrp_e(x, training=False)[0].numpy(), dtype=np.float32
            ).squeeze()
        heatmaps_eps[label][key] = R_e
        base = heatmaps_by_device[label][key]
        rows_n4.append(
            {
                "dataset": key,
                "device": label,
                "sum_R_base": float(np.nansum(base)),
                "sum_R_eps_large": float(np.nansum(R_e)),
                "|R|_max_base": float(np.nanmax(np.abs(base))),
                "|R|_max_eps_large": float(np.nanmax(np.abs(R_e))),
                "pct_|R|_zeros_base": 100.0
                * _frac_abs_r_in_mask(base.squeeze(), zero_mask),
                "pct_|R|_zeros_eps_large": 100.0
                * _frac_abs_r_in_mask(R_e.squeeze(), zero_mask),
            }
        )
        print(
            f"[{label}/{key}] ΣR {float(np.nansum(base)):.4g} → "
            f"{float(np.nansum(R_e)):.4g}  |R|_max "
            f"{float(np.nanmax(np.abs(base))):.4g} → "
            f"{float(np.nanmax(np.abs(R_e))):.4g}"
        )
print(f"Dauer N.4: {time.perf_counter() - t0_eps:.1f} s")
df_n4 = pd.DataFrame(rows_n4)
display(df_n4)

# 2×3: CPU base vs CPU ε-large (Instabilität sichtbar?)
if "CPU" in heatmaps_eps:
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.0))
    fig.suptitle(
        f"N.4  CPU: Basis-ε vs. ε={EPS_LARGE}  ·  Subject {subject_id_plot}\n"
        "Wenn CPU mit großem ε der GPU ähnlicher wird → Nenner-Problem",
        fontsize=11,
    )
    for col, key in enumerate(DATASETS):
        for row, (tag, R) in enumerate(
            [
                ("CPU Basis", heatmaps_by_device["CPU"][key]),
                (f"CPU ε={EPS_LARGE}", heatmaps_eps["CPU"][key]),
            ]
        ):
            ax = axes[row, col]
            vmax = float(np.nanpercentile(np.abs(R), 99.5)) or 1.0
            ax.imshow(
                sagittal_slc(R.squeeze(), cx),
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_title(
                f"{tag} · {DATASETS[key]['label']}\n"
                f"ΣR={float(np.nansum(R)):.3g}  |R|_max={float(np.nanmax(np.abs(R))):.3g}",
                fontsize=9,
            )
            ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    if SHOW_PLOTS_INLINE:
        display(fig)
    plt.close(fig)

print(
    "Hinweis N.4: αβ-Conv nutzt festes 1e-9 in explainability/layers/conv.py — "
    "größeres Dense-ε allein heilt Padding-Lecks oft nur teilweise."
)


# ---------------------------------------------------------------------------
# N.5  Nur Forward model(x)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("N.5  Bitgenau Forward: model(x) CPU vs. GPU (ohne LRP)")
print("=" * 72)

rows_n5: list[dict[str, object]] = []
if has_gpu:
    for key in DATASETS:
        model = models[key]
        vol = load_volume_fns[key](str(plot_paths[key]))
        x = np.expand_dims(vol, 0).astype(np.float32)
        with tf.device("/CPU:0"):
            y_cpu = model(x, training=False).numpy()
        with tf.device("/GPU:0"):
            y_gpu = model(x, training=False).numpy()
        diff = np.abs(y_cpu.astype(np.float64) - y_gpu.astype(np.float64))
        rows_n5.append(
            {
                "dataset": key,
                "y_cpu": float(np.squeeze(y_cpu)),
                "y_gpu": float(np.squeeze(y_gpu)),
                "max|Δ|": float(diff.max()),
                "rel|Δ|": float(diff.max() / max(abs(float(np.squeeze(y_gpu))), 1e-12)),
            }
        )
        print(
            f"[{key}] y_cpu={float(np.squeeze(y_cpu)):.8g}  "
            f"y_gpu={float(np.squeeze(y_gpu)):.8g}  "
            f"max|Δ|={float(diff.max()):.4g}"
        )
    display(pd.DataFrame(rows_n5))
    print(
        "Erwartung: max|Δ| winzig → Forward OK, Diskrepanz sitzt im LRP-Backward."
    )
else:
    print("N.5 übersprungen (keine GPU).")


# ---------------------------------------------------------------------------
# N.6  Zwischen-Aktivierungen (erste/letzte Conv3D)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("N.6  Conv3D-Zwischenaktivierungen CPU vs. GPU")
print("=" * 72)

rows_n6: list[dict[str, object]] = []
if has_gpu:
    for key in DATASETS:
        model = models[key]
        vol = load_volume_fns[key](str(plot_paths[key]))
        x = np.expand_dims(vol, 0).astype(np.float32)
        convs = [lyr for lyr in model.layers if type(lyr).__name__ == "Conv3D"]
        if not convs:
            print(f"[{key}] keine Conv3D — übersprungen")
            continue
        targets = [convs[0], convs[-1]]
        probe = KerasModel(model.input, [t.output for t in targets])
        with tf.device("/CPU:0"):
            outs_cpu = probe(x, training=False)
        with tf.device("/GPU:0"):
            outs_gpu = probe(x, training=False)
        if not isinstance(outs_cpu, (list, tuple)):
            outs_cpu, outs_gpu = [outs_cpu], [outs_gpu]
        for lyr, a_cpu, a_gpu in zip(targets, outs_cpu, outs_gpu):
            ac = a_cpu.numpy().astype(np.float64)
            ag = a_gpu.numpy().astype(np.float64)
            d = np.abs(ac - ag)
            rows_n6.append(
                {
                    "dataset": key,
                    "layer": lyr.name,
                    "shape": tuple(int(s) for s in ac.shape),
                    "max|a|": float(np.max(np.abs(ag))),
                    "max|Δ|": float(d.max()),
                    "RMSE": float(np.sqrt(np.mean(d**2))),
                    "max|Δ|/max|a|": float(d.max() / max(np.max(np.abs(ag)), 1e-12)),
                }
            )
            print(
                f"[{key}/{lyr.name}] max|Δ|={float(d.max()):.4g}  "
                f"rel={float(d.max() / max(np.max(np.abs(ag)), 1e-12)):.3g}"
            )
        del probe
    display(pd.DataFrame(rows_n6))
    print(
        "Erwartung: relative Δ ≪ 1, aber nicht null — LRP (R/z) kann sie verstärken."
    )
else:
    print("N.6 übersprungen (keine GPU).")


# ---------------------------------------------------------------------------
# N.7  Praxis-Empfehlung
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("N.7  Praxis-Empfehlung (Zusammenfassung)")
print("=" * 72)

# Heuristik aus N.1 / N.3
summary_lines = [
    "Für Paper/Plots: LRP auf GPU mit training=False rechnen.",
    "CPU-Heatmaps bei Rahmen-Artefakten und ΣR ≫ y_pred nicht interpretieren.",
]
if has_gpu and len(df_n1):
    cpu_ratios = df_n1[df_n1["device"] == "CPU"]["sum_R/y_pred"].to_numpy(dtype=float)
    gpu_ratios = df_n1[df_n1["device"] == "GPU"]["sum_R/y_pred"].to_numpy(dtype=float)
    if np.nanmedian(np.abs(cpu_ratios)) > 10 * max(np.nanmedian(np.abs(gpu_ratios)), 1e-6):
        summary_lines.append(
            "Befund N.1: CPU-Relevanzerhaltung klar gebrochen → Device-Numerik bestätigt."
        )
if len(df_n3):
    cpu_z = df_n3[df_n3["device"] == "CPU"]["pct_|R|_in_zeros"].median()
    if "GPU" in df_n3["device"].values:
        gpu_z = df_n3[df_n3["device"] == "GPU"]["pct_|R|_in_zeros"].median()
        summary_lines.append(
            f"Befund N.3: Median %|R| in Zeros  CPU={cpu_z:.1f}%  GPU={gpu_z:.1f}%."
        )
if has_gpu and rows_n5:
    max_fwd = max(r["max|Δ|"] for r in rows_n5)
    summary_lines.append(
        f"Befund N.5: max Forward-|Δ|={max_fwd:.4g} "
        + (
            "→ Forward ok, Fokus auf LRP-Backward."
            if max_fwd < 1e-3
            else "→ auch Forward weicht ab, Conv/cuDNN prüfen."
        )
    )
summary_lines.append(
    "Nächste Code-Fixes (optional): |z|<τ abschneiden in StandardLRPLayer; "
    "αβ-Eps in conv.py erhöhen; LRP-Tests CPU+GPU."
)
for line in summary_lines:
    print(f"  • {line}")

print(
    "\nAbschnitt N fertig. Objekte: df_n1 … df_n4, layerwise, heatmaps_eps, "
    "rows_n5, rows_n6."
)
