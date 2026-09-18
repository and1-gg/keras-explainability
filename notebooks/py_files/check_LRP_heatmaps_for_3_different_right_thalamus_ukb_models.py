# ---
# jupyter:
#   jupytext:
#     formats: notebooks/ipynb_files//ipynb,notebooks/py_files//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
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
# Gruppen-LRP über die ersten `N_GROUP` Subjects je Holdout-TSV (Sum-Norm).

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


LRP_STRATEGY = LRPStrategy(
    layers=[
        {"flat": True},
        {"flat": True},
        {"alpha": 2, "beta": 1},
        {"alpha": 2, "beta": 1},
        {"alpha": 2, "beta": 1},
        {"alpha": 2, "beta": 1},
        {"epsilon": 0.25},
    ]
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
# - Untere Reihe: unnormierte LRP-Relevanzen, eigene Colorbar je Panel.
# - In jedem Panel: Subject-ID, wahres und prädiziertes Volumen.

# %%
fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.8))
fig.suptitle(
    f"Subject {subject_id_plot}  ·  IDX_PRED={IDX_PRED}/{N_SUBJ_PRED}  ·  "
    f"sagittal x={cx}",
    fontsize=12,
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
    vmax_r = float(np.nanmax(np.abs(heat))) or 1.0
    r_slc = sagittal_slc(right.astype(np.float32), cx) > 0
    im = ax.imshow(
        sagittal_slc(heat, cx),
        cmap="seismic",
        vmin=-vmax_r,
        vmax=vmax_r,
    )
    ax.imshow(rgba_mask(r_slc, COLOR_RIGHT), interpolation="nearest")
    ax.set_title(
        f"LRP unnormiert · {DATASETS[key]['label']}\n"
        f"{subject_id_plot}\n"
        f"true={plot_true[key]:.0f}  pred={plot_pred[key]:.0f}  "
        f"|R|_max={vmax_r:.3g}",
        fontsize=9,
    )
    ax.axis("off")
    im_lrps.append(im)

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
fig.tight_layout(rect=[0, 0.05, 1, 0.95])

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

    group_heatmaps[key] = heat_acc.astype(np.float32)
    group_masks[key] = group_mask
    group_meta[key] = {
        "n": n_ok,
        "sum_H": float(np.sum(heat_acc)),
        "max_H": float(np.max(heat_acc)),
        "mask_voxels": int(np.sum(group_mask > 0)),
        "first_sids": first_sids,
    }
    print(
        f"[{key}] n={n_ok}  ΣH={group_meta[key]['sum_H']:.4g} "
        f"(≈ N_GROUP={N_GROUP})  max(H)={group_meta[key]['max_H']:.4g}  "
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
        f"n={meta['n']}  ΣH={meta['sum_H']:.2f}  max={meta['max_H']:.3g}",
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
fig.tight_layout(rect=[0, 0.08, 1, 0.86])

if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)

print(
    f"Gruppenfigur fertig (N_GROUP={N_GROUP}, Berechnung {elapsed_group_s:.1f} s)."
)
