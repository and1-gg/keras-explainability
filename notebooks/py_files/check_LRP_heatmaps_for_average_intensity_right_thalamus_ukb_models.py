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
# # LRP-Heatmaps: `average-intensity_right-whole-thalamus` (UKB)
#
# Vergleich von drei auf UKB trainierten SFCN-Modellen für die Label-Variable
# `average-intensity_right-whole-thalamus` (mittlere Intensität des rechten Thalamus):
#
# 1. **unveränderte** MRI-Scans (`cropped.nii.gz`)
# 2. **gejitterte** Scans (rechter Thalamus erhalten, Rest permutiert)
# 3. **all-zero** außer rechtem Thalamus
#
# Ablauf: Vorhersagen für `N_SUBJ_PRED` Holdout-Subjects → Scatter mit MAE und
# Pearson-r → Tabelle wahr vs. prädiziert → für Subject `IDX_PRED` sagittale
# Inputs (`x=70`) mit rechter Thalamus-Maske → drei unnormierte LRP-Heatmaps,
# die bei jedem Lauf neu berechnet werden.
#

# %% [markdown]
# ## A. Imports
#

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


# %% [markdown]
# ## B. Konfiguration
#
# - `N_SUBJ_PRED`: Anzahl Holdout-Subjects, für die jedes der drei Modelle
#   vorhersagt. Die Subjects sind die ersten Zeilen der Holdout-TSV von
#   Dataset **(1)**; dieselben IDs werden in den TSVs von **(2)** und **(3)**
#   nachgeschlagen.
# - `IDX_PRED`: Index des Subjects für die 2×3-Figur. Muss
#   `0 <= IDX_PRED < N_SUBJ_PRED` erfüllen, sonst Fehler.
#

# %%
N_SUBJ_PRED = 3
IDX_PRED = 1

PRED_BATCH_SIZE = 8
SAGITTAL_X = 70
SHOW_PLOTS_INLINE = True
COLOR_RIGHT = (0.15, 0.65, 0.25, 0.50)  # grün, alpha=0.5
PRED_VAR = "average-intensity_right-whole-thalamus"

if not (0 <= int(IDX_PRED) < int(N_SUBJ_PRED)):
    raise ValueError(
        f"IDX_PRED={IDX_PRED} muss kleiner als N_SUBJ_PRED={N_SUBJ_PRED} sein "
        f"(gültig: 0 .. {N_SUBJ_PRED - 1})."
    )

# (1) unveränderte MRI (dataset ukb, original scans)
RUN_DIR_NORMAL = Path(
    "/mnt/ceph2/dl_project/data/nn-trainings/mri/"
    "average-intensity_right-whole-thalamus/"
    "training_run_09h17m23s_23sep2026"
).resolve()
PREDICT_TSV_NORMAL = Path(
    "/mnt/users/andreasre/git-repos/pyment-and1/training_runs/input_files/mri/"
    "right_whole_thalamus/average_intensity/original_scans/predict.tsv"
).resolve()
MASK_TMPL_NORMAL = (
    "/mnt/ceph2/dl_project/data/mri-scans/not_altered/ukb/recon/"
    "{sid}/mri/aseg_mni152_right_thalamus_cropped.nii.gz"
)

# (2) gejitterte MRI (rechter Thalamus erhalten)
RUN_DIR_JITTER = Path(
    "/mnt/ceph2/dl_project/data/nn-trainings/mri/"
    "average-intensity_right-whole-thalamus/"
    "training_run_17h45m44s_23sep2026"
).resolve()
PREDICT_TSV_JITTER = Path(
    "/mnt/users/andreasre/git-repos/pyment-and1/training_runs/input_files/mri/"
    "right_whole_thalamus/average_intensity/jittered_data/predict.tsv"
).resolve()
MASK_TMPL_JITTER = (
    "/mnt/ceph2/dl_project/data/mri-scans/jittered_data/ukb/recon/"
    "{sid}/mri/aseg_mni152_right_thalamus_cropped.nii.gz"
)

# (3) all-zero außer rechtem Thalamus
RUN_DIR_ALL_ZERO = Path(
    "/mnt/ceph2/dl_project/data/nn-trainings/mri/"
    "average-intensity_right-whole-thalamus/"
    "training_run_22h16m52s_23sep2026"
).resolve()
PREDICT_TSV_ALL_ZERO = Path(
    "/mnt/users/andreasre/git-repos/pyment-and1/training_runs/input_files/mri/"
    "right_whole_thalamus/average_intensity/all_zero_except_right_thalamus/predict.tsv"
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
        # `_cropped` liegt für viele Subjects nicht unter not_altered;
        # die Datei ohne Suffix hat dieselbe Shape wie das Input-Volume.
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
print(f"PRED_VAR    = {PRED_VAR}")
for key, meta in DATASETS.items():
    print(f"[{key}] run={meta['run_dir'].name}")
    print(f"        tsv={meta['predict_tsv']}")


# %% [markdown]
# ## C. Repo-Pfade (`keras-explainability` + `pyment-and1`)
#
# Zur Laufzeit wird Code aus beiden Repos gebraucht. `sys.path` wird um die
# keras-explainability-Root (Paket `explainability`) und um
# `pyment-and1/src` (Paket `pybrainmetrics`) ergänzt.
#

# %%
PYMENT_AND1_ROOT = Path("/mnt/users/andreasre/git-repos/pyment-and1").resolve()
KERAS_XAI_ROOT_DEFAULT = Path("/mnt/users/andreasre/git-repos/keras-explainability").resolve()


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
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(KERAS_XAI_ROOT_DEFAULT)
    root = _first_existing_dir(candidates)
    if root is not None and (root / "explainability").is_dir():
        return root
    raise FileNotFoundError(
        "keras-explainability-Root nicht gefunden. "
        "Notebook aus dem Repo starten oder KERAS_XAI_ROOT setzen."
    )


def find_pybrainmetrics_src() -> Path:
    env = os.environ.get("PYBRAINMETRICS_SRC")
    candidates: list[Path] = [PYMENT_AND1_ROOT / "src"]
    if env:
        candidates.insert(0, Path(env).expanduser())
    candidates.extend(
        [
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

print("pyment-and1:  ", PYMENT_AND1_ROOT)
print("keras-xai:     ", keras_xai_root)
print("pybrainmetrics:", pybm_src)
print("sys.path[0:2]: ", sys.path[:2])
print("TensorFlow:    ", tf.__version__)


# %% [markdown]
# ## D. Labels laden und Subject-Pool festlegen
#
# Jede Variante hat eine eigene Holdout-`predict.tsv` (Dateipfad, Subject-ID,
# `average-intensity_right-whole-thalamus`). Die drei Dateien enthalten dieselben
# Subject-IDs in unterschiedlicher Reihenfolge. Für den Vergleich werden die
# ersten `N_SUBJ_PRED` IDs aus der TSV der unveränderten Scans genommen und in
# den anderen TSVs nachgeschlagen.
#

# %%
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
    df = pd.read_csv(tsv, sep="\t")
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
# Pro Datenvariante wird das zugehörige `model.keras` geladen. Der NIfTI-Loader
# kommt aus `pybrainmetrics` und folgt `config.yaml` (`nifti-native`).
# LRP-Composite für SFCN: zwei `flat`-Schichten, vier αβ-Schichten (α=2, β=1),
# ε am Dense-Ausgang. Gespeicherte Heatmap-NIfTIs werden nicht gelesen.
#
# Zusätzlich liest jede Variante `config_training.yaml`. Ist
# `training.normalisation.use: true` und `z_score` in `types`, werden μ und σ
# aus `label_normalisation.yaml` für die spätere Rücktransformation geladen.
#

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
    ],
)

"""
LRP_STRATEGY = LRPStrategy(
    layers=[
        {"flat": True},
        {"flat": True},
        {"alpha": 1, "beta": 0},
        {"alpha": 1, "beta": 0},
        {"alpha": 1, "beta": 0},
        {"alpha": 1, "beta": 0},
        {"epsilon": 0.25},
    ],
)
"""

def zscore_inverse_params(run_dir: Path) -> dict[str, float | bool]:
    """Read config_training.yaml. Inverse z-score only if normalisation.use is true."""
    train_cfg = OmegaConf.load(run_dir / "config_training.yaml")
    norm_cfg = getattr(getattr(train_cfg, "training", None), "normalisation", None)
    use_flag = bool(norm_cfg is not None and getattr(norm_cfg, "use", False))
    types = list(getattr(norm_cfg, "types", []) or []) if norm_cfg is not None else []
    z_on = use_flag and ("z_score" in types)
    if not z_on:
        return {"use": False, "mean": 0.0, "std": 1.0}
    params_path = run_dir / "label_normalisation.yaml"
    if not params_path.is_file():
        raise FileNotFoundError(
            f"normalisation.use=true (z_score), aber {params_path} fehlt."
        )
    params = OmegaConf.load(params_path)
    if str(getattr(params, "type", "")) != "z_score":
        raise ValueError(f"Unerwarteter Normalisierungstyp in {params_path}: {params}")
    std = float(params.std)
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError(f"Ungültige z-score-Std in {params_path}: {std}")
    return {"use": True, "mean": float(params.mean), "std": std}


def inverse_zscore_values(values, params: dict[str, float | bool]):
    """Map network outputs from z-space back to the original label scale."""
    arr = np.asarray(values, dtype=np.float64)
    if not params["use"]:
        return arr
    return arr * float(params["std"]) + float(params["mean"])


models: dict[str, object] = {}
load_volume_fns: dict[str, object] = {}
zscore_params: dict[str, dict[str, float | bool]] = {}
for key, meta in DATASETS.items():
    model, cfg, delta = load_keras_model(meta["run_dir"])
    models[key] = model
    load_volume_fns[key] = make_load_volume(cfg)
    zscore_params[key] = zscore_inverse_params(meta["run_dir"])
    zp = zscore_params[key]
    print(
        f"[{key}] geladen  Δw={delta:.3g}  loader={cfg.data.loader}  "
        f"label={cfg.data.prediction_variable}"
    )
    if zp["use"]:
        print(
            f"        z-score aktiv (config_training.yaml)  "
            f"μ={zp['mean']:.6g}  σ={zp['std']:.6g}"
        )
    else:
        print("        keine Label-z-score-Normalisierung")


# %% [markdown]
# ## F. Vorhersagen für `N_SUBJ_PRED` Subjects
#
# Jedes Modell sagt für dieselben Holdout-Subjects die mittlere Intensität des
# rechten Thalamus voraus. Der Dateipfad und der wahre Wert kommen aus der
# jeweiligen `predict.tsv` (Originalskala).
#
# Die Modellausgabe liegt im z-Score-Raum, wenn in `config_training.yaml`
# `normalisation.use: true` steht. Dann wird jede Vorhersage mit
# `y = z · σ + μ` auf die Intensitätsskala zurückgerechnet. Dieselben Werte
# gehen in Scatter, Tabelle und die 2×3-Figur.
#

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
        y_pred_net = float(np.squeeze(model.predict(np.expand_dims(vol, 0), verbose=0)))
        y_pred = float(np.squeeze(inverse_zscore_values(y_pred_net, zscore_params[dataset_key])))
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


preds_by_dataset: dict[str, pd.DataFrame] = {
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
    print(f"[{key}] n={len(df)}  MAE={mae:.4f}  r={r_val:.3f}")

metrics_df = pd.DataFrame(metrics_rows)


# %% [markdown]
# ## G. Scatter-Plot (3 Panels)
#
# Ein Panel je Datenvariante. Jeder Punkt ist ein Holdout-Subject
# (`N_SUBJ_PRED`). Im Titel stehen MAE und Pearson-Korrelation.
# Subject `IDX_PRED` ist rot umrandet. Die Diagonale ist die ideale Vorhersage.
#

# %%
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
fig.suptitle(
    f"Wahr vs. prädiziert  ·  {PRED_VAR}  ·  n={N_SUBJ_PRED}",
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
    ax.set_xlabel("wahre Intensität")
    ax.set_ylabel("prädizierte Intensität")
    ax.set_title(f"{DATASETS[key]['label']}\nMAE={mae:.4f}  r={r_val:.3f}")
    ax.set_aspect("equal", adjustable="box")
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
# ## H. Tabelle: wahre vs. prädizierte Intensität
#
# Für alle drei Datensätze und die `N_SUBJ_PRED` Subjects: wahrer Wert aus der
# Holdout-TSV und Vorhersage des jeweiligen Modells
# (`average-intensity_right-whole-thalamus`).
#

# %%
table_parts = []
for key, df in preds_by_dataset.items():
    part = df[["subject_id", "y_true", "y_pred"]].copy()
    part = part.rename(
        columns={
            "y_true": f"wahr_{key}",
            "y_pred": f"praed_{key}",
        }
    )
    table_parts.append(part.set_index("subject_id"))

table_df = pd.concat(table_parts, axis=1).reset_index()
print(f"Wahr vs. prädiziert ({PRED_VAR}):")
display(table_df.round(4))
display(metrics_df.round(4))


# %% [markdown]
# ## I. Input-Bilder und Thalamus-Maske für `IDX_PRED`
#
# Sagittaler Schnitt `x=70` des Input-Volumes aus der jeweiligen `predict.tsv`.
# Die rechte Thalamus-Maske wird grün mit `alpha=0.5` darübergelegt.
#
# Masken:
#
# 1. `…/not_altered/ukb/recon/<sid>/mri/aseg_mni152_right_thalamus_cropped.nii.gz`
#    (Fallback ohne `_cropped`, gleiche Shape)
# 2. `…/jittered_data/ukb/recon/<sid>/mri/aseg_mni152_right_thalamus_cropped.nii.gz`
# 3. `…/only_brain_regions/ukb/recon/<sid>/mri/aseg_mni152_right_thalamus_cropped.nii.gz`
#    (Fallback ohne `_cropped`, bzw. cropped unter `right-thalamus/`)
#

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
        f"        true={plot_true[key]:.4f}  pred={plot_pred[key]:.4f}  "
        f"shape={vol.shape}"
    )

cx = int(np.clip(SAGITTAL_X, 0, next(iter(plot_vols.values())).shape[0] - 1))
print(f"sagittal x = {cx}")


# %% [markdown]
# ## J. LRP-Heatmaps neu berechnen (unnormiert)
#
# Für Subject `IDX_PRED` werden die drei Heatmaps bei jedem Lauf neu berechnet.
# Gespeicherte Dateien (`lrp_heatmap_raw.nii.gz` o. Ä.) werden nicht geladen.
#
# War `normalisation.use: true`, wird jede Heatmap mit dem Trainings-σ
# multipliziert (`R · σ`). Das ist der lineare Teil der z-Score-Rücktransformation.
# μ wird nicht auf einzelne Voxel addiert: der Mittelwert ist ein globaler Offset
# der Vorhersage, keine räumliche Relevanz. Danach gilt
# `ΣR · σ + μ ≈` zurückgerechnete Vorhersage. Der Thalamus-Anteil
# `ΣR_Thalamus / ΣR` ändert sich durch die Multiplikation mit σ nicht.
# Anschließend wird die Dauer aller drei Berechnungen ausgegeben.
#

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
    R = np.asarray(lrp(np.expand_dims(vol, 0))[0].numpy(), dtype=np.float64).squeeze()
    zp = zscore_params[key]
    if zp["use"]:
        R = R * float(zp["std"])
    heatmaps[key] = np.asarray(R, dtype=np.float32)
    scale_note = (
        f"  · R·σ (σ={float(zp['std']):.6g})"
        if zp["use"]
        else "  · keine z-score-Skalierung"
    )
    print(
        f"[{key}] LRP fertig  shape={heatmaps[key].shape}  "
        f"|R|_max={float(np.nanmax(np.abs(heatmaps[key]))):.4g}  "
        f"ΣR={float(np.sum(heatmaps[key])):.4g}{scale_note}"
    )
elapsed_s = time.perf_counter() - t0
print(
    f"\nDauer Berechnung aller 3 Heatmaps: {elapsed_s:.2f} s "
    f"({elapsed_s / 60.0:.2f} min)"
)


# %% [markdown]
# ## K. Figur 2×3: Inputs (oben) + LRP-Heatmaps (unten)
#
# Drei Spalten, eine je Datensatz **(1)**–**(3)**.
#
# - Obere Reihe: Input-Bild, sagittal `x=70`, eigene Colorbar. Rechte
#   Thalamus-Maske in Grün (`alpha=0.5`).
# - Untere Reihe: LRP-Relevanzen auf der Intensitätsskala (bei aktivem z-Score
#   `R · σ`), eigene Colorbar je Panel (`vmin/vmax = ±P99.5(|R|)` nur für die
#   Farbskala). `|R|_max` im Titel ist der Peak nach der Rücktransformation.
#   Zusätzlich der Anteil der Gesamtrelevanz im rechten Thalamus:
#   `100 · (Σ R im rechten Thalamus) / (Σ R über alle Voxel)`.
# - In jedem Panel: Subject-ID, wahre Intensität und zurückgerechnete Vorhersage
#   (`z · σ + μ`, falls `normalisation.use: true`).
#

# %%
fig, axes = plt.subplots(2, 3, figsize=(14.5, 10.4))
fig.suptitle(
    (
        f"Subject {subject_id_plot}  ·  IDX_PRED={IDX_PRED}/{N_SUBJ_PRED}  ·  "
        f"sagittal x={cx}  ·  {PRED_VAR}"
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
        f"wahr={plot_true[key]:.4f}  präd={plot_pred[key]:.4f}",
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
    pct_thal = 100.0 * sum_R_thal / sum_R if abs(sum_R) > 0 else float("nan")
    abs_max = float(np.nanmax(np.abs(heat))) or 1.0
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
        f"LRP · {DATASETS[key]['label']}\n"
        f"{subject_id_plot}\n"
        f"wahr={plot_true[key]:.4f}  präd={plot_pred[key]:.4f}  "
        f"|R|_max={abs_max:.3g}\n"
        f"Thalamus = {pct_thal:.1f}% der Gesamtrelevanz",
        fontsize=9,
    )
    print(
        f"[{key}] Thalamus-Anteil = {pct_thal:.2f}%  "
        f"(ΣR_Thalamus={sum_R_thal:.4g} / ΣR={sum_R:.4g})"
    )
    ax.axis("off")
    im_lrps.append(im)

for col, im in enumerate(im_vols):
    cbar = fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
    cbar.set_label("Intensität")
for col, im in enumerate(im_lrps):
    cbar = fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
    cbar.set_label("LRP-Relevanz")

fig.legend(
    handles=[Patch(facecolor=COLOR_RIGHT, edgecolor="none", label="rechter Thalamus")],
    loc="lower center",
    ncol=1,
    frameon=False,
    fontsize=9,
)
fig.tight_layout(rect=[0, 0.05, 1, 0.92])

if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)

print(
    f"Heatmap-Berechnung (Abschnitt J): {elapsed_s:.2f} s für alle 3 Modelle "
    f"(Subject {subject_id_plot})."
)

