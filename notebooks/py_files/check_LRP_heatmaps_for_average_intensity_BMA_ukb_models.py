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
# # LRP-Heatmaps: Bone-Marrow-Adiposity (BMA) avg-intensity (UKB)
#
# Vergleich von zwei auf UKB trainierten SFCN-Modellen für die Label-Variablen
#
# 1. `average-intensity_bone-marrow-adiposity`
# 2. `average-intensity_bone-marrow-adiposity_exclMDoutliers`
#
# Beide Modelle nutzen dieselben großen Original-MRI-Scans
# (`nu_mni152_flirt.nii.gz`, kein `cropped.nii.gz`). Ablauf: Vorhersagen für `N_SUBJ_PRED`
# Holdout-Subjects → Scatter mit MAE und Pearson-r → Tabelle wahr vs.
# prädiziert → für Subject `IDX_PRED` sagittale Inputs (`x=70`) und zwei
# unnormierte LRP-Heatmaps, die bei jedem Lauf neu berechnet werden →
# Gruppen-LRP über `N_group` zufällig gewählte Holdout-Subjects (Sum-Norm).
# Vorhersagen und Heatmaps werden bei aktivem z-Score aus
# `config_training.yaml` zurücktransformiert.
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
from omegaconf import OmegaConf, open_dict
from scipy.stats import pearsonr
from tqdm import tqdm



# %% [markdown]
# ## B. Konfiguration
#
# - `N_SUBJ_PRED`: Anzahl Holdout-Subjects, für die jedes der beiden Modelle
#   vorhersagt. Die Subjects sind die ersten Zeilen der gemeinsamen
#   Holdout-`predict.tsv`.
# - `IDX_PRED`: Index des Subjects für die 2×2-Figur. Muss
#   `0 <= IDX_PRED < N_SUBJ_PRED` erfüllen, sonst Fehler.
# - `N_group`: Anzahl zufällig gewählter Holdout-Subjects für die Gruppen-LRP
#   (Sum-Norm, Abschnitte L–M); Ziehung aus der gemeinsamen `predict.tsv`
#   mit Seed `GROUP_SEED`.
#

# %%
N_SUBJ_PRED = 3
IDX_PRED = 1
N_group = 100
GROUP_SEED = 42

PRED_BATCH_SIZE = 8
SAGITTAL_X = 70
SHOW_PLOTS_INLINE = True

TRAIN_TSV = Path(
    "/mnt/users/andreasre/git-repos/pyment-and1/training_runs/input_files/mri/"
    "bone_marrow/average_intensity/original_scans/train.tsv"
).resolve()
PREDICT_TSV = Path(
    "/mnt/users/andreasre/git-repos/pyment-and1/training_runs/input_files/mri/"
    "bone_marrow/average_intensity/original_scans/predict.tsv"
).resolve()
INPUT_TMPL = "/mnt/ceph/data/ukb/recon/{sid}/mri/nu_mni152_flirt.nii.gz"

if not (0 <= int(IDX_PRED) < int(N_SUBJ_PRED)):
    raise ValueError(
        f"IDX_PRED={IDX_PRED} muss kleiner als N_SUBJ_PRED={N_SUBJ_PRED} sein "
        f"(gültig: 0 .. {N_SUBJ_PRED - 1})."
    )
if int(N_group) < 1:
    raise ValueError(f"N_group muss >= 1 sein, got {N_group}.")

# (1) BMA avg-intensity (inkl. MD-Outlier)
RUN_DIR_BMA = Path(
    "/mnt/ceph2/dl_project/data/nn-trainings/mri/"
    "average-intensity_bone-marrow-adiposity/"
    "training_run_17h58m06s_07oct2026"
).resolve()

# (2) BMA avg-intensity ohne MD-Outlier
RUN_DIR_BMA_EXCL = Path(
    "/mnt/ceph2/dl_project/data/nn-trainings/mri/"
    "average-intensity_bone-marrow-adiposity_exclMDoutliers/"
    "training_run_18h13m39s_07oct2026"
).resolve()

DATASETS = {
    "bma": {
        "label": "(1) BMA inkl. Outlier",
        "run_dir": RUN_DIR_BMA,
        "predict_tsv": PREDICT_TSV,
        "pred_var": "average-intensity_bone-marrow-adiposity",
    },
    "bma_excl": {
        "label": "(2) BMA excl. MD-Outlier",
        "run_dir": RUN_DIR_BMA_EXCL,
        "predict_tsv": PREDICT_TSV,
        "pred_var": "average-intensity_bone-marrow-adiposity_exclMDoutliers",
    },
}

for key, meta in DATASETS.items():
    model_path = meta["run_dir"] / "model.keras"
    cfg_path = meta["run_dir"] / "config.yaml"
    train_cfg_path = meta["run_dir"] / "config_training.yaml"
    for p, name in (
        (model_path, f"{key} model.keras"),
        (cfg_path, f"{key} config.yaml"),
        (train_cfg_path, f"{key} config_training.yaml"),
        (meta["predict_tsv"], f"{key} predict.tsv"),
    ):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{name} fehlt: {p}")

if not TRAIN_TSV.is_file():
    raise FileNotFoundError(f"train.tsv fehlt: {TRAIN_TSV}")

print(f"N_SUBJ_PRED = {N_SUBJ_PRED}")
print(f"IDX_PRED    = {IDX_PRED}")
print(f"N_group     = {N_group}")
print(f"GROUP_SEED  = {GROUP_SEED}")
print(f"TRAIN_TSV   = {TRAIN_TSV}")
print(f"PREDICT_TSV = {PREDICT_TSV}")
for key, meta in DATASETS.items():
    print(f"[{key}] run={meta['run_dir'].name}")
    print(f"        pred_var={meta['pred_var']}")



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
from pybrainmetrics.data.spatial_crop import (  # noqa: E402
    assert_volume_shape,
    parse_crop_cfg,
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
# Holdout-Labels kommen aus der gemeinsamen `predict.tsv` (Originalskala).
# Trainingslabels liegen in `train.tsv` (nur zur Dokumentation / Pfadprüfung).
# Die ersten `N_SUBJ_PRED` Subject-IDs bilden den Pool; `IDX_PRED` wählt das
# Plot-Subject.
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


def load_predict_tsv(tsv: Path, pred_var: str) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t")
    return _normalize_label_columns(df, pred_var)


def tsv_lookup_by_subject(df: pd.DataFrame) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        out[str(row["participant_id"])] = row
    return out


# dieselbe TSV, aber je Modell eigene Label-Spalte
labels_full = {
    key: load_predict_tsv(meta["predict_tsv"], meta["pred_var"])
    for key, meta in DATASETS.items()
}
lookup = {key: tsv_lookup_by_subject(df) for key, df in labels_full.items()}

# Subject-Pool aus der ersten Datensatz-Variante (gleiche IDs in beiden)
_first_key = next(iter(DATASETS))
subject_ids = (
    labels_full[_first_key]["participant_id"].astype(str).head(int(N_SUBJ_PRED)).tolist()
)
missing_ids = [
    sid for sid in subject_ids if not all(sid in lookup[k] for k in DATASETS)
]
if missing_ids:
    raise RuntimeError(f"Subject-IDs fehlen in mind. einer TSV: {missing_ids}")

subject_id_plot = subject_ids[int(IDX_PRED)]
print(f"Subject-Pool (n={len(subject_ids)}): {subject_ids}")
print(f"Plot-Subject IDX_PRED={IDX_PRED}: {subject_id_plot}")


# %% [markdown]
# ## E. Modelle, Volume-Loader und LRP-Strategie
#
# Pro Label-Variable wird das zugehörige `model.keras` geladen.
#
# Wichtig: Es gibt **keine** `cropped.nii.gz`-Dateien. Eingabe ist immer
# `nu_mni152_flirt.nii.gz` — genau wie in den BMA-`train.tsv`/`predict.tsv`.
# Beim Training stand in `config.yaml` dennoch `preprocessing.crop.mode:
# center` mit `roi_size: [160, 212, 160]`; das Modell hat Input-Shape
# `(160, 212, 160, 1)`. Derselbe on-the-fly-Schritt vom vollen MNI-Volume auf
# das Modell-FOV muss hier für Predict/LRP passieren, sonst Shape-Fehler.
# Plots (Abschnitt I/K) zeigen wieder das volle `nu_mni152_flirt`-FOV.
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
    """Load NIfTI and apply on-the-fly crop from ``preprocessing.crop``."""
    norm = float(cfg.preprocessing.normalization_factor)
    loader = str(getattr(cfg.data, "loader", "nifti-nibabel")).lower()
    crop_cfg = parse_crop_cfg(cfg.preprocessing)
    expected_shape = list(cfg.model.input.shape_cropped)

    def _load(path: str) -> np.ndarray:
        if loader == "nifti-native":
            vol = _load_single_volume_native(
                path,
                norm,
                crop_cfg=crop_cfg,
                expected_shape=expected_shape,
            )
        else:
            vol = _load_single_volume(
                path,
                norm,
                crop_cfg=crop_cfg,
                expected_shape=expected_shape,
            )
        if vol.ndim == 3:
            vol = np.expand_dims(vol, axis=-1)
        assert_volume_shape(
            vol,
            expected_shape,
            path=path,
            context="nach Crop",
        )
        return vol.astype(np.float32)

    return _load


LRP_STRATEGY_a2b1 = LRPStrategy(
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

LRP_STRATEGY_a1b0 = LRPStrategy(
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

LRP_STRATEGY = LRP_STRATEGY_a2b1
#LRP_STRATEGY = LRP_STRATEGY_a1b0


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
crop_cfgs: dict[str, dict] = {}
zscore_params: dict[str, dict[str, float | bool]] = {}
for key, meta in DATASETS.items():
    model, cfg, delta = load_keras_model(meta["run_dir"])
    models[key] = model
    load_volume_fns[key] = make_load_volume(cfg)
    crop_cfgs[key] = parse_crop_cfg(cfg.preprocessing)
    zscore_params[key] = zscore_inverse_params(meta["run_dir"])
    zp = zscore_params[key]
    print(
        f"[{key}] geladen  Δw={delta:.3g}  loader={cfg.data.loader}  "
        f"crop={crop_cfgs[key]['mode']} (nur intern fürs Netz)  "
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
# Jedes Modell sagt für dieselben Holdout-Subjects die jeweilige BMA
# avg-intensity voraus. Der Dateipfad und der wahre Wert kommen aus der
# gemeinsamen `predict.tsv` (Originalskala, Spalte je `pred_var`).
#
# Die Modellausgabe liegt im z-Score-Raum, wenn in `config_training.yaml`
# `normalisation.use: true` steht. Dann wird jede Vorhersage mit
# `y = z · σ + μ` auf die Intensitätsskala zurückgerechnet. Dieselben Werte
# gehen in Scatter, Tabelle und die 2×2-Figur.
#

# %%
def predict_subjects(dataset_key: str, sids: list[str]) -> pd.DataFrame:
    model = models[dataset_key]
    load_vol = load_volume_fns[dataset_key]
    pred_var = DATASETS[dataset_key]["pred_var"]
    rows: list[dict[str, object]] = []
    for sid in sids:
        row = lookup[dataset_key][sid]
        # immer großes nu_mni152_flirt (kein cropped.nii.gz)
        path = Path(INPUT_TMPL.format(sid=sid))
        if not path.is_file():
            raise FileNotFoundError(
                f"[{dataset_key}/{sid}] nu_mni152_flirt fehlt: {path}"
            )
        y_true = float(row[pred_var])
        vol = load_vol(str(path))
        y_pred_net = float(np.squeeze(model.predict(np.expand_dims(vol, 0), verbose=0)))
        y_pred = float(
            np.squeeze(inverse_zscore_values(y_pred_net, zscore_params[dataset_key]))
        )
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
            "pred_var": DATASETS[key]["pred_var"],
            "n": len(df),
            "MAE": mae,
            "pearson_r": r_val,
        }
    )
    print(f"[{key}] n={len(df)}  MAE={mae:.4f}  r={r_val:.3f}")

metrics_df = pd.DataFrame(metrics_rows)


# %% [markdown]
# ## G. Scatter-Plot (2 Panels)
#
# Ein Panel je BMA-Label. Jeder Punkt ist ein Holdout-Subject
# (`N_SUBJ_PRED`). Im Titel stehen MAE und Pearson-Korrelation.
# Subject `IDX_PRED` ist rot umrandet. Die Diagonale ist die ideale Vorhersage.
#

# %%
fig, axes = plt.subplots(1, len(DATASETS), figsize=(4.8 * len(DATASETS), 4.2))
if len(DATASETS) == 1:
    axes = np.asarray([axes])
fig.suptitle(
    f"Wahr vs. prädiziert  ·  BMA avg-intensity  ·  n={N_SUBJ_PRED}",
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
    ax.set_xlabel("wahre BMA avg-intensity")
    ax.set_ylabel("prädizierte BMA avg-intensity")
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
# ## H. Tabelle: wahre vs. prädizierte BMA avg-intensity
#
# Für beide Datensätze und die `N_SUBJ_PRED` Subjects: wahrer Wert aus der
# Holdout-TSV und zurückgerechnete Vorhersage des jeweiligen Modells.
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
print("Wahr vs. prädiziert (BMA avg-intensity):")
display(table_df.round(4))
display(metrics_df.round(4))


# %% [markdown]
# ## I. Input-Bilder für `IDX_PRED`
#
# Es werden ausschließlich die großen Volumes
# `/mnt/ceph/data/ukb/recon/<subject-id>/mri/nu_mni152_flirt.nii.gz`
# geladen und geplottet (kein `cropped.nii.gz`). Sagittaler Schnitt: `x=70`
# im vollen MNI-FOV (`182×218×182`).
#

# %%
def load_full_nifti(path: Path) -> np.ndarray:
    """Load full-size ``nu_mni152_flirt.nii.gz`` without spatial crop."""
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32).squeeze()


def sagittal_slc(vol: np.ndarray, cx: int) -> np.ndarray:
    return np.rot90(np.asarray(vol).squeeze()[cx])


def crop_window(full_shape: tuple[int, ...], crop_cfg: dict) -> tuple[slice, slice, slice]:
    """Return the 3 spatial slices that map cropped FOV back into full MNI space."""
    mode = crop_cfg["mode"]
    d, h, w = (int(x) for x in full_shape[:3])
    if mode == "none":
        return slice(0, d), slice(0, h), slice(0, w)
    if mode == "slices":
        (d0, d1), (h0, h1), (w0, w1) = crop_cfg["slices"]
        return slice(d0, d1), slice(h0, h1), slice(w0, w1)
    # center
    rd, rh, rw = (int(x) for x in crop_cfg["roi_size"])
    d0, h0, w0 = (d - rd) // 2, (h - rh) // 2, (w - rw) // 2
    return slice(d0, d0 + rd), slice(h0, h0 + rh), slice(w0, w0 + rw)


def pad_crop_to_full(
    cropped: np.ndarray,
    full_shape: tuple[int, ...],
    crop_cfg: dict,
) -> np.ndarray:
    """Embed a cropped LRP map into the full ``nu_mni152_flirt`` grid (zeros outside)."""
    R = np.asarray(cropped, dtype=np.float32).squeeze()
    out = np.zeros(tuple(int(x) for x in full_shape[:3]), dtype=np.float32)
    sl = crop_window(full_shape, crop_cfg)
    target = out[sl]
    if target.shape != R.shape:
        raise ValueError(
            f"Crop-Fenster {target.shape} passt nicht zur Heatmap {R.shape} "
            f"(full={full_shape[:3]}, crop={crop_cfg})."
        )
    out[sl] = R
    return out


plot_paths: dict[str, Path] = {}
plot_vols: dict[str, np.ndarray] = {}
plot_true: dict[str, float] = {}
plot_pred: dict[str, float] = {}

for key in DATASETS:
    vol_path = Path(INPUT_TMPL.format(sid=subject_id_plot))
    if not vol_path.is_file():
        raise FileNotFoundError(
            f"Erwarte nu_mni152_flirt.nii.gz, fehlt: {vol_path}"
        )
    if "nu_mni152_flirt" not in vol_path.name:
        raise ValueError(f"Unerwartete Datei (kein nu_mni152_flirt): {vol_path}")
    vol = load_full_nifti(vol_path)
    pred_row = preds_by_dataset[key].set_index("subject_id").loc[subject_id_plot]
    plot_paths[key] = vol_path
    plot_vols[key] = vol
    plot_true[key] = float(pred_row["y_true"])
    plot_pred[key] = float(pred_row["y_pred"])
    print(f"[{key}] volume: {vol_path}")
    print(
        f"        true={plot_true[key]:.4f}  pred={plot_pred[key]:.4f}  "
        f"shape={vol.shape} (volles MNI-FOV)"
    )

cx = int(np.clip(SAGITTAL_X, 0, next(iter(plot_vols.values())).shape[0] - 1))
print(f"sagittal x = {cx} (volles nu_mni152_flirt-FOV)")


# %% [markdown]
# ## J. LRP-Heatmaps neu berechnen (unnormiert)
#
# Für Subject `IDX_PRED` werden die zwei Heatmaps bei jedem Lauf neu berechnet.
# Eingabe ist weiterhin `nu_mni152_flirt.nii.gz`; fürs Netz wird nur intern
# gecroppt. Die Heatmap wird danach wieder in das volle MNI-Gitter eingebettet,
# damit der Plot bei `x=70` zum Input passt. Gespeicherte Dateien werden nicht
# geladen.
#
# War `normalisation.use: true`, wird jede Heatmap mit dem Trainings-σ
# multipliziert (`R · σ`). Das ist der lineare Teil der z-Score-Rücktransformation.
# μ wird nicht auf einzelne Voxel addiert: der Mittelwert ist ein globaler Offset
# der Vorhersage, keine räumliche Relevanz. Danach gilt
# `ΣR · σ + μ ≈` zurückgerechnete Vorhersage.
# Anschließend wird die Dauer beider Berechnungen ausgegeben.
#

# %%
heatmaps: dict[str, np.ndarray] = {}
heatmaps_cropped: dict[str, np.ndarray] = {}
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
    # Quelldatei = großes nu_mni152_flirt; Crop nur im Loader fürs Netz
    vol = load_vol(str(plot_paths[key]))
    R = np.asarray(lrp(np.expand_dims(vol, 0))[0].numpy(), dtype=np.float64).squeeze()
    zp = zscore_params[key]
    if zp["use"]:
        R = R * float(zp["std"])
    R = np.asarray(R, dtype=np.float32)
    heatmaps_cropped[key] = R
    heatmaps[key] = pad_crop_to_full(R, plot_vols[key].shape, crop_cfgs[key])
    scale_note = (
        f"  · R·σ (σ={float(zp['std']):.6g})"
        if zp["use"]
        else "  · keine z-score-Skalierung"
    )
    print(
        f"[{key}] LRP fertig  cropped={heatmaps_cropped[key].shape}  "
        f"full={heatmaps[key].shape}  "
        f"|R|_max={float(np.nanmax(np.abs(heatmaps[key]))):.4g}  "
        f"ΣR={float(np.sum(heatmaps[key])):.4g}{scale_note}"
    )
elapsed_s = time.perf_counter() - t0
print(
    f"\nDauer Berechnung aller {len(DATASETS)} Heatmaps: {elapsed_s:.2f} s "
    f"({elapsed_s / 60.0:.2f} min)"
)


# %% [markdown]
# ## K. Figur 2×2: Inputs (oben) + LRP-Heatmaps (unten)
#
# Zwei Spalten, eine je BMA-Label **(1)** / **(2)**. Beide Reihen nutzen das
# volle `nu_mni152_flirt`-FOV (nicht `cropped.nii.gz`).
#
# - Obere Reihe: volles Input-Bild, sagittal `x=70`, eigene Colorbar je Panel.
# - Untere Reihe: LRP-Relevanzen (bei aktivem z-Score `R · σ`), in dasselbe
#   volle Gitter zurückgeschrieben, eigene Colorbar je Panel
#   (`vmin/vmax = ±P99.5(|R|)` nur für die Farbskala).
# - In jedem Panel: Subject-ID, wahre und zurückgerechnete prädizierte
#   BMA avg-intensity (`z · σ + μ`, falls `normalisation.use: true`).
#

# %%
n_cols = len(DATASETS)
fig, axes = plt.subplots(2, n_cols, figsize=(5.2 * n_cols, 9.6))
if n_cols == 1:
    axes = np.asarray(axes).reshape(2, 1)
fig.suptitle(
    (
        f"Subject {subject_id_plot}  ·  IDX_PRED={IDX_PRED}/{N_SUBJ_PRED}  ·  "
        f"sagittal x={cx}  ·  nu_mni152_flirt  ·  BMA avg-intensity"
    ),
    fontsize=11,
)

keys = list(DATASETS.keys())
im_vols = []
for col, key in enumerate(keys):
    ax = axes[0, col]
    vol = plot_vols[key]
    pos = vol[vol > 0]
    vmax_i = float(np.percentile(pos, 99.5)) if pos.size else 1.0
    im = ax.imshow(sagittal_slc(vol, cx), cmap="gray", vmin=0.0, vmax=vmax_i)
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
    heat = heatmaps[key]  # volles MNI-Gitter
    abs_max = float(np.nanmax(np.abs(heat))) or 1.0
    vmax_r = float(np.nanpercentile(np.abs(heat), 99.5)) or abs_max
    im = ax.imshow(
        sagittal_slc(heat, cx),
        cmap="RdBu_r",
        vmin=-vmax_r,
        vmax=vmax_r,
    )
    ax.set_title(
        f"LRP · {DATASETS[key]['label']}\n"
        f"{subject_id_plot}\n"
        f"wahr={plot_true[key]:.4f}  präd={plot_pred[key]:.4f}  "
        f"|R|_max={abs_max:.3g}",
        fontsize=9,
    )
    ax.axis("off")
    im_lrps.append(im)

for col, im in enumerate(im_vols):
    cbar = fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
    cbar.set_label("Intensität")
for col, im in enumerate(im_lrps):
    cbar = fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
    cbar.set_label("LRP-Relevanz")

fig.tight_layout(rect=[0, 0.02, 1, 0.93])

if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)

print(
    f"Heatmap-Berechnung (Abschnitt J): {elapsed_s:.2f} s für alle {len(DATASETS)} Modelle "
    f"(Subject {subject_id_plot})."
)


# %% [markdown]
# ## L. Gruppen-LRP für `N_group` Subjects (Sum-Norm)
#
# Zufällige Ziehung von **`N_group` Subjects** aus der gemeinsamen Holdout-
# `predict.tsv` (Seed `GROUP_SEED`). Dieselbe Subject-Liste gilt für beide
# BMA-Modelle.
#
# **Gruppen-MRI:** voxelweiser Mittelwert der vollen `nu_mni152_flirt`-Volumes
# über die `N_group` Subjects (identisch für beide Modelle).
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
# H = \sum_{s=1}^{N_{\mathrm{group}}} R^{(s)}_{\mathrm{norm}}
# $$
#
# Die Sum-Norm macht eine etwaige Label-z-Score-Skalierung ($R \cdot \sigma$)
# irrelevant. Accumuliert wird im Crop-Raum des Netzes; für die Figur wird $H$
# in das volle `nu_mni152_flirt`-Gitter zurückgeschrieben.
#

# %%
def normalize_lrp_sum_abs(R: np.ndarray) -> np.ndarray:
    """R_norm_i = |R_i| / sum(|R|); Summe über Voxel = 1."""
    abs_r = np.abs(np.asarray(R, dtype=np.float64))
    denom = float(np.sum(abs_r))
    if denom <= 0.0:
        raise RuntimeError("LRP-Summe |R| ist 0 — Normierung nicht möglich.")
    return (abs_r / denom).astype(np.float64)


# Zufällige Subject-Auswahl aus der gemeinsamen predict.tsv
_first_key_group = next(iter(DATASETS))
df_pool = labels_full[_first_key_group]
if len(df_pool) < int(N_group):
    raise RuntimeError(
        f"predict.tsv hat nur {len(df_pool)} Zeilen, N_group={N_group}."
    )
df_group = df_pool.sample(n=int(N_group), random_state=int(GROUP_SEED)).reset_index(
    drop=True
)
group_subject_ids = df_group["participant_id"].astype(str).tolist()
print(
    f"Gruppen-Subjects: N_group={N_group}, seed={GROUP_SEED}, "
    f"erste IDs={group_subject_ids[:5]}"
)

group_heatmaps: dict[str, np.ndarray] = {}
group_heatmaps_full: dict[str, np.ndarray] = {}
group_meta: dict[str, dict[str, object]] = {}

# Referenz-Shape + Gruppen-MRI (Mittelwert der vollen Volumes)
_ref_sid = group_subject_ids[0]
_ref_full = load_full_nifti(Path(INPUT_TMPL.format(sid=_ref_sid)))
_ref_full_shape = _ref_full.shape

t0_group = time.perf_counter()
vol_acc = np.zeros(_ref_full_shape, dtype=np.float64)
for sid in tqdm(group_subject_ids, desc="group-MRI"):
    vol_path = Path(INPUT_TMPL.format(sid=sid))
    if not vol_path.is_file():
        raise FileNotFoundError(f"[group-MRI/{sid}] Volume fehlt: {vol_path}")
    vol_full = load_full_nifti(vol_path)
    if vol_full.shape != _ref_full_shape:
        raise ValueError(
            f"[group-MRI/{sid}] Shape {vol_full.shape} ≠ {_ref_full_shape}"
        )
    vol_acc += vol_full.astype(np.float64)

group_mri_mean = (vol_acc / float(len(group_subject_ids))).astype(np.float32)
print(
    f"Gruppen-MRI: mean über n={len(group_subject_ids)}  "
    f"shape={group_mri_mean.shape}  "
    f"max={float(np.max(group_mri_mean)):.4g}"
)

for key in DATASETS:
    model = models[key]
    load_vol = load_volume_fns[key]
    lrp = LRP(
        model,
        layer=len(model.layers) - 1,
        idx=0,
        strategy=LRP_STRATEGY,
    )

    heat_acc: np.ndarray | None = None
    n_ok = 0
    first_sids: list[str] = []

    for sid in tqdm(group_subject_ids, desc=f"group-LRP [{key}]"):
        vol_path = Path(INPUT_TMPL.format(sid=sid))
        if not vol_path.is_file():
            raise FileNotFoundError(f"[{key}/{sid}] Volume fehlt: {vol_path}")

        vol = load_vol(str(vol_path))
        R = lrp(np.expand_dims(vol, 0))[0].numpy()
        R_norm = normalize_lrp_sum_abs(R)

        if heat_acc is None:
            heat_acc = np.zeros_like(R_norm, dtype=np.float64)
        if R_norm.shape != heat_acc.shape:
            raise ValueError(
                f"[{key}/{sid}] Heatmap-Shape {R_norm.shape} ≠ {heat_acc.shape}"
            )

        heat_acc += R_norm
        n_ok += 1
        if len(first_sids) < 3:
            first_sids.append(sid)

    assert heat_acc is not None
    sum_H = float(np.sum(heat_acc))
    max_H = float(np.max(heat_acc))

    group_heatmaps[key] = heat_acc.astype(np.float32)
    group_heatmaps_full[key] = pad_crop_to_full(
        group_heatmaps[key], _ref_full_shape, crop_cfgs[key]
    )
    group_meta[key] = {
        "n": n_ok,
        "sum_H": sum_H,
        "max_H": max_H,
        "first_sids": first_sids,
        "subject_ids": group_subject_ids,
    }
    print(
        f"[{key}] n={n_ok}  ΣH={sum_H:.4g} (≈ N_group={N_group})  "
        f"max(H)={max_H:.4g}  cropped={group_heatmaps[key].shape}  "
        f"full={group_heatmaps_full[key].shape}  erste IDs={first_sids}"
    )

elapsed_group_s = time.perf_counter() - t0_group
print(
    f"\nDauer Gruppen-MRI + LRP ({len(DATASETS)}× N_group={N_group}): "
    f"{elapsed_group_s:.1f} s ({elapsed_group_s / 60.0:.2f} min)"
)


# %% [markdown]
# ## M. Figur 2×2: Gruppen-MRI (oben) + Gruppen-LRP (unten)
#
# Zwei Spalten, eine je BMA-Label **(1)** / **(2)**. Beide Reihen nutzen das
# volle `nu_mni152_flirt`-FOV.
#
# - Obere Reihe: Gruppen-MRI (voxelweiser Mittelwert über `N_group` Subjects),
#   sagittal `x=70`, eigene Colorbar je Panel.
# - Untere Reihe: Gruppen-LRP (Sum-Norm), Werte ≥ 0, eigene Colorbar je Panel.
#
# $$
# H = \sum_{s=1}^{N_{\mathrm{group}}} R^{(s)}_{\mathrm{norm}}
# \qquad\text{mit}\qquad
# \sum_i R^{(s)}_{\mathrm{norm},i} = 1
# $$
#

# %%
n_cols = len(DATASETS)
fig, axes = plt.subplots(2, n_cols, figsize=(5.2 * n_cols, 9.6))
if n_cols == 1:
    axes = np.asarray(axes).reshape(2, 1)
fig.suptitle(
    (
        f"Gruppen-MRI + LRP (Sum-Norm)  ·  $N_{{\mathrm{{group}}}}={N_group}$"
        f"  ·  seed={GROUP_SEED}  ·  sagittal $x={SAGITTAL_X}$\n"
        + r"$R^{(s)}_{\mathrm{norm},i}=\dfrac{|R^{(s)}_i|}{\sum_j |R^{(s)}_j|}"
        r"\;\Rightarrow\;"
        r"\sum_i R^{(s)}_{\mathrm{norm},i}=1$"
        "\n"
        r"$H=\sum_{s=1}^{N_{\mathrm{group}}} R^{(s)}_{\mathrm{norm}}$"
    ),
    fontsize=11,
)

keys = list(DATASETS.keys())
cx_g = int(np.clip(SAGITTAL_X, 0, group_mri_mean.shape[0] - 1))
pos = group_mri_mean[group_mri_mean > 0]
vmax_i = float(np.percentile(pos, 99.5)) if pos.size else 1.0

im_vols = []
for col, key in enumerate(keys):
    ax = axes[0, col]
    im = ax.imshow(
        sagittal_slc(group_mri_mean, cx_g),
        cmap="gray",
        vmin=0.0,
        vmax=vmax_i,
    )
    ax.set_title(
        f"{DATASETS[key]['label']}\n"
        f"Gruppen-MRI (mean, n={N_group})",
        fontsize=10,
    )
    ax.axis("off")
    im_vols.append(im)

im_groups = []
for col, key in enumerate(keys):
    ax = axes[1, col]
    H = group_heatmaps_full[key]
    vmax_h = float(np.nanmax(H)) or 1.0
    im = ax.imshow(
        sagittal_slc(H, cx_g),
        cmap="hot",
        vmin=0.0,
        vmax=vmax_h,
    )
    meta = group_meta[key]
    ax.set_title(
        f"Gruppen-LRP · {DATASETS[key]['label']}\n"
        f"n={meta['n']}  ΣH={meta['sum_H']:.2f}  max={meta['max_H']:.3g}",
        fontsize=10,
    )
    ax.axis("off")
    im_groups.append(im)

for col, im in enumerate(im_vols):
    cbar = fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)
    cbar.set_label("Intensität (mean)")
for col, im in enumerate(im_groups):
    cbar = fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
    cbar.set_label(r"$\sum_s R^{(s)}_{\mathrm{norm}}$")

fig.tight_layout(rect=[0, 0.02, 1, 0.88])

if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)

print(
    f"Gruppenfigur fertig (N_group={N_group}, Berechnung {elapsed_group_s:.1f} s)."
)

