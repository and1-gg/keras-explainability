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
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: py-uv_keras-xai (uv)
#     language: python
#     name: py-uv_keras-xai
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.8
# ---

# %%
from __future__ import annotations

import os
import sys
from pathlib import Path

# Inline-Backend *vor* dem pyplot-Import setzen (sonst bleibt oft Agg aktiv
# und plt.show() zeigt im Notebook nichts, obwohl savefig die PNG schreibt).
# %matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display
from omegaconf import OmegaConf, open_dict
from scipy.stats import pearsonr

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
    "ukb": DATA_ROOT / "ukb",
}

#DATASET = "ixi"  # <- hier umschalten
#DATASET = "ukb"  # <- hier umschalten (nutzt automatisch Holdout aus predict.tsv)
#DATASET = "ds004711_AgeRisk"  # <- hier umschalten
DATASET = "ds003114_OvarianHormones"  # <- hier umschalten
N_SUBJECTS = 50
PRED_BATCH_SIZE = 8

# Nur für UKB: Holdout-Subjects aus dem offiziellen predict-Split (nicht train.tsv).
# Die ersten N_SUBJECTS Zeilen sind disjoint zum Training (siehe pyment-and1).
UKB_HOLDOUT_PREDICT_TSV = Path(
    "~/git-repos/pyment-and1/training_runs/input_files/mri/"
    "right_whole_thalamus/predict.tsv"
).expanduser().resolve()

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
from pybrainmetrics.modeling.train import _build_single_device_model

print("keras-xai:     ", keras_xai_root)
print("pybrainmetrics:", pybm_src)
print("TensorFlow:    ", tf.__version__)


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


def load_dataset_labels(
    dataset_dir: Path,
    pred_var: str,
    n_subjects: int,
    *,
    labels_file: Path | None = None,
) -> pd.DataFrame:
    """Labels + filepaths für Prediction/Heatmaps.

    Standard: subjects_dl_input.tsv im Datensatzordner, sonst FreeSurfer-Stats.
    Für UKB-Holdout: explizit ``labels_file`` setzen (z. B. predict.tsv).
    """
    if labels_file is not None:
        if not labels_file.is_file():
            raise FileNotFoundError(labels_file)
        df = pd.read_csv(labels_file, sep=None, engine="python")
        df = _normalize_label_columns(df, pred_var)
        df = df.dropna(subset=[pred_var, "filepath"]).head(int(n_subjects))
        print(f"Quelle: {labels_file} (Holdout), n={len(df)}")
        return df

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


def print_subject_filepaths(df: pd.DataFrame) -> None:
    """Subject-ID und vollständigen Pfad zu cropped.nii.gz ausgeben."""
    id_col = "participant_id" if "participant_id" in df.columns else "subject-id"
    print(f"Subjects + cropped.nii.gz (n={len(df)}):")
    for _, row in df.iterrows():
        print(f"  {row[id_col]}  {row['filepath']}")


cfg = OmegaConf.load(CONFIG_PATH)
pred_var = cfg.data.prediction_variable

ukb_labels_file = UKB_HOLDOUT_PREDICT_TSV if DATASET == "ukb" else None
df = load_dataset_labels(
    DATASETS[DATASET],
    pred_var,
    N_SUBJECTS,
    labels_file=ukb_labels_file,
)
print_subject_filepaths(df)
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

# %%
# LRP/Explainability verträgt sich nicht mit MirroredStrategy (unbekannte
# TensorShapes). Deshalb Single-Device-Modell für die Heatmaps.
model = _build_single_device_model(cfg)

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

print("Layer:", len(model.layers), "→ Zielschicht:", model.layers[-1].name)

# %%
# Ausgabeordner + Volume-Loader (gleiche Pipeline wie Training/Prediction).
# Normierung: cfg.preprocessing.normalization_factor (= 1.0 für dieses Modell),
# NICHT sigma=255 wie im Brain-Age-Demo-Notebook.

from pybrainmetrics.data.dataset import (
    _load_single_volume,
    _load_single_volume_native,
)

target_dir = (
    keras_xai_root
    / "output"
    / "notebooks"
    / "create_heatmaps_for_right_thalamus_model"
    / DATASET
)
target_dir.mkdir(parents=True, exist_ok=True)
print("PNG-Vorschau →", target_dir)

heatmaps_dir = RUN_DIR / "heatmaps"
heatmaps_dir.mkdir(parents=True, exist_ok=True)
print("NIfTI-Heatmaps →", heatmaps_dir)

NORM_FACTOR = float(cfg.preprocessing.normalization_factor)
LOADER = str(getattr(cfg.data, "loader", "nifti-nibabel")).lower()
use_native = LOADER == "nifti-native"


def load_volume(path: str) -> np.ndarray:
    """Lädt cropped.nii.gz → (167, 212, 160, 1) float32."""
    if use_native:
        vol = _load_single_volume_native(path, NORM_FACTOR)
    else:
        vol = _load_single_volume(path, NORM_FACTOR)
    if vol.ndim == 3:
        vol = np.expand_dims(vol, axis=-1)
    return vol.astype(np.float32)


N_HEATMAP_PLOTS = 3  # PNG-Vorschau für die ersten N Subjekte
print(f"Subjekte LRP + NIfTI ({DATASET}): n={len(df)}")
print(f"PNG-Vorschau: erste {min(N_HEATMAP_PLOTS, len(df))} Subjekte")
print(df[["participant_id", "filepath", pred_var]].head())

# %%
# LRP-Erklärer + Heatmap-Visualisierung
# (Kernschritte aus Explain_brain_age_predictions, Abschnitt 6)
#
# Composite-Strategie für SFCN (7 gewichtete Schichten: 6×Conv3D + Dense):
#   flat → flat → αβ → αβ → αβ → αβ → ε
# Maskierung: Relevanz außerhalb des Gehirns (Voxel==0) auf 0 setzen,
# weil die flat-Regel sonst Hintergrund beleuchtet.

import nibabel as nib
from tqdm import tqdm

from explainability import LRP, LRPStrategy

alpha = 2
beta = 1

strategy = LRPStrategy(
    layers=[
        {"flat": True},
        {"flat": True},
        {"alpha": alpha, "beta": beta},
        {"alpha": alpha, "beta": beta},
        {"alpha": alpha, "beta": beta},
        {"alpha": alpha, "beta": beta},
        {"epsilon": 0.25},
    ]
)

lrp = LRP(model, layer=len(model.layers) - 1, idx=0, strategy=strategy)
print("LRP-Schichten:", len(lrp.layers))


def mask_explanation(volume: np.ndarray, explanation: np.ndarray) -> np.ndarray:
    """Nullt Relevanz außerhalb des Gehirns (wie im Brain-Age-Notebook)."""
    x = volume.squeeze()
    expl = explanation.squeeze().astype(np.float32)
    return expl * (x != 0).astype(np.float32)


def save_heatmap_nifti(
    explanation: np.ndarray,
    reference_nii_path: str,
    out_path: Path,
) -> None:
    """Speichert LRP-Heatmap im Gitter/Affine des Referenz-cropped.nii.gz."""
    ref = nib.load(reference_nii_path)
    data = np.asarray(explanation, dtype=np.float32).squeeze()
    ref_shape = ref.shape
    if data.shape != ref_shape:
        raise ValueError(
            f"Shape-Mismatch Heatmap {data.shape} vs Referenz {ref_shape} "
            f"({reference_nii_path})"
        )
    header = ref.header.copy()
    header.set_data_dtype(np.float32)
    img = nib.Nifti1Image(data, affine=ref.affine, header=header)
    nib.save(img, str(out_path))


def plot_lrp_heatmap(
    volume: np.ndarray,
    explanation: np.ndarray,
    *,
    title: str,
    save_path: Path | None = None,
):
    """Sagittal / koronal / axial: Anatomie + LRP um das relevanteste Voxel."""
    x = volume.squeeze()
    expl = mask_explanation(volume, explanation)
    vmax = float(np.amax(np.abs(expl)))
    if vmax > 0:
        expl = expl / vmax

    peak = np.unravel_index(int(np.argmax(np.abs(expl))), expl.shape)
    # Randnahe Peaks: Schnittindex in gültigen Bereich clippen
    peak = tuple(
        int(np.clip(p, 4, s - 5)) for p, s in zip(peak, expl.shape)
    )
    fig, ax = plt.subplots(6, 8, figsize=(16, 15))
    fig.suptitle(title, fontsize=12)

    last_hm = None
    for i in range(-4, 4):
        col = i + 4
        ax[0][col].imshow(np.rot90(x[peak[0] + i]), cmap="Greys_r")
        ax[0][col].axis("off")
        last_hm = ax[1][col].imshow(
            np.rot90(expl[peak[0] + i]), cmap="seismic", clim=(-1, 1)
        )
        ax[1][col].axis("off")
        ax[2][col].imshow(np.rot90(x[:, peak[1] + i]), cmap="Greys_r")
        ax[2][col].axis("off")
        ax[3][col].imshow(
            np.rot90(expl[:, peak[1] + i]), cmap="seismic", clim=(-1, 1)
        )
        ax[3][col].axis("off")
        ax[4][col].imshow(x[:, :, peak[2] + i], cmap="Greys_r")
        ax[4][col].axis("off")
        ax[5][col].imshow(
            expl[:, :, peak[2] + i], cmap="seismic", clim=(-1, 1)
        )
        ax[5][col].axis("off")

    ax[0][0].set_ylabel("sagittal\nMRT", fontsize=9)
    ax[1][0].set_ylabel("sagittal\nLRP", fontsize=9)
    ax[2][0].set_ylabel("koronal\nMRT", fontsize=9)
    ax[3][0].set_ylabel("koronal\nLRP", fontsize=9)
    ax[4][0].set_ylabel("axial\nMRT", fontsize=9)
    ax[5][0].set_ylabel("axial\nLRP", fontsize=9)

    fig.tight_layout(rect=[0, 0, 0.90, 0.98])
    cbar = fig.colorbar(
        last_hm,
        ax=ax.ravel().tolist(),
        fraction=0.03,
        pad=0.02,
        shrink=0.55,
    )
    cbar.set_ticks([-1.0, 0.0, 1.0])
    cbar.set_ticklabels(["−1", "0", "+1"])
    cbar.set_label(
        "LRP-Relevanz (normiert auf [−1, 1])\n"
        "Rot: treibt die Vorhersage nach oben\n"
        "Blau: zieht die Vorhersage nach unten",
        fontsize=10,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print("gespeichert:", save_path)
    display(fig)
    plt.close(fig)
    return expl, peak


# %%
# LRP für alle N_SUBJECTS: NIfTI nach RUN_DIR/heatmaps, PNG-Vorschau für erste N.

saved_niftis: list[Path] = []

for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="LRP")):
    sid = str(row["participant_id"])
    path = str(row["filepath"])
    y_true = float(row[pred_var])

    vol = load_volume(path)
    X = np.expand_dims(vol, axis=0)
    y_pred = float(np.squeeze(model.predict(X, verbose=0)))
    R = lrp(X)[0].numpy()
    R_masked = mask_explanation(vol, R)

    nii_path = heatmaps_dir / f"{DATASET}_{sid}.nii.gz"
    save_heatmap_nifti(R_masked, path, nii_path)
    saved_niftis.append(nii_path)

    print(
        f"{sid}: true={y_true:.1f}  pred={y_pred:.1f}  "
        f"sum(R)={float(np.sum(R_masked)):.3f}  →  {nii_path.name}"
    )

    if i < N_HEATMAP_PLOTS:
        plot_lrp_heatmap(
            vol,
            R,
            title=f"{DATASET}  {sid}  true={y_true:.1f}  pred={y_pred:.1f}",
            save_path=target_dir / f"lrp_heatmap_{sid}.png",
        )

print(f"Fertig. {len(saved_niftis)} NIfTI-Heatmaps unter: {heatmaps_dir}")
print(f"PNG-Vorschau unter: {target_dir}")

# %%
