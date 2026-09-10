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
# # LRP-Analyse: rechtes Thalamus-Volumen (CNN)
#
# Dieses Notebook prüft, **wo** ein auf `Right-Whole_thalamus` trainiertes 3D-CNN
# seine Vorhersage herleitet (Layer-wise Relevance Propagation).
#
# **Teil A — Original-Modell** (`training_run_21h19m18s_20aug2026`):
#
# 1. `N_SUBJECTS` Holdout-Fälle aus **IXI** und **UKB** laden
# 2. Vorhersagen + LRP-Heatmaps berechnen
# 3. FreeSurfer-`aseg` als Thalamus-Maske nach MNI152 bringen
# 4. True vs. Predicted
# 5. Intensitäts-QC (3D + Histogramm) des ersten ungejitterten Subjects
# 6. Anteil der Relevanz **im Thalamus** vs. außerhalb auswerten
# 7. Interaktiver 3D-Plot (Gehirn + beide Thalamus-Masken + unnormierte LRP)
#    für das **erste** Subject von IXI und UKB
#
# **Teil B — Jitter-/Simulations-Framework** (nur UKB, außer B.5):
#
# 1. True vs. Predicted mit dem auf gejitterten Volumes trainierten Modell
# 2. Intensitäts-QC (3D + Histogramm) des ersten gejitterten UKB-Holdout-Subjects
# 3. QC der gejitterten Volumes (Nicht-Holdout)
# 4. LRP-Heatmaps auf gejitterten Holdout-Volumes (Original-Modell)
# 5. Interaktiver 3D-Plot der Jitter-Modell-LRP (erstes Subject IXI + UKB)
#
# **Teil C — Relevanzerhaltung Schicht für Schicht:**
#
# Für das **erste** Holdout-Subject jedes Datensatzes in `DATASET_DIRS` die Summe der
# LRP-Relevanz pro Schicht (Original-Modell vs. Jitter-Modell), inkl. Layer-Labels
# (`conv` / `maxpool` / `gap` / `dense`).

# %% [markdown]
# ## A.1. Imports

# %%
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
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
# ## A.2. Konfiguration
#
# - `N_SUBJECTS`: wie viele Fälle **pro Dataset** berechnet werden (Heatmap + Maske).
# - `N_PLOT_SUBJECTS`: wie viele davon geplottet werden (PNG + optional inline).
#   Darf höchstens `N_SUBJECTS` sein, Default ist 2.

# %%
RUN_DIR = Path(
    "~/data/nn-trainings/mri/Right-Whole_thalamus/"
    "training_run_21h19m18s_20aug2026"
).expanduser().resolve()

DATA_ROOT = Path("/mnt/ceph/data")
DATASET_DIRS = {
    "ixi": DATA_ROOT / "ixi",
    "ukb": DATA_ROOT / "ukb",
}
DATASETS = ["ixi", "ukb"]

N_SUBJECTS = 5
N_PLOT_SUBJECTS = 2
PRED_BATCH_SIZE = 8
SHOW_PLOTS_INLINE = True

UKB_HOLDOUT_PREDICT_TSV = Path(
    "~/git-repos/pyment-and1/training_runs/input_files/mri/"
    "right_whole_thalamus/predict.tsv"
).expanduser().resolve()

MNI152_1MM = Path("/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz")
MODEL_PATH = RUN_DIR / "model.keras"
CONFIG_PATH = RUN_DIR / "config.yaml"

N_PLOT_SUBJECTS = int(min(N_PLOT_SUBJECTS, N_SUBJECTS))
if N_PLOT_SUBJECTS < 1:
    raise ValueError("N_PLOT_SUBJECTS muss >= 1 sein.")

unknown = [d for d in DATASETS if d not in DATASET_DIRS]
if unknown:
    raise ValueError(f"Unbekannte DATASETS={unknown!r}. Erlaubt: {list(DATASET_DIRS)}")
for p, name in (
    (MODEL_PATH, "Modell"),
    (CONFIG_PATH, "Config"),
    (MNI152_1MM, "MNI152-Referenz"),
):
    if not p.is_file():
        raise FileNotFoundError(f"{name} fehlt: {p}")

print(f"RUN_DIR:          {RUN_DIR}")
print(f"DATASETS:         {DATASETS}")
print(f"N_SUBJECTS:       {N_SUBJECTS}")
print(f"N_PLOT_SUBJECTS:  {N_PLOT_SUBJECTS}")


# %% [markdown]
# ## A.3. Repo-Pfade (keras-explainability + pybrainmetrics)

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
            "pybrainmetrics nicht gefunden. PYBRAINMETRICS_SRC setzen."
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
print("TensorFlow:    ", tf.__version__)


# %% [markdown]
# ## A.4. Labels laden
#
# UKB kommt aus dem offiziellen **predict-Split** (Holdout, nicht Training).
# IXI aus `subjects_dl_input.tsv` bzw. FreeSurfer-Thalamus-Stats.

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
    if labels_file is not None:
        df = pd.read_csv(labels_file, sep=None, engine="python")
        df = _normalize_label_columns(df, pred_var)
        return df.dropna(subset=[pred_var, "filepath"]).head(int(n_subjects))

    for name in ("subjects_dl_input.tsv", "participants_dl_input.tsv"):
        cand = dataset_dir / name
        if cand.is_file():
            df = pd.read_csv(cand, sep=None, engine="python")
            df = _normalize_label_columns(df, pred_var)
            return df.dropna(subset=[pred_var, "filepath"]).head(int(n_subjects))

    vol_path = dataset_dir / "T1stats" / "ThalamicNuclei.volumes.txt_concat.stats"
    if not vol_path.is_file():
        raise FileNotFoundError(f"Keine Labels unter {dataset_dir}")
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
                "subject-id": sid,
                "participant_id": sid,
                "Left-Whole_thalamus": float(row["Left-Whole_thalamus"]),
                "Right-Whole_thalamus": float(row["Right-Whole_thalamus"]),
            }
        )
    if not rows:
        raise FileNotFoundError(f"Keine cropped.nii.gz zu Stats in {dataset_dir}")
    df = _normalize_label_columns(pd.DataFrame(rows), pred_var)
    return df.dropna(subset=[pred_var, "filepath"]).head(int(n_subjects))


cfg = OmegaConf.load(CONFIG_PATH)
pred_var = cfg.data.prediction_variable

dataset_labels: dict[str, pd.DataFrame] = {}
for dataset_id in DATASETS:
    labels_file = UKB_HOLDOUT_PREDICT_TSV if dataset_id == "ukb" else None
    df = load_dataset_labels(
        DATASET_DIRS[dataset_id],
        pred_var,
        N_SUBJECTS,
        labels_file=labels_file,
    )
    labels_tsv = RUN_DIR / f"{dataset_id}_predict_labels_n{len(df)}.tsv"
    df.to_csv(labels_tsv, sep="\t", index=False)
    dataset_labels[dataset_id] = df
    print(f"[{dataset_id}] n={len(df)}  ({labels_file or DATASET_DIRS[dataset_id]})")

with open_dict(cfg):
    cfg.paths.csv_dir = str(RUN_DIR)
    first_ds = DATASETS[0]
    cfg.data.predict_labels_file = str(
        RUN_DIR / f"{first_ds}_predict_labels_n{len(dataset_labels[first_ds])}.tsv"
    )
    if "prediction" not in cfg.training:
        cfg.training.prediction = {}
    cfg.training.prediction.batch_size = int(PRED_BATCH_SIZE)

print("pred_var:", pred_var)


# %% [markdown]
# ## A.5. Modell, Volume-Loader, LRP
#
# Single-Device-Modell (keine `MirroredStrategy`). LRP-Composite für SFCN:
# zwei `flat`-Schichten, vier αβ-Schichten, ε am Dense-Ausgang.
#
# Die Heatmaps bleiben **unmaskiert**: Relevanz auf Hintergrundvoxeln (`x==0`)
# wird **nicht** auf 0 gesetzt (kein `mask_explanation`). So bleibt die volle
# LRP-Ausgabe für Plot, NIfTI und ΣR-Auswertung erhalten — inkl. der durch
# `flat` in den Hintergrund fließenden Relevanz.

# %%
model = _build_single_device_model(cfg)
w_before = model.get_weights()[0].copy()
model.load_weights(str(MODEL_PATH))
delta = float(np.mean(np.abs(model.get_weights()[0] - w_before)))
if delta < 1e-9:
    raise RuntimeError("Gewichte wurden nicht geladen — Layer-Namen prüfen.")

NORM_FACTOR = float(cfg.preprocessing.normalization_factor)
LOADER = str(getattr(cfg.data, "loader", "nifti-nibabel")).lower()


def load_volume(path: str) -> np.ndarray:
    if LOADER == "nifti-native":
        vol = _load_single_volume_native(path, NORM_FACTOR)
    else:
        vol = _load_single_volume(path, NORM_FACTOR)
    if vol.ndim == 3:
        vol = np.expand_dims(vol, axis=-1)
    return vol.astype(np.float32)


strategy = LRPStrategy(
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
lrp = LRP(model, layer=len(model.layers) - 1, idx=0, strategy=strategy)


def save_heatmap_nifti(explanation: np.ndarray, reference_nii_path: str, out_path: Path) -> None:
    ref = nib.load(reference_nii_path)
    data = np.asarray(explanation, dtype=np.float32).squeeze()
    if data.shape != ref.shape:
        raise ValueError(f"Shape-Mismatch Heatmap {data.shape} vs Referenz {ref.shape}")
    header = ref.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data, affine=ref.affine, header=header), str(out_path))


print(f"Gewichtsdelta: {delta:.3g}  |  LRP-Schichten: {len(lrp.layers)}")


# %% [markdown]
# ## A.6. Thalamus-Maske (`aseg` → MNI152 → Crop)
#
# FreeSurfer-Labels: links=10, rechts=49. Crop wie Training-FOV `167×212×160`.
# Bereits vorhandene Masken werden übersprungen.

# %%
def _require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{what} fehlt: {path}")
    return path.resolve()


def setup_fsl_environment(fsldir: str | Path | None = None) -> None:
    env = os.environ
    root = Path(fsldir or env.get("FSLDIR") or "/usr/local/fsl").expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"FSLDIR existiert nicht: {root}")
    env["FSLDIR"] = str(root.resolve())
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    for sub in ("share/fsl/bin", "bin"):
        bin_dir = str(root / sub)
        if Path(bin_dir).is_dir() and bin_dir not in env.get("PATH", "").split(os.pathsep):
            env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")


def _fsl_bin(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    fsldir = Path(os.environ.get("FSLDIR", "/usr/local/fsl"))
    for sub in ("share/fsl/bin", "bin"):
        cand = fsldir / sub / name
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(f"FSL-Tool {name!r} nicht gefunden.")


def _run_cmd(cmd: list[str], *, label: str) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, env=os.environ)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"{label} fehlgeschlagen:\n{detail}") from exc


setup_fsl_environment()


@dataclass
class AsegThalamusMaskPipeline:
    dataset_id: str
    dataset_dir: Path
    heatmaps_dir: Path
    mni152_ref: Path = MNI152_1MM
    crop_slices: tuple[slice, slice, slice] = (slice(6, 173), slice(2, 214), slice(0, 160))

    def recon_mri_dir(self, subject_id: str) -> Path:
        return self.dataset_dir / "recon" / subject_id / "mri"

    def subject_work_dir(self, subject_id: str) -> Path:
        return self.heatmaps_dir / subject_id

    def final_mask_path(self, subject_id: str) -> Path:
        return (
            self.subject_work_dir(subject_id)
            / f"{subject_id}_aseg_thalamus_mask_mni152_cropped.nii.gz"
        )

    def resolve_inputs(self, subject_id: str) -> tuple[Path, Path]:
        mri = self.recon_mri_dir(subject_id)
        brainmask = _require_file(
            mri / "brainmask_reoriented.nii.gz",
            f"[{self.dataset_id}/{subject_id}] brainmask_reoriented",
        )
        aseg_candidates = [
            mri / "aseg_reoriented.nii.gz",
            self.dataset_dir / "recon" / subject_id / "aseg_reoriented.nii.gz",
        ]
        aseg = next((p for p in aseg_candidates if p.is_file()), None)
        if aseg is None:
            raise FileNotFoundError(
                f"[{self.dataset_id}/{subject_id}] aseg_reoriented fehlt. Geprüft: {aseg_candidates}"
            )
        return brainmask, aseg.resolve()

    def crop_nifti(self, input_path: Path, output_path: Path) -> None:
        img = nib.load(str(input_path))
        cropped = img.slicer[self.crop_slices]
        data = np.asarray(cropped.get_fdata(), dtype=np.float32)
        out = nib.Nifti1Image(data, cropped.affine, cropped.header)
        out.header.set_data_dtype(np.float32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out, str(output_path))

    def run(self, subject_id: str, *, skip_existing: bool = True) -> Path:
        work = self.subject_work_dir(subject_id)
        left_c = work / "aseg_mni152_left_thalamus_cropped.nii.gz"
        right_c = work / "aseg_mni152_right_thalamus_cropped.nii.gz"
        final = self.final_mask_path(subject_id)
        if skip_existing and final.is_file() and left_c.is_file() and right_c.is_file():
            return final

        flirt, fslmaths = _fsl_bin("flirt"), _fsl_bin("fslmaths")
        brainmask, aseg = self.resolve_inputs(subject_id)
        work.mkdir(parents=True, exist_ok=True)

        mni152_out = work / "mni152.nii.gz"
        xfm = work / "T1_to_mni152.mat"
        aseg_mni = work / "aseg_mni152.nii.gz"
        left = work / "aseg_mni152_left_thalamus.nii.gz"
        right = work / "aseg_mni152_right_thalamus.nii.gz"
        both = work / "aseg_mni152_thalamus_mask.nii.gz"

        _run_cmd(
            [flirt, "-in", str(brainmask), "-out", str(mni152_out),
             "-ref", str(self.mni152_ref), "-dof", "6", "-omat", str(xfm)],
            label=f"flirt brainmask→MNI152 ({subject_id})",
        )
        _run_cmd(
            [flirt, "-in", str(aseg), "-out", str(aseg_mni),
             "-ref", str(self.mni152_ref), "-dof", "6", "-applyxfm",
             "-init", str(xfm), "-interp", "nearestneighbour"],
            label=f"flirt aseg→MNI152 ({subject_id})",
        )
        _run_cmd(
            [fslmaths, str(aseg_mni), "-thr", "10", "-uthr", "10", "-bin", str(left)],
            label=f"fslmaths left ({subject_id})",
        )
        _run_cmd(
            [fslmaths, str(aseg_mni), "-thr", "49", "-uthr", "49", "-bin", str(right)],
            label=f"fslmaths right ({subject_id})",
        )
        _run_cmd(
            [fslmaths, str(left), "-add", str(right), "-bin", str(both)],
            label=f"fslmaths combine ({subject_id})",
        )
        self.crop_nifti(both, final)
        self.crop_nifti(left, left_c)
        self.crop_nifti(right, right_c)
        return _require_file(final, "finale Thalamus-Maske")



# %% [markdown]
# ## A.7. Heatmaps + Masken für alle Subjects
#
# Pro Dataset `N_SUBJECTS` Fälle. Overlay-Plots (LRP **unnormiert**, rechter Thalamus grün, linker Thalamus lila) für die ersten `N_PLOT_SUBJECTS`.
# Schnitte fest: sagittal `x=70`, koronal `y=104`, axial `z=78`.

# %%
def plot_lrp_overlay(
    heatmap: np.ndarray,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
    *,
    title: str,
    save_path: Path | None = None,
    show_inline: bool = True,
    sagittal_x: int = 70,
    coronal_y: int = 104,
    axial_z: int = 78,
) -> None:
    """Drei Schnitte: unnormierte LRP + linker (lila) / rechter (grün) Thalamus."""
    from matplotlib.patches import Patch

    heat = np.asarray(heatmap, dtype=np.float32).squeeze()
    left = (
        (np.asarray(left_mask, dtype=np.float32).squeeze() > 0)
        if left_mask is not None
        else np.zeros(heat.shape, dtype=bool)
    )
    right = (
        (np.asarray(right_mask, dtype=np.float32).squeeze() > 0)
        if right_mask is not None
        else np.zeros(heat.shape, dtype=bool)
    )
    nx, ny, nz = heat.shape
    cx = int(np.clip(sagittal_x, 0, nx - 1))
    cy = int(np.clip(coronal_y, 0, ny - 1))
    cz = int(np.clip(axial_z, 0, nz - 1))
    vmax = float(np.nanmax(np.abs(heat))) or 1.0
    color_left = (0.60, 0.20, 0.80, 0.40)
    color_right = (0.15, 0.65, 0.25, 0.40)

    def _rgba(mask_slc: np.ndarray, rgba: tuple[float, ...]) -> np.ndarray:
        out = np.zeros((*mask_slc.shape, 4), dtype=np.float32)
        out[mask_slc] = rgba
        return out

    slices = [
        (np.rot90(heat[cx]), np.rot90(left[cx]), np.rot90(right[cx]), f"sagittal x={cx}"),
        (np.rot90(heat[:, cy]), np.rot90(left[:, cy]), np.rot90(right[:, cy]), f"koronal y={cy}"),
        (heat[:, :, cz], left[:, :, cz], right[:, :, cz], f"axial z={cz}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    fig.suptitle(title, fontsize=10)
    im = None
    for ax, (h_slc, l_slc, r_slc, slc_title) in zip(axes, slices):
        im = ax.imshow(h_slc, cmap="seismic", vmin=-vmax, vmax=vmax)
        ax.imshow(_rgba(l_slc, color_left), interpolation="nearest")
        ax.imshow(_rgba(r_slc, color_right), interpolation="nearest")
        ax.set_title(slc_title, fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02, label="LRP-Relevanz")
    fig.legend(
        handles=[
            Patch(facecolor=color_left, edgecolor="none", label="linker Thalamus"),
            Patch(facecolor=color_right, edgecolor="none", label="rechter Thalamus"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    if save_path is not None:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    if show_inline:
        display(fig)
    plt.close(fig)


def plot_lrp_slices(
    volume: np.ndarray,
    explanation: np.ndarray,
    *,
    title: str,
    save_path: Path | None = None,
    show_inline: bool = True,
) -> None:
    """6×8-Schnitte um das LRP-Peak-Voxel (unnormiert) — Zoom in die 3D-Heatmap."""
    x = np.asarray(volume, dtype=np.float32).squeeze()
    expl = np.asarray(explanation, dtype=np.float32).squeeze()
    vmax = float(np.amax(np.abs(expl))) or 1.0
    peak = np.unravel_index(int(np.argmax(np.abs(expl))), expl.shape)
    peak = tuple(int(np.clip(p, 4, s - 5)) for p, s in zip(peak, expl.shape))

    fig, ax = plt.subplots(6, 8, figsize=(16, 15))
    fig.suptitle(title, fontsize=12)
    last_hm = None
    for i in range(-4, 4):
        col = i + 4
        ax[0][col].imshow(np.rot90(x[peak[0] + i]), cmap="Greys_r")
        ax[0][col].axis("off")
        last_hm = ax[1][col].imshow(
            np.rot90(expl[peak[0] + i]), cmap="seismic", vmin=-vmax, vmax=vmax
        )
        ax[1][col].axis("off")
        ax[2][col].imshow(np.rot90(x[:, peak[1] + i]), cmap="Greys_r")
        ax[2][col].axis("off")
        ax[3][col].imshow(
            np.rot90(expl[:, peak[1] + i]), cmap="seismic", vmin=-vmax, vmax=vmax
        )
        ax[3][col].axis("off")
        ax[4][col].imshow(x[:, :, peak[2] + i], cmap="Greys_r")
        ax[4][col].axis("off")
        ax[5][col].imshow(
            expl[:, :, peak[2] + i], cmap="seismic", vmin=-vmax, vmax=vmax
        )
        ax[5][col].axis("off")

    ax[0][0].set_ylabel("sagittal\nMRT", fontsize=9)
    ax[1][0].set_ylabel("sagittal\nLRP", fontsize=9)
    ax[2][0].set_ylabel("koronal\nMRT", fontsize=9)
    ax[3][0].set_ylabel("koronal\nLRP", fontsize=9)
    ax[4][0].set_ylabel("axial\nMRT", fontsize=9)
    ax[5][0].set_ylabel("axial\nLRP", fontsize=9)
    fig.tight_layout(rect=[0, 0, 0.90, 0.98])
    cbar = fig.colorbar(last_hm, ax=ax.ravel().tolist(), fraction=0.03, pad=0.02, shrink=0.55)
    cbar.set_label(
        "LRP-Relevanz (unnormiert)\nRot: treibt die Vorhersage nach oben\nBlau: zieht sie nach unten",
        fontsize=10,
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print("gespeichert:", save_path)
    if show_inline:
        display(fig)
    plt.close(fig)


saved_niftis: list[Path] = []
preds_by_dataset: dict[str, list[dict[str, object]]] = {}
mask_errors: list[str] = []

# Feste Overlay-Schnitte (sagittal / koronal / axial)
OVERLAY_SAGITTAL_X = 70
OVERLAY_CORONAL_Y = 104
OVERLAY_AXIAL_Z = 78

for dataset_id in DATASETS:
    df = dataset_labels[dataset_id]
    heatmaps_dir = RUN_DIR / "heatmaps" / dataset_id
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = (
        keras_xai_root / "output" / "notebooks"
        / "create_heatmaps_for_right_thalamus_model" / dataset_id
    )
    plot_dir.mkdir(parents=True, exist_ok=True)
    mask_pipeline = AsegThalamusMaskPipeline(
        dataset_id=dataset_id,
        dataset_dir=DATASET_DIRS[dataset_id],
        heatmaps_dir=heatmaps_dir,
    )
    preds_by_dataset[dataset_id] = []
    print(f"\n=== {dataset_id}: n={len(df)}, overlays={min(N_PLOT_SUBJECTS, len(df))} ===")

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=dataset_id)):
        sid = str(row["participant_id"])
        path = str(row["filepath"])
        y_true = float(row[pred_var])
        if not Path(path).is_file():
            mask_errors.append(f"[{dataset_id}/{sid}] cropped fehlt: {path}")
            continue

        vol = load_volume(path)
        y_pred = float(np.squeeze(model.predict(np.expand_dims(vol, 0), verbose=0)))
        # Unmaskierte LRP-Ausgabe (inkl. Hintergrundvoxel) — kein mask_explanation.
        R = lrp(np.expand_dims(vol, 0))[0].numpy()
        preds_by_dataset[dataset_id].append(
            {
                "subject_id": sid,
                pred_var: y_true,
                "prediction": y_pred,
                "sum_R": float(np.sum(R)),
            }
        )
        nii_path = heatmaps_dir / sid / f"lrp_heatmap_{dataset_id}_{sid}.nii.gz"
        nii_path.parent.mkdir(parents=True, exist_ok=True)
        save_heatmap_nifti(R, path, nii_path)
        saved_niftis.append(nii_path)

        try:
            mask_pipeline.run(sid)
        except (FileNotFoundError, RuntimeError) as exc:
            mask_errors.append(f"[{dataset_id}/{sid}] Maske: {exc}")

        left_mask_path = heatmaps_dir / sid / "aseg_mni152_left_thalamus_cropped.nii.gz"
        right_mask_path = heatmaps_dir / sid / "aseg_mni152_right_thalamus_cropped.nii.gz"
        left_data = nib.load(str(left_mask_path)).get_fdata() if left_mask_path.is_file() else None
        right_data = nib.load(str(right_mask_path)).get_fdata() if right_mask_path.is_file() else None

        if i < N_PLOT_SUBJECTS:
            plot_lrp_overlay(
                R,
                left_data,
                right_data,
                title=f"{dataset_id}  {sid}  true={y_true:.0f}  pred={y_pred:.0f}",
                save_path=plot_dir / f"lrp_overlay_{sid}.png",
                show_inline=SHOW_PLOTS_INLINE,
                sagittal_x=OVERLAY_SAGITTAL_X,
                coronal_y=OVERLAY_CORONAL_Y,
                axial_z=OVERLAY_AXIAL_Z,
            )

        # 6×8-Slice-Zoom (Raster aus dem Screenshot) auskommentiert.
        # Die drei Schnitte sagittal=70, koronal=104, axial=78 kommen aus plot_lrp_overlay.
        # if i == 0:
        #     plot_lrp_slices(
        #         vol,
        #         R,
        #         title=f"{dataset_id}  {sid}  Slice-Zoom um LRP-Peak",
        #         save_path=plot_dir / f"lrp_slices_{sid}.png",
        #         show_inline=SHOW_PLOTS_INLINE,
        #     )

print(f"\nHeatmaps: {len(saved_niftis)}  |  Masken-Fehler: {len(mask_errors)}")
for m in mask_errors:
    print(" -", m)

# %% [markdown]
# ## A.8. True vs. Predicted
#
# Scatter und Pearson-r / MAE pro Dataset (`n = N_SUBJECTS`).

# %%
for dataset_id, rows in preds_by_dataset.items():
    if not rows:
        print(f"[{dataset_id}] keine Vorhersagen.")
        continue
    preds_df = pd.DataFrame(rows)
    y_true = preds_df[pred_var].astype(float).to_numpy()
    y_pred = preds_df["prediction"].astype(float).to_numpy()
    r_val, _ = pearsonr(y_true, y_pred) if len(preds_df) >= 2 else (np.nan, None)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    print(f"[{dataset_id}] n={len(preds_df)}  r={r_val:.3f}  MAE={mae:.1f}")
    display(preds_df[["subject_id", pred_var, "prediction"]].round(1))

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y_true, y_pred, alpha=0.8)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"true {pred_var}")
    ax.set_ylabel("prediction")
    ax.set_title(f"{dataset_id}  r={r_val:.3f}  MAE={mae:.1f}")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    scatter_path = RUN_DIR / f"scatter_true_vs_pred_{dataset_id}_n{len(preds_df)}.png"
    fig.savefig(scatter_path, dpi=110)
    if SHOW_PLOTS_INLINE:
        display(fig)
    plt.close(fig)


# %% [markdown]
# ## A.9. Intensitäts-QC — erstes Subject (ungejittert)
#
# Für das **erste** Holdout-Subject des ersten Datensatzes in `DATASETS` (typisch IXI):
# Intensitäten des ungejitterten `cropped.nii.gz` — min/max, Histogramm (40 Bins,
# nur `x≠0`) und interaktiver 3D-Intensitätsplot (Plotly, feste HTML-Höhe).

# %%
import plotly.graph_objects as go
from IPython.display import HTML


def _fix_plotly_umd(html: str) -> str:
    """Plotly≥6 Bundles setzen fälschlich root.moduleName statt root.Plotly."""
    return html.replace(
        "root.moduleName = factory();",
        "root.Plotly = factory();",
        1,
    )


def plot_volume_intensity_qc(
    volume: np.ndarray,
    *,
    title: str,
    n_bins: int = 40,
    max_points: int = 40_000,
    step: int = 2,
    show_inline: bool = True,
) -> None:
    """min/max printen, Histogramm (≤n_bins) und 3D-Intensitätsplot."""
    vol = np.asarray(volume, dtype=np.float32).squeeze()
    flat = vol.ravel()
    brain = flat[flat != 0]
    nx, ny, nz = vol.shape

    print(f"{title}")
    print(f"Shape:   {vol.shape}  (x, y, z)  →  {flat.size:,} Voxel")
    print(f"min/max (gesamtes Volume): {flat.min():.6g} / {flat.max():.6g}")
    print(f"mean/median (gesamtes Volume): {flat.mean():.6g} / {np.median(flat):.6g}")
    print(f"Anteil x==0 (Hintergrund): {100.0 * np.mean(flat == 0):.1f}%")
    if brain.size:
        print(f"min/max (nur x≠0): {brain.min():.6g} / {brain.max():.6g}")
        print(f"mean/median (nur x≠0): {brain.mean():.6g} / {np.median(brain):.6g}")

    fig_h, ax = plt.subplots(figsize=(8, 4))
    if brain.size:
        ax.hist(brain, bins=int(n_bins), color="steelblue", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Intensität")
    ax.set_ylabel("Anzahl Voxel")
    ax.set_title(
        f"{title}\nHistogramm ({n_bins} Bins, nur x≠0)  |  "
        f"Volume min/max = {flat.min():.4g} / {flat.max():.4g}"
    )
    if brain.size:
        ax.axvline(brain.mean(), color="C1", ls="--", lw=1.2, label=f"mean={brain.mean():.3g}")
        ax.axvline(
            np.median(brain), color="C3", ls=":", lw=1.2, label=f"median={np.median(brain):.3g}"
        )
        ax.legend(frameon=False)
    fig_h.tight_layout()
    if show_inline:
        display(fig_h)
    plt.close(fig_h)

    idx = np.argwhere(vol != 0)
    if idx.size:
        idx = idx[
            (idx[:, 0] % step == 0) & (idx[:, 1] % step == 0) & (idx[:, 2] % step == 0)
        ]
        if len(idx) > max_points:
            rng = np.random.default_rng(0)
            idx = idx[rng.choice(len(idx), size=max_points, replace=False)]
        vals = vol[idx[:, 0], idx[:, 1], idx[:, 2]]
        vmax = float(np.percentile(vals, 99.5)) if vals.size else 1.0
    else:
        vals = np.array([], dtype=np.float32)
        vmax = 1.0

    fig3d = go.Figure(
        data=[
            go.Scatter3d(
                x=idx[:, 0] if idx.size else [],
                y=idx[:, 1] if idx.size else [],
                z=idx[:, 2] if idx.size else [],
                mode="markers",
                marker=dict(
                    size=1.8,
                    color=vals,
                    colorscale="Gray",
                    cmin=0.0,
                    cmax=vmax,
                    opacity=0.55,
                    colorbar=dict(title="Intensität", thickness=18, len=0.7),
                ),
                hovertemplate=(
                    "x=%{x:.0f} y=%{y:.0f} z=%{z:.0f}<br>"
                    "I=%{marker.color:.4g}<extra></extra>"
                ),
            )
        ]
    )
    fig3d.update_layout(
        title=f"{title} · 3D-Intensitäten (n={len(idx)} Punkte)",
        scene=dict(
            xaxis_title="x (sagittal)",
            yaxis_title="y (koronal)",
            zaxis_title="z (axial)",
            aspectmode="data",
            xaxis=dict(range=[0, nx - 1]),
            yaxis=dict(range=[0, ny - 1]),
            zaxis=dict(range=[0, nz - 1]),
        ),
        width=980,
        height=780,
        margin=dict(l=0, r=60, t=60, b=10),
    )
    if show_inline:
        html = _fix_plotly_umd(
            fig3d.to_html(
                include_plotlyjs=True,
                full_html=False,
                config={"responsive": True, "displayModeBar": True},
            )
        )
        display(
            HTML(
                '<div style="width:100%; max-width:1100px; height:820px; '
                'border:1px solid #ddd; margin:0.5rem 0; overflow:hidden;">'
                f"{html}"
                "</div>"
            )
        )


# Erstes Subject des ersten Datensatzes in DATASETS (Holdout-/Analyse-Labels).
_ds0 = DATASETS[0]
_row0 = dataset_labels[_ds0].iloc[0]
_sid0 = str(_row0["participant_id"])
_path0 = Path(str(_row0["filepath"]))
if not _path0.is_file():
    raise FileNotFoundError(f"[{_ds0}/{_sid0}] Volume fehlt: {_path0}")

_vol0 = np.asarray(nib.load(str(_path0)).get_fdata(), dtype=np.float32).squeeze()
plot_volume_intensity_qc(
    _vol0,
    title=f"A.9  {_ds0}  {_sid0}  ungejittert  ({_path0.name})",
    n_bins=40,
    show_inline=SHOW_PLOTS_INLINE,
)


# %% [markdown]
# ## A.10. Relevanz im Thalamus
#
# Anteil der |LRP|-Summe in linker / rechter Thalamus-Maske vs. außerhalb.
# Kompakte Voxel-Statistik statt einer Tabelle aller ~5,7 Mio. Voxel.
#
# Die Heatmaps auf der Platte sind **unmaskiert** (kein `mask_explanation`):
# ΣR enthält auch Relevanz auf Hintergrundvoxeln (`x==0`), die vor allem durch
# die `flat`-Regel der eingangsnahen Convs entsteht. `sum_R` sollte daher nahe
# an `pred` liegen (Teil C: Relevanzerhaltung ~100 %).
#
# Die Spalten `pct_|R|_left` / `pct_|R|_right` / `pct_|R|_outside` sind Anteile
# an der **gesamten** |R|-Summe (inkl. Hintergrund).
# `sum_R_left` / `sum_R_right`: vorzeichenbehaftete Relevanz-Summe der Voxel
# in der linken bzw. rechten Thalamus-Maske.
#
# Danach (A.10b): Histogramm der Relevanzwerte für das erste Subject je Dataset —
# einmal über alle Voxel, einmal nur im rechten Thalamus (100 Bins, \([-0.4, +0.4]\)).

# %%
def _load_nii(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32).squeeze()


def _pred_for_subject(dataset_id: str, sid: str) -> float | None:
    for rec in preds_by_dataset.get(dataset_id, []):
        if str(rec["subject_id"]) == sid:
            return float(rec["prediction"])
    return None


rows: list[dict[str, object]] = []

for dataset_id in DATASETS:
    for _, row in dataset_labels[dataset_id].iterrows():
        sid = str(row["participant_id"])
        y_true = float(row[pred_var])
        y_pred = _pred_for_subject(dataset_id, sid)
        subject_dir = RUN_DIR / "heatmaps" / dataset_id / sid
        heatmap_path = subject_dir / f"lrp_heatmap_{dataset_id}_{sid}.nii.gz"
        left_p = subject_dir / "aseg_mni152_left_thalamus_cropped.nii.gz"
        right_p = subject_dir / "aseg_mni152_right_thalamus_cropped.nii.gz"
        if not heatmap_path.is_file() or not left_p.is_file() or not right_p.is_file():
            print(f"[{dataset_id}/{sid}] Heatmap oder Maske fehlt — übersprungen.")
            continue

        heat = _load_nii(heatmap_path)
        left_mask = _load_nii(left_p) > 0
        right_mask = _load_nii(right_p) > 0
        sum_total = float(np.sum(heat))
        sum_abs = float(np.sum(np.abs(heat)))
        sum_left = float(np.sum(heat[left_mask]))
        sum_right = float(np.sum(heat[right_mask]))
        sum_abs_left = float(np.sum(np.abs(heat[left_mask])))
        sum_abs_right = float(np.sum(np.abs(heat[right_mask])))
        sum_abs_out = sum_abs - sum_abs_left - sum_abs_right
        nz = heat[heat != 0]

        rows.append(
            {
                "dataset": dataset_id,
                "subject_id": sid,
                "true": y_true,
                "pred": y_pred,
                "sum_R": sum_total,
                "sum_|R|": sum_abs,
                "sum_R_left": sum_left,
                "sum_R_right": sum_right,
                "pct_|R|_left": 100.0 * sum_abs_left / sum_abs if sum_abs else np.nan,
                "pct_|R|_right": 100.0 * sum_abs_right / sum_abs if sum_abs else np.nan,
                "pct_|R|_outside": 100.0 * sum_abs_out / sum_abs if sum_abs else np.nan,
                "n_nonzero": int(nz.size),
                "min_nz": float(nz.min()) if nz.size else np.nan,
                "max_R": float(heat.max()),
            }
        )

summary = pd.DataFrame(rows)
display(summary.round(4))
out_tsv = RUN_DIR / "lrp_relevance_left_right_thalamus_by_subject.tsv"
summary.to_csv(out_tsv, sep="\t", index=False, float_format="%.6e")
print("gespeichert:", out_tsv)

# %% [markdown]
# ### A.10b. Histogramm der LRP-Relevanzen (erstes Subject je Dataset)
#
# Für das **erste** Holdout-Subject von IXI und UKB:
#
# 1. Histogramm **aller** Voxel-Relevanzen der unmaskierten Heatmap
# 2. Histogramm nur der Relevanzen **im rechten Thalamus** (FreeSurfer-Maske)
#
# Jeweils **100 Bins** im Intervall \([-0.4,\,+0.4]\). Werte außerhalb des
# Intervalls werden nicht in die Balken gezählt, aber als Anzahl ausgewiesen.

# %%
HIST_R_LO, HIST_R_HI = -0.4, 0.4
HIST_N_BINS = 100
HIST_BINS = np.linspace(HIST_R_LO, HIST_R_HI, HIST_N_BINS + 1)

hist_plot_dir = (
    keras_xai_root
    / "output"
    / "notebooks"
    / "analysis_LRP_for_right_thalamus_volume_based_on_CNN_prediction"
    / "relevance_histograms"
)
hist_plot_dir.mkdir(parents=True, exist_ok=True)


def plot_relevance_histograms(
    heat: np.ndarray,
    right_mask: np.ndarray,
    *,
    dataset_id: str,
    subject_id: str,
    bins: np.ndarray = HIST_BINS,
    show_inline: bool = True,
    save_dir: Path | None = hist_plot_dir,
) -> None:
    """Zwei Histogramme: alle Voxel vs. nur rechter Thalamus."""
    heat = np.asarray(heat, dtype=np.float32).squeeze()
    right = np.asarray(right_mask, dtype=bool).squeeze()
    r_all = heat.ravel()
    r_right = heat[right]

    def _outside_count(vals: np.ndarray) -> int:
        return int(np.sum((vals < bins[0]) | (vals > bins[-1])))

    panels = [
        (
            r_all,
            f"{dataset_id}  {subject_id}  ·  alle Voxel",
            "alle_voxel",
        ),
        (
            r_right,
            f"{dataset_id}  {subject_id}  ·  rechter Thalamus",
            "rechter_thalamus",
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    for ax, (vals, title, tag) in zip(axes, panels):
        n_out = _outside_count(vals)
        ax.hist(
            vals,
            bins=bins,
            color="steelblue",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.axvline(0.0, color="k", ls=":", lw=0.9)
        ax.set_xlabel("LRP-Relevanz R")
        ax.set_ylabel("Anzahl Voxel")
        ax.set_title(
            f"{title}\n"
            f"n={vals.size:,}  |  min/max={vals.min():.4g}/{vals.max():.4g}  |  "
            f"außerhalb [{bins[0]:.1f},{bins[-1]:.1f}]: {n_out:,}",
            fontsize=9,
        )
        print(
            f"[{dataset_id}/{subject_id}/{tag}] n={vals.size}  "
            f"min={vals.min():.6g}  max={vals.max():.6g}  "
            f"mean={vals.mean():.6g}  außerhalb_Bins={n_out}"
        )

    fig.suptitle(
        f"LRP-Relevanz-Histogramm ({HIST_N_BINS} Bins, [{HIST_R_LO}, {HIST_R_HI}])",
        fontsize=11,
    )
    fig.tight_layout()
    if save_dir is not None:
        out = save_dir / f"lrp_hist_{dataset_id}_{subject_id}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print("gespeichert:", out)
    if show_inline:
        display(fig)
    plt.close(fig)


for dataset_id in DATASETS:
    df0 = dataset_labels[dataset_id]
    if df0.empty:
        print(f"[{dataset_id}] keine Subjects.")
        continue
    row0 = df0.iloc[0]
    sid0 = str(row0["participant_id"])
    subject_dir = RUN_DIR / "heatmaps" / dataset_id / sid0
    heatmap_path = subject_dir / f"lrp_heatmap_{dataset_id}_{sid0}.nii.gz"
    right_p = subject_dir / "aseg_mni152_right_thalamus_cropped.nii.gz"
    if not heatmap_path.is_file() or not right_p.is_file():
        print(f"[{dataset_id}/{sid0}] Heatmap oder rechte Maske fehlt — übersprungen.")
        continue
    plot_relevance_histograms(
        _load_nii(heatmap_path),
        _load_nii(right_p) > 0,
        dataset_id=dataset_id,
        subject_id=sid0,
        show_inline=SHOW_PLOTS_INLINE,
    )

# %% [markdown]
# ## A.11. Interaktiver 3D-Plot (erstes Subject je Dataset)
#
# Für das **erste** IXI- und UKB-Subject: Gehirnkontur plus beide FreeSurfer-Thalamus-Masken
# (`aseg` 10 links / 49 rechts) und die **unnormierte** LRP-Heatmap.
#
# Maus: drehen, Scrollrad: zoomen, Shift+Ziehen: verschieben. Farben wie in den 2D-Overlays
# (lila / grün). LRP-Punkte sind die Voxel mit dem größten \(|R|\) (sonst ~2 Mio. Punkte).

# %%
import plotly.graph_objects as go
from IPython.display import HTML, display
from scipy.ndimage import binary_erosion

# Für JupyterHub *und* nbconvert-HTML:
# fig.show()/notebook-Renderer → oft height:100% ohne Elternhöhe → leerer Export.
# Stattdessen fig.to_html(...) als text/html ausgeben (feste Pixelhöhe).
_PLOTLY_JS_DONE = False

# Farben analog zu plot_lrp_overlay (Abschnitt A.7)
COLOR_LEFT = "rgb(153, 51, 204)"
COLOR_RIGHT = "rgb(38, 166, 64)"
COLOR_BRAIN = "rgb(170, 170, 170)"
MAX_BRAIN_POINTS = 8_000
MAX_LRP_POINTS = 10_000


def _load_vol(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32).squeeze()


def _surface_xyz(mask: np.ndarray, *, step: int = 1, max_points: int | None = None):
    """Randvoxel einer binären Maske (optional unterabgetastet)."""
    m = np.asarray(mask, dtype=bool)
    if step > 1:
        m = m[::step, ::step, ::step]
    if not np.any(m):
        return np.array([]), np.array([]), np.array([])
    eroded = binary_erosion(m, iterations=1)
    surf = m & ~eroded if np.any(eroded) else m
    idx = np.argwhere(surf)
    if step > 1:
        idx = idx * int(step)
    if max_points is not None and len(idx) > max_points:
        rng = np.random.default_rng(0)
        idx = idx[rng.choice(len(idx), size=max_points, replace=False)]
    return idx[:, 0].astype(float), idx[:, 1].astype(float), idx[:, 2].astype(float)


def _mask_xyz(mask: np.ndarray, *, max_points: int | None = 12_000):
    idx = np.argwhere(np.asarray(mask, dtype=bool))
    if max_points is not None and len(idx) > max_points:
        rng = np.random.default_rng(1)
        idx = idx[rng.choice(len(idx), size=max_points, replace=False)]
    if len(idx) == 0:
        return np.array([]), np.array([]), np.array([])
    return idx[:, 0].astype(float), idx[:, 1].astype(float), idx[:, 2].astype(float)


def _top_lrp_xyz(heat: np.ndarray, max_points: int):
    """Voxel mit dem größten |R| (unnormierte Werte bleiben in `vals`)."""
    flat = np.abs(heat).ravel()
    n_nz = int(np.count_nonzero(flat))
    k = int(min(max_points, n_nz))
    if k < 1:
        return np.array([]), np.array([]), np.array([]), np.array([])
    take = np.argpartition(flat, -k)[-k:]
    take = take[np.argsort(flat[take])[::-1]]
    zyx = np.unravel_index(take, heat.shape)
    vals = heat[zyx]
    return (
        zyx[0].astype(float),
        zyx[1].astype(float),
        zyx[2].astype(float),
        np.asarray(vals, dtype=np.float32),
    )


def plot_thalamus_lrp_3d(
    *,
    dataset_id: str,
    subject_id: str,
    volume: np.ndarray,
    heatmap: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    y_true: float | None,
    y_pred: float | None,
    pred_var_name: str,
    save_html: Path | None = None,
    title_suffix: str | None = None,
) -> go.Figure:
    vol = np.asarray(volume, dtype=np.float32).squeeze()
    heat = np.asarray(heatmap, dtype=np.float32).squeeze()
    left = np.asarray(left_mask, dtype=np.float32).squeeze() > 0
    right = np.asarray(right_mask, dtype=np.float32).squeeze() > 0

    bx, by, bz = _surface_xyz(vol != 0, step=3, max_points=MAX_BRAIN_POINTS)
    lx, ly, lz = _mask_xyz(left)
    rx, ry, rz = _mask_xyz(right)
    hx, hy, hz, hv = _top_lrp_xyz(heat, MAX_LRP_POINTS)

    vmax = float(np.nanmax(np.abs(hv))) if hv.size else float(np.nanmax(np.abs(heat)) or 1.0)
    if vmax == 0.0:
        vmax = 1.0

    true_s = f"{y_true:.1f}" if y_true is not None else "n/a"
    pred_s = f"{y_pred:.1f}" if y_pred is not None else "n/a"
    head = f"<b>{dataset_id.upper()} · {subject_id}</b>"
    if title_suffix:
        head = f"{head} · {title_suffix}"
    title = (
        f"{head}<br>"
        f"3D-Gehirn mit FreeSurfer-Thalamus-Masken und unnormierter LRP-Relevanz<br>"
        f"true {pred_var_name} = {true_s} · Vorhersage = {pred_s} · "
        f"Top-{hv.size} LRP-Voxel nach |R|"
    )

    fig = go.Figure()

    if bx.size:
        fig.add_trace(
            go.Scatter3d(
                x=bx, y=by, z=bz,
                mode="markers",
                name="Gehirnkontur (cropped T1)",
                marker=dict(size=1.4, color=COLOR_BRAIN, opacity=0.12),
                hoverinfo="skip",
                legendgroup="brain",
            )
        )

    if lx.size >= 4:
        fig.add_trace(
            go.Mesh3d(
                x=lx, y=ly, z=lz,
                alphahull=0,
                name="linker Thalamus (aseg 10)",
                color=COLOR_LEFT,
                opacity=0.28,
                flatshading=True,
                hovertemplate="linker Thalamus<br>x=%{x:.0f} y=%{y:.0f} z=%{z:.0f}<extra></extra>",
                legendgroup="left",
                showlegend=True,
            )
        )
    elif lx.size:
        fig.add_trace(
            go.Scatter3d(
                x=lx, y=ly, z=lz, mode="markers",
                name="linker Thalamus (aseg 10)",
                marker=dict(size=3, color=COLOR_LEFT, opacity=0.45),
                legendgroup="left",
            )
        )

    if rx.size >= 4:
        fig.add_trace(
            go.Mesh3d(
                x=rx, y=ry, z=rz,
                alphahull=0,
                name="rechter Thalamus (aseg 49)",
                color=COLOR_RIGHT,
                opacity=0.28,
                flatshading=True,
                hovertemplate="rechter Thalamus<br>x=%{x:.0f} y=%{y:.0f} z=%{z:.0f}<extra></extra>",
                legendgroup="right",
                showlegend=True,
            )
        )
    elif rx.size:
        fig.add_trace(
            go.Scatter3d(
                x=rx, y=ry, z=rz, mode="markers",
                name="rechter Thalamus (aseg 49)",
                marker=dict(size=3, color=COLOR_RIGHT, opacity=0.45),
                legendgroup="right",
            )
        )

    if hv.size:
        fig.add_trace(
            go.Scatter3d(
                x=hx, y=hy, z=hz,
                mode="markers",
                name="LRP-Relevanz (unnormiert)",
                marker=dict(
                    size=2.2,
                    color=hv,
                    colorscale="RdBu_r",
                    cmin=-vmax,
                    cmax=vmax,
                    opacity=0.75,
                    colorbar=dict(
                        title=dict(
                            text="LRP-Relevanz<br>(unnormiert)",
                            side="right",
                        ),
                        thickness=18,
                        len=0.65,
                        x=1.02,
                    ),
                ),
                hovertemplate=(
                    "LRP (unnormiert): %{marker.color:.6f}<br>"
                    "Voxel x=%{x:.0f} y=%{y:.0f} z=%{z:.0f}<extra></extra>"
                ),
                legendgroup="lrp",
            )
        )

    nx, ny, nz = heat.shape
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        legend=dict(
            title=dict(text="Legende"),
            itemsizing="constant",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        scene=dict(
            xaxis_title="x (Voxel, sagittal)",
            yaxis_title="y (Voxel, koronal)",
            zaxis_title="z (Voxel, axial)",
            aspectmode="data",
            xaxis=dict(range=[0, nx - 1]),
            yaxis=dict(range=[0, ny - 1]),
            zaxis=dict(range=[0, nz - 1]),
            camera=dict(eye=dict(x=1.55, y=1.35, z=0.85)),
        ),
        width=980,
        height=780,
        margin=dict(l=0, r=80, t=90, b=10),
        hovermode="closest",
    )

    def _fix_plotly_umd(html: str) -> str:
        """Plotly≥6 Bundles setzen fälschlich root.moduleName statt root.Plotly.

        Ohne diesen Patch bleibt Plotly.newPlot undefined → leerer Rahmen im HTML-Export.
        """
        return html.replace(
            "root.moduleName = factory();",
            "root.Plotly = factory();",
            1,
        )

    if save_html is not None:
        save_html.parent.mkdir(parents=True, exist_ok=True)
        standalone = _fix_plotly_umd(
            fig.to_html(include_plotlyjs=True, full_html=True)
        )
        save_html.write_text(standalone, encoding="utf-8")
        print("3D-HTML:", save_html)

    global _PLOTLY_JS_DONE
    # Erste Figur: Plotly.js einbetten; weitere: nur Daten.
    include_js = True if not _PLOTLY_JS_DONE else False
    html = _fix_plotly_umd(
        fig.to_html(
            include_plotlyjs=include_js,
            full_html=False,
            config={"responsive": True, "displayModeBar": True},
        )
    )
    _PLOTLY_JS_DONE = True
    # Feste Höhe — sonst kollabiert der nbconvert-Export auf 0 Pixel.
    display(
        HTML(
            '<div style="width:100%; max-width:1100px; height:820px; '
            'border:1px solid #ddd; margin:0.5rem 0; overflow:hidden;">'
            f"{html}"
            "</div>"
        )
    )
    return fig


plot_dir_3d = (
    keras_xai_root
    / "output"
    / "notebooks"
    / "analysis_LRP_for_right_thalamus_volume_based_on_CNN_prediction"
)
plot_dir_3d.mkdir(parents=True, exist_ok=True)

for dataset_id in DATASETS:
    df = dataset_labels[dataset_id]
    if df.empty:
        print(f"[{dataset_id}] keine Subjects.")
        continue

    row = df.iloc[0]
    sid = str(row["participant_id"])
    y_true = float(row[pred_var])
    y_pred = None
    for rec in preds_by_dataset.get(dataset_id, []):
        if str(rec["subject_id"]) == sid:
            y_pred = float(rec["prediction"])
            break
    t1_path = Path(str(row["filepath"]))
    subject_dir = RUN_DIR / "heatmaps" / dataset_id / sid
    heatmap_path = subject_dir / f"lrp_heatmap_{dataset_id}_{sid}.nii.gz"
    left_path = subject_dir / "aseg_mni152_left_thalamus_cropped.nii.gz"
    right_path = subject_dir / "aseg_mni152_right_thalamus_cropped.nii.gz"

    missing = [p for p in (t1_path, heatmap_path, left_path, right_path) if not p.is_file()]
    if missing:
        print(f"[{dataset_id}/{sid}] Dateien fehlen:")
        for p in missing:
            print("  -", p)
        continue

    print(f"\n[{dataset_id}] 3D-Plot für erstes Subject: {sid}")
    plot_thalamus_lrp_3d(
        dataset_id=dataset_id,
        subject_id=sid,
        volume=_load_vol(t1_path),
        heatmap=_load_vol(heatmap_path),
        left_mask=_load_vol(left_path),
        right_mask=_load_vol(right_path),
        y_true=y_true,
        y_pred=y_pred,
        pred_var_name=pred_var,
        save_html=plot_dir_3d / f"{dataset_id}_{sid}_thalamus_lrp_3d.html",
    )

# %% [markdown]
# ## B.1. True vs. Predicted (jittered model)
#
# Scatter und Pearson-r / MAE für die **`N_SUBJECTS` UKB-Holdout-Subjects** — diesmal mit dem
# auf **gejitterten** Volumes trainierten Modell
#
# ```
# /mnt/users/andreasre/data/nn-trainings/mri/Right-Whole_thalamus/
#     training_run_05h09m52s_04sep2026
# ```
#
# (Training: `jittered_data/.../T1_mni152_right_thalamus_preserved_others_shuffled.nii.gz`;
# Holdout-Predict wie in Teil A auf den originalen `cropped.nii.gz`).
#
# Vergleichspunkt zu **A.8** (Original-Modell auf denselben Holdout-Fällen).
# Danach folgt **B.2** (Intensitäts-QC des ersten gejitterten Holdout-Subjects).

# %%
JITTER_MODEL_RUN_DIR = Path(
    "~/data/nn-trainings/mri/Right-Whole_thalamus/"
    "training_run_05h09m52s_04sep2026"
).expanduser().resolve()
JITTER_MODEL_PATH = JITTER_MODEL_RUN_DIR / "model.keras"
JITTER_CONFIG_PATH = JITTER_MODEL_RUN_DIR / "config.yaml"

for p, name in (
    (JITTER_MODEL_PATH, "Jitter-Modell"),
    (JITTER_CONFIG_PATH, "Jitter-Config"),
):
    if not p.is_file():
        raise FileNotFoundError(f"{name} fehlt: {p}")

jitter_cfg = OmegaConf.load(JITTER_CONFIG_PATH)
with open_dict(jitter_cfg):
    jitter_cfg.paths.csv_dir = str(JITTER_MODEL_RUN_DIR)
    if "prediction" not in jitter_cfg.training:
        jitter_cfg.training.prediction = {}
    jitter_cfg.training.prediction.batch_size = int(PRED_BATCH_SIZE)

jitter_model = _build_single_device_model(jitter_cfg)
w0 = jitter_model.get_weights()[0].copy()
jitter_model.load_weights(str(JITTER_MODEL_PATH))
jitter_delta = float(np.mean(np.abs(jitter_model.get_weights()[0] - w0)))
if jitter_delta < 1e-9:
    raise RuntimeError("Jitter-Modell-Gewichte wurden nicht geladen.")
print(f"JITTER_MODEL_RUN_DIR: {JITTER_MODEL_RUN_DIR}")
print(f"Gewichtsdelta (jittered model): {jitter_delta:.3g}")

# Dieselben UKB-Holdout-Subjects wie in Teil A (dataset_labels["ukb"]).
df_ukb = dataset_labels["ukb"]
jitter_model_rows: list[dict[str, object]] = []
for _, row in tqdm(df_ukb.iterrows(), total=len(df_ukb), desc="ukb-jitter-model"):
    sid = str(row["participant_id"])
    path = str(row["filepath"])
    y_true = float(row[pred_var])
    if not Path(path).is_file():
        print(f"[ukb/{sid}] Volume fehlt: {path}")
        continue
    vol = load_volume(path)
    y_pred = float(np.squeeze(jitter_model.predict(np.expand_dims(vol, 0), verbose=0)))
    jitter_model_rows.append(
        {"subject_id": sid, pred_var: y_true, "prediction": y_pred}
    )

if not jitter_model_rows:
    raise RuntimeError("Keine Vorhersagen mit dem Jitter-Modell.")

jitter_preds_df = pd.DataFrame(jitter_model_rows)
y_true = jitter_preds_df[pred_var].astype(float).to_numpy()
y_pred = jitter_preds_df["prediction"].astype(float).to_numpy()
r_val, _ = pearsonr(y_true, y_pred) if len(jitter_preds_df) >= 2 else (np.nan, None)
mae = float(np.mean(np.abs(y_true - y_pred)))
print(f"[ukb | jittered model] n={len(jitter_preds_df)}  r={r_val:.3f}  MAE={mae:.1f}")
display(jitter_preds_df[["subject_id", pred_var, "prediction"]].round(1))

fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.scatter(y_true, y_pred, alpha=0.8)
lo = float(min(y_true.min(), y_pred.min()))
hi = float(max(y_true.max(), y_pred.max()))
ax.plot([lo, hi], [lo, hi], "k--", lw=1)
ax.set_xlabel(f"true {pred_var}")
ax.set_ylabel("prediction (jittered model)")
ax.set_title(f"ukb  jittered model  r={r_val:.3f}  MAE={mae:.1f}")
ax.set_aspect("equal", adjustable="box")
fig.tight_layout()
scatter_path = (
    JITTER_MODEL_RUN_DIR
    / f"scatter_true_vs_pred_ukb_holdout_n{len(jitter_preds_df)}.png"
)
fig.savefig(scatter_path, dpi=110)
print("gespeichert:", scatter_path)
if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)

# %% [markdown]
# ## B.2. Intensitäts-QC — erstes UKB-Holdout-Subject (gejittert)
#
# Analog zu **A.9**, aber für das **erste** UKB-Holdout-Subject und das gejitterte
# Volume
# `T1_mni152_right_thalamus_preserved_others_shuffled.nii.gz`
# (rechter Thalamus erhalten, Rest permutiert).
# min/max, Histogramm (40 Bins, nur `x≠0`) und 3D-Intensitätsplot.
# Nutzt `plot_volume_intensity_qc` aus A.9.

# %%
_JITTER_ROOT_B2 = Path("/mnt/users/andreasre/data/jittered_data")
_JITTER_FILE_B2 = "T1_mni152_right_thalamus_preserved_others_shuffled.nii.gz"

_row_ukb0 = dataset_labels["ukb"].iloc[0]
_sid_ukb0 = str(_row_ukb0["participant_id"])
_jitter_path0 = (
    _JITTER_ROOT_B2 / "ukb" / "recon" / _sid_ukb0 / "mri" / _JITTER_FILE_B2
)
if not _jitter_path0.is_file():
    raise FileNotFoundError(
        f"[ukb/{_sid_ukb0}] gejittertes Volume fehlt: {_jitter_path0}"
    )

_vol_jit0 = np.asarray(
    nib.load(str(_jitter_path0)).get_fdata(), dtype=np.float32
).squeeze()
plot_volume_intensity_qc(
    _vol_jit0,
    title=f"B.2  ukb  {_sid_ukb0}  gejittert  ({_jitter_path0.name})",
    n_bins=40,
    show_inline=SHOW_PLOTS_INLINE,
)

# %% [markdown]
# ## B.3. QC der gejitterten UKB-Volumes (Nicht-Holdout)
#
# Sanity-Check für das **Simulations-/Jitter-Framework**: In den Dateien
#
# ```
# /mnt/users/andreasre/data/jittered_data/ukb/recon/<subject-id>/mri/
#     T1_mni152_right_thalamus_preserved_others_shuffled.nii.gz
# ```
#
# bleibt der **rechte Thalamus** unverändert, das restliche Gehirn wird voxelweise
# permutiert ("shuffled"). Wenn das Modell tatsächlich das rechte Thalamus-Volumen
# liest, darf diese Manipulation die Vorhersage kaum verändern (→ Abschnitt B.4).
#
# Nur für `ukb`, da nur dieser Datensatz gejittert vorliegt.
#
# Vorgehen:
#
# 1. `N_JITTER_QC_SUBJECTS` Subjects **zufällig** (`JITTER_QC_SEED`, reproduzierbar) aus dem
#    Jitter-Verzeichnis ziehen, dabei **alle** Subject-IDs des Holdout-Splits
#    (`UKB_HOLDOUT_PREDICT_TSV`, n = 10 000) ausschließen — die bleiben Abschnitt B.4 vorbehalten.
# 2. Pro Subject sagittal / koronal / axial plotten (gleiche Schnitte wie Abschnitt A.7:
#    `x=70`, `y=104`, `z=78`), mit **Colorbar an der Seite** (Grauwert-Intensität).
#    Die grüne Kontur markiert die erhaltene rechte Thalamus-Maske.
# 3. Numerischer QC: Pearson-r zwischen Original (`cropped.nii.gz`) und gejittertem Volume,
#    getrennt **innerhalb** der rechten Thalamus-Maske (erwartet ≈ 1.0) und **außerhalb**
#    im übrigen Gehirn (erwartet ≪ 1.0).

# %%
# --- Jitter-Konfiguration (nur ukb) --------------------------------------
JITTER_ROOT = Path("/mnt/users/andreasre/data/jittered_data")
JITTER_DATASET = "ukb"
JITTER_FILENAME = "T1_mni152_right_thalamus_preserved_others_shuffled.nii.gz"
JITTER_MASK_LEFT_NAME = "aseg_mni152_left_thalamus_cropped.nii.gz"
JITTER_MASK_RIGHT_NAME = "aseg_mni152_right_thalamus_cropped.nii.gz"

N_JITTER_QC_SUBJECTS = 3
JITTER_QC_SEED = 0

# Schnitte wie in Abschnitt A.7 (Fallback, falls Abschnitt A.7 nicht gelaufen ist).
JITTER_SAGITTAL_X = int(globals().get("OVERLAY_SAGITTAL_X", 70))
JITTER_CORONAL_Y = int(globals().get("OVERLAY_CORONAL_Y", 104))
JITTER_AXIAL_Z = int(globals().get("OVERLAY_AXIAL_Z", 78))

if JITTER_DATASET != "ukb":
    raise ValueError("Gejitterte Daten liegen nur für 'ukb' vor.")
JITTER_RECON_DIR = JITTER_ROOT / JITTER_DATASET / "recon"
if not JITTER_RECON_DIR.is_dir():
    raise FileNotFoundError(f"Jitter-Verzeichnis fehlt: {JITTER_RECON_DIR}")


def jitter_mri_dir(subject_id: str) -> Path:
    return JITTER_RECON_DIR / subject_id / "mri"


def jitter_volume_path(subject_id: str) -> Path:
    return jitter_mri_dir(subject_id) / JITTER_FILENAME


def original_volume_path(subject_id: str) -> Path:
    return DATASET_DIRS[JITTER_DATASET] / "recon" / subject_id / "mri" / "cropped.nii.gz"


def jitter_mask_paths(subject_id: str) -> tuple[Path, Path]:
    """Linke / rechte Thalamus-Maske: bevorzugt aus dem Jitter-Ordner,
    sonst aus den in Abschnitt A.7 erzeugten Masken."""
    jit = jitter_mri_dir(subject_id)
    fallback = RUN_DIR / "heatmaps" / JITTER_DATASET / subject_id
    left = jit / JITTER_MASK_LEFT_NAME
    right = jit / JITTER_MASK_RIGHT_NAME
    if not left.is_file():
        left = fallback / JITTER_MASK_LEFT_NAME
    if not right.is_file():
        right = fallback / JITTER_MASK_RIGHT_NAME
    return left, right


def load_ukb_holdout_ids() -> set[str]:
    """Alle Subject-IDs des offiziellen predict-Splits (Holdout)."""
    df = pd.read_csv(UKB_HOLDOUT_PREDICT_TSV, sep=None, engine="python")
    col = next(
        (c for c in ("subject-id", "participant_id", "Subject") if c in df.columns), None
    )
    if col is None:
        raise ValueError(f"Keine Subject-Spalte in {UKB_HOLDOUT_PREDICT_TSV}")
    return {str(s) for s in df[col].dropna()}


def pick_jitter_subjects(n: int, *, seed: int, exclude: set[str]) -> list[str]:
    """`n` zufällige Jitter-Subjects, die *nicht* im Holdout liegen."""
    candidates = sorted(
        p.name
        for p in JITTER_RECON_DIR.iterdir()
        if p.is_dir() and not p.name.startswith(".") and p.name not in exclude
    )
    if not candidates:
        raise FileNotFoundError(f"Keine Nicht-Holdout-Subjects in {JITTER_RECON_DIR}")

    rng = np.random.default_rng(seed)
    picked: list[str] = []
    # Erst permutieren, dann prüfen: nicht alle ~56 000 Ordner stat'en.
    for idx in rng.permutation(len(candidates)):
        sid = candidates[int(idx)]
        if jitter_volume_path(sid).is_file():
            picked.append(sid)
            if len(picked) >= int(n):
                break
    if len(picked) < int(n):
        raise FileNotFoundError(
            f"Nur {len(picked)} von {n} Subjects mit {JITTER_FILENAME} gefunden."
        )
    return picked


def _load_jitter_nii(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32).squeeze()


def jitter_qc_stats(subject_id: str) -> dict[str, object]:
    """Original vs. gejittert: Korrelation im Thalamus und im Rest des Gehirns."""
    jit = _load_jitter_nii(jitter_volume_path(subject_id))
    stats: dict[str, object] = {
        "subject_id": subject_id,
        "shape": tuple(int(s) for s in jit.shape),
        "min": float(np.nanmin(jit)),
        "max": float(np.nanmax(jit)),
        "r_thalamus": np.nan,
        "r_outside": np.nan,
    }

    orig_path = original_volume_path(subject_id)
    _, right_path = jitter_mask_paths(subject_id)
    if not orig_path.is_file() or not right_path.is_file():
        stats["note"] = "Original oder rechte Maske fehlt — Korrelation übersprungen."
        return stats

    orig = _load_jitter_nii(orig_path)
    right = _load_jitter_nii(right_path) > 0
    if orig.shape != jit.shape or right.shape != jit.shape:
        stats["note"] = f"Shape-Mismatch: {orig.shape} / {jit.shape} / {right.shape}"
        return stats

    brain = (orig != 0) | (jit != 0)
    inside = brain & right
    outside = brain & ~right
    if inside.sum() >= 2:
        stats["r_thalamus"] = float(pearsonr(orig[inside], jit[inside])[0])
    if outside.sum() >= 2:
        stats["r_outside"] = float(pearsonr(orig[outside], jit[outside])[0])
    stats["note"] = ""
    return stats


def plot_jitter_slices(
    volume: np.ndarray,
    right_mask: np.ndarray | None = None,
    *,
    title: str,
    save_path: Path | None = None,
    show_inline: bool = True,
    sagittal_x: int = JITTER_SAGITTAL_X,
    coronal_y: int = JITTER_CORONAL_Y,
    axial_z: int = JITTER_AXIAL_Z,
) -> None:
    """Sagittal / koronal / axial durch ein gejittertes Volume, Colorbar an der Seite.

    Grüne Kontur = erhaltene rechte Thalamus-Maske (Orientierung wie Abschnitt A.7).
    """
    from matplotlib.lines import Line2D

    vol = np.asarray(volume, dtype=np.float32).squeeze()
    right = (
        (np.asarray(right_mask, dtype=np.float32).squeeze() > 0)
        if right_mask is not None
        else np.zeros(vol.shape, dtype=bool)
    )
    nx, ny, nz = vol.shape
    cx = int(np.clip(sagittal_x, 0, nx - 1))
    cy = int(np.clip(coronal_y, 0, ny - 1))
    cz = int(np.clip(axial_z, 0, nz - 1))
    # Robuster Kontrast: Hintergrund (0) dominiert das Histogramm.
    vmax = float(np.percentile(vol[vol > 0], 99.5)) if np.any(vol > 0) else 1.0
    color_right = (0.15, 0.65, 0.25)

    slices = [
        (np.rot90(vol[cx]), np.rot90(right[cx]), f"sagittal x={cx}"),
        (np.rot90(vol[:, cy]), np.rot90(right[:, cy]), f"koronal y={cy}"),
        (vol[:, :, cz], right[:, :, cz], f"axial z={cz}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    fig.suptitle(title, fontsize=10)
    im = None
    for ax, (v_slc, r_slc, slc_title) in zip(axes, slices):
        im = ax.imshow(v_slc, cmap="gray", vmin=0.0, vmax=vmax)
        if r_slc.any():
            ax.contour(r_slc.astype(float), levels=[0.5], colors=[color_right], linewidths=0.8)
        ax.set_title(slc_title, fontsize=9)
        ax.axis("off")
    fig.legend(
        handles=[Line2D([0], [0], color=color_right, lw=1.5, label="rechter Thalamus (erhalten)")],
        loc="lower center",
        frameon=False,
        fontsize=9,
    )
    # Colorbar auf eigener Achse rechts — sonst schiebt tight_layout sie ins letzte Panel.
    fig.subplots_adjust(left=0.02, right=0.88, top=0.84, bottom=0.10, wspace=0.05)
    fig.colorbar(im, cax=fig.add_axes([0.90, 0.14, 0.015, 0.70]), label="Intensität")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
    if show_inline:
        display(fig)
    plt.close(fig)


holdout_ids = load_ukb_holdout_ids()
jitter_qc_subjects = pick_jitter_subjects(
    N_JITTER_QC_SUBJECTS, seed=JITTER_QC_SEED, exclude=holdout_ids
)
print(f"Holdout-IDs ausgeschlossen: n={len(holdout_ids)}")
print(f"Zufällige Jitter-Subjects (seed={JITTER_QC_SEED}): {jitter_qc_subjects}")

jitter_qc_plot_dir = (
    keras_xai_root
    / "output"
    / "notebooks"
    / "analysis_LRP_for_right_thalamus_volume_based_on_CNN_prediction"
    / "jitter_qc"
)

qc_rows: list[dict[str, object]] = []
for sid in jitter_qc_subjects:
    stats = jitter_qc_stats(sid)
    qc_rows.append(stats)
    _, right_path = jitter_mask_paths(sid)
    right_data = _load_jitter_nii(right_path) if right_path.is_file() else None
    title = (
        f"{JITTER_DATASET}  {sid}  gejittert (rechter Thalamus erhalten)   "
        f"r_thal={stats['r_thalamus']:.3f}  r_außen={stats['r_outside']:.3f}"
    )
    plot_jitter_slices(
        _load_jitter_nii(jitter_volume_path(sid)),
        right_data,
        title=title,
        save_path=jitter_qc_plot_dir / f"jitter_qc_{sid}.png",
        show_inline=SHOW_PLOTS_INLINE,
    )

display(pd.DataFrame(qc_rows))

# %% [markdown]
# ## B.4. LRP-Heatmaps für die gejitterten UKB-Holdout-Volumes
#
# Identisch zu **Abschnitt A.7**, nur ist der Input jetzt das gejitterte Volume
#
# ```
# /mnt/users/andreasre/data/jittered_data/ukb/recon/<subject-id>/mri/
#     T1_mni152_right_thalamus_preserved_others_shuffled.nii.gz
# ```
#
# für **dieselben `N_SUBJECTS` UKB-Holdout-Subjects** wie oben (`dataset_labels["ukb"]`).
# Die Dateien haben schon das Training-FOV `167×212×160`, es wird also nicht neu gecropped.
#
# Pro Subject: Vorhersage → LRP → Heatmap als NIfTI unter
# `RUN_DIR/heatmaps_jittered/ukb/<subject-id>/` → Overlay-Plot mit denselben Schnitten
# (sagittal `x=70`, koronal `y=104`, axial `z=78`), LRP **unnormiert**, rechter Thalamus grün,
# linker Thalamus lila. Thalamus-Masken kommen aus dem Jitter-Ordner, sonst als Fallback
# aus den in Abschnitt A.7 erzeugten Masken (identische Anatomie, `flirt` läuft nicht erneut).
#
# Auswertung am Ende: `pred_original` vs. `pred_jittered` sowie der Anteil von \(|R|\)
# im rechten Thalamus. Erwartung, wenn das Modell wirklich das rechte Thalamus-Volumen liest:
# Vorhersage bleibt stabil und der Relevanz-Anteil im rechten Thalamus steigt, weil das
# permutierte Umfeld keine nutzbare Struktur mehr trägt.
#
# Voraussetzungen: Abschnitte 1–7 gelaufen (`model`, `lrp`, `dataset_labels`, `plot_lrp_overlay`)
# sowie Abschnitt B.3 (Jitter-Pfad-Helfer). Fehlende Jitter-Dateien werden übersprungen und
# am Ende aufgelistet.

# %%
JITTER_HEATMAPS_DIR = RUN_DIR / "heatmaps_jittered" / JITTER_DATASET
N_PLOT_JITTER_SUBJECTS = int(N_SUBJECTS)  # alle Holdout-Subjects plotten

jitter_plot_dir = (
    keras_xai_root
    / "output"
    / "notebooks"
    / "analysis_LRP_for_right_thalamus_volume_based_on_CNN_prediction"
    / "jitter_lrp"
)
jitter_plot_dir.mkdir(parents=True, exist_ok=True)


def _original_prediction(subject_id: str) -> float | None:
    """Vorhersage auf dem *unmanipulierten* Volume aus Abschnitt A.7, falls vorhanden."""
    for rec in globals().get("preds_by_dataset", {}).get(JITTER_DATASET, []):
        if str(rec["subject_id"]) == subject_id:
            return float(rec["prediction"])
    return None


def _right_thalamus_share(heat: np.ndarray, right: np.ndarray) -> float:
    """Anteil der |LRP|-Summe innerhalb der rechten Thalamus-Maske."""
    total = float(np.sum(np.abs(heat)))
    if total <= 0.0:
        return float("nan")
    return float(np.sum(np.abs(heat[right > 0]))) / total


def _original_right_share(subject_id: str, right: np.ndarray) -> float:
    """Gleicher Anteil für die Original-Heatmap aus Abschnitt A.7 (falls vorhanden)."""
    path = (
        RUN_DIR
        / "heatmaps"
        / JITTER_DATASET
        / subject_id
        / f"lrp_heatmap_{JITTER_DATASET}_{subject_id}.nii.gz"
    )
    if not path.is_file():
        return float("nan")
    return _right_thalamus_share(_load_jitter_nii(path), right)


jitter_saved_niftis: list[Path] = []
jitter_rows: list[dict[str, object]] = []
jitter_missing: list[str] = []

df_jitter = dataset_labels[JITTER_DATASET]
print(f"=== {JITTER_DATASET} (gejittert): n={len(df_jitter)} Holdout-Subjects ===")

for i, (_, row) in enumerate(
    tqdm(df_jitter.iterrows(), total=len(df_jitter), desc=f"{JITTER_DATASET}-jittered")
):
    sid = str(row["participant_id"])
    y_true = float(row[pred_var])
    jitter_path = jitter_volume_path(sid)
    if not jitter_path.is_file():
        jitter_missing.append(f"[{JITTER_DATASET}/{sid}] Jitter-Volume fehlt: {jitter_path}")
        continue

    vol = load_volume(str(jitter_path))
    y_pred = float(np.squeeze(model.predict(np.expand_dims(vol, 0), verbose=0)))
    R = lrp(np.expand_dims(vol, 0))[0].numpy()

    nii_path = JITTER_HEATMAPS_DIR / sid / f"lrp_heatmap_{JITTER_DATASET}_jittered_{sid}.nii.gz"
    nii_path.parent.mkdir(parents=True, exist_ok=True)
    save_heatmap_nifti(R, str(jitter_path), nii_path)
    jitter_saved_niftis.append(nii_path)

    left_path, right_path = jitter_mask_paths(sid)
    left_data = _load_jitter_nii(left_path) if left_path.is_file() else None
    right_data = _load_jitter_nii(right_path) if right_path.is_file() else None
    if right_data is None:
        jitter_missing.append(f"[{JITTER_DATASET}/{sid}] rechte Thalamus-Maske fehlt.")

    y_pred_orig = _original_prediction(sid)
    jitter_rows.append(
        {
            "subject_id": sid,
            pred_var: y_true,
            "pred_original": y_pred_orig,
            "pred_jittered": y_pred,
            "delta_pred": None if y_pred_orig is None else y_pred - y_pred_orig,
            "right_share_original": (
                float("nan") if right_data is None else _original_right_share(sid, right_data)
            ),
            "right_share_jittered": (
                float("nan") if right_data is None else _right_thalamus_share(R, right_data)
            ),
        }
    )

    if i < N_PLOT_JITTER_SUBJECTS:
        plot_lrp_overlay(
            R,
            left_data,
            right_data,
            title=(
                f"{JITTER_DATASET} gejittert  {sid}  true={y_true:.0f}  pred={y_pred:.0f}"
                + ("" if y_pred_orig is None else f"  (orig={y_pred_orig:.0f})")
            ),
            save_path=jitter_plot_dir / f"lrp_overlay_jittered_{sid}.png",
            show_inline=SHOW_PLOTS_INLINE,
            sagittal_x=JITTER_SAGITTAL_X,
            coronal_y=JITTER_CORONAL_Y,
            axial_z=JITTER_AXIAL_Z,
        )

print(f"\nJitter-Heatmaps: {len(jitter_saved_niftis)}  |  übersprungen/Warnungen: {len(jitter_missing)}")
for m in jitter_missing:
    print(" -", m)

if jitter_rows:
    jitter_df = pd.DataFrame(jitter_rows)
    display(jitter_df.round(3))
    if jitter_df["delta_pred"].notna().any():
        mae_shift = float(np.mean(np.abs(jitter_df["delta_pred"].dropna().to_numpy())))
        print(f"mittlere |pred_jittered - pred_original|: {mae_shift:.1f}")

# %% [markdown]
# ## B.5. Interaktiver 3D-Plot — Jitter-Modell (erstes Subject je Dataset)
#
# Analog zu **A.11**, aber die LRP-Heatmap kommt vom **auf gejitterten Volumes
# trainierten Modell** (`training_run_05h09m52s_04sep2026`). Input sind weiterhin die
# originalen Holdout-`cropped.nii.gz` (wie in B.1 / Teil C), Thalamus-Masken aus A.7.
#
# Pro Dataset das **erste** Subject: Vorhersage → LRP → NIfTI unter
# `RUN_DIR/heatmaps_jitter_model/<dataset>/<subject-id>/` → interaktives HTML
# (Gehirnkontur + beide FreeSurfer-Masken + unnormierte LRP).
#
# Voraussetzungen: A.4–A.7 (Labels, Loader, Masken), A.11 (`plot_thalamus_lrp_3d`).
# `_ensure_jitter_model` (unten) nutzt `jitter_model` aus B.1 oder lädt nach.

# %%
def _ensure_jitter_model():
    """B.1 legt `jitter_model` an; falls noch nicht gelaufen, hier nachladen."""
    if "jitter_model" in globals() and globals()["jitter_model"] is not None:
        return globals()["jitter_model"]
    run_dir = Path(
        "~/data/nn-trainings/mri/Right-Whole_thalamus/"
        "training_run_05h09m52s_04sep2026"
    ).expanduser().resolve()
    model_path = run_dir / "model.keras"
    config_path = run_dir / "config.yaml"
    if not model_path.is_file():
        raise FileNotFoundError(f"Jitter-Modell fehlt: {model_path}")
    jcfg = OmegaConf.load(config_path)
    with open_dict(jcfg):
        jcfg.paths.csv_dir = str(run_dir)
        if "prediction" not in jcfg.training:
            jcfg.training.prediction = {}
        jcfg.training.prediction.batch_size = int(PRED_BATCH_SIZE)
    jm = _build_single_device_model(jcfg)
    w0 = jm.get_weights()[0].copy()
    jm.load_weights(str(model_path))
    if float(np.mean(np.abs(jm.get_weights()[0] - w0))) < 1e-9:
        raise RuntimeError("Jitter-Modell-Gewichte wurden nicht geladen.")
    return jm


JITTER_MODEL_HEATMAPS_DIR = RUN_DIR / "heatmaps_jitter_model"
plot_dir_3d_jitter = (
    keras_xai_root
    / "output"
    / "notebooks"
    / "analysis_LRP_for_right_thalamus_volume_based_on_CNN_prediction"
)
plot_dir_3d_jitter.mkdir(parents=True, exist_ok=True)

jitter_model_3d = _ensure_jitter_model()
jitter_lrp_3d = LRP(
    jitter_model_3d,
    layer=len(jitter_model_3d.layers) - 1,
    idx=0,
    strategy=strategy,
)

for dataset_id in DATASETS:
    df = dataset_labels[dataset_id]
    if df.empty:
        print(f"[{dataset_id}] keine Subjects.")
        continue

    row = df.iloc[0]
    sid = str(row["participant_id"])
    y_true = float(row[pred_var])
    t1_path = Path(str(row["filepath"]))
    mask_dir = RUN_DIR / "heatmaps" / dataset_id / sid
    left_path = mask_dir / "aseg_mni152_left_thalamus_cropped.nii.gz"
    right_path = mask_dir / "aseg_mni152_right_thalamus_cropped.nii.gz"

    missing = [p for p in (t1_path, left_path, right_path) if not p.is_file()]
    if missing:
        print(f"[{dataset_id}/{sid}] Dateien fehlen:")
        for p in missing:
            print("  -", p)
        continue

    vol = load_volume(str(t1_path))
    x = np.expand_dims(vol, 0)
    y_pred = float(np.squeeze(jitter_model_3d.predict(x, verbose=0)))
    R = jitter_lrp_3d(x)[0].numpy()

    hm_path = (
        JITTER_MODEL_HEATMAPS_DIR
        / dataset_id
        / sid
        / f"lrp_heatmap_jitter_model_{dataset_id}_{sid}.nii.gz"
    )
    save_heatmap_nifti(R, str(t1_path), hm_path)
    print(
        f"\n[{dataset_id}] Jitter-Modell 3D für erstes Subject: {sid}  "
        f"true={y_true:.1f}  pred={y_pred:.1f}"
    )
    print("Heatmap:", hm_path)

    plot_thalamus_lrp_3d(
        dataset_id=dataset_id,
        subject_id=sid,
        volume=_load_vol(t1_path),
        heatmap=R,
        left_mask=_load_vol(left_path),
        right_mask=_load_vol(right_path),
        y_true=y_true,
        y_pred=y_pred,
        pred_var_name=pred_var,
        title_suffix="Jitter-Modell",
        save_html=(
            plot_dir_3d_jitter
            / f"{dataset_id}_{sid}_thalamus_lrp_3d_jitter_model.html"
        ),
    )

# %% [markdown]
# ## C. Relevanzerhaltung Schicht für Schicht
#
# ### Motivation
#
# Layer-wise Relevance Propagation soll Relevanz **weder erzeugen noch vernichten**,
# sondern nur umverteilen. Für jede Schicht $\ell$ gilt idealerweise
#
# $$
# \sum_j R_j^{(\ell)} \;=\; \sum_k R_k^{(\ell+1)}
# $$
#
# und am Eingang
#
# $$
# \sum_i R_i^{(\mathrm{input})} \;\approx\; f(x)
# $$
#
# (hier: die skalare Regression-Ausgabe / das maskierte Logit).
#
# Sprünge in $\sum R$ entlang des Rückwärtspfads kommen typischerweise von:
#
# * **ε-Stabilisierung** am Dense-Ausgang (Nenner wird absichtlich vergrößert)
# * **flat**-Regeln an den frühen Convs ($a\leftarrow 1$, streng genommen nicht erhaltend)
# * Bias-Behandlung und Pooling-Strategien
# * echten Implementierungsfehlern (dann oft Größenordnungs-Sprünge)
#
# Dieser Abschnitt misst $\sum R$ **pro LRP-Schicht** für das **erste** Subject aus
# `dataset_labels` je Eintrag in `DATASETS` / `DATASET_DIRS` — einmal mit dem
# **Original-Modell** (Teil A) und einmal mit dem **auf gejitterten Volumes trainierten
# Modell** (Teil B). Die Inputs sind jeweils die Holdout-`filepath`s aus Teil A
# (originales `cropped.nii.gz`), damit der Modellvergleich fair bleibt.
#
# Dargestellt werden die **gewichtstragenden / Pooling-Schichten** des Rückwärtspfads
# (Dense, GlobalAveragePool, Conv3D, MaxPool), beschriftet mit dem Forward-Namen
# (`block-0_conv`, `predictions`, …), der LRP-Regel (`flat`, `αβ`, `ε`, …) und dem
# numerischen $\sum R$. Zusätzlich speichern wir die vollständige Schichttabelle
# (inkl. ReLU/NoOp) als CSV.
#
# ### Erwartetes Ergebnis (Interpretation)
#
# * Die Kurve sollte nach dem Dense-(ε)-Schritt weitgehend **flach** sein; der größte
#   relative Sprung liegt oft am ε-Dense und ggf. an den flat-Convs nahe dem Input.
# * **Original- vs. Jitter-Modell:** Absolute $\sum R$-Niveaus können differieren
#   (andere Vorhersage $f(x)$), die *Form* der Erhaltungskurve sollte bei gleicher
#   Composite-Strategie aber ähnlich sein. Deutlich andere Lecks würden auf
#   Architektur-/Strategie-Unterschiede oder ein kaputtes Gewichteladen hinweisen.
# * IXI vs. UKB: bei stabilem Explainer ähnliche Schichtprofile; starke
#   Datensatz-Abhängigkeit der Lecks wäre ungewöhnlich und prüfenswert.
#
# Die Code-Zelle druckt Bilanz (`Start` / `Ende` / Erhaltungsquote) und speichert Plots
# unter `output/notebooks/.../layerwise_relevance/`.

# %%
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.layers import Lambda
from explainability.layers import PoolingLRPLayer, StandardLRPLayer
from explainability.layers.layer import LRPLayer

LAYERWISE_OUT_DIR = (
    keras_xai_root
    / "output"
    / "notebooks"
    / "analysis_LRP_for_right_thalamus_volume_based_on_CNN_prediction"
    / "layerwise_relevance"
)
LAYERWISE_OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_backward_start(lrp_model) -> int:
    for i, layer in enumerate(lrp_model.layers):
        if isinstance(layer, Lambda) and "output_mask" in layer.name:
            return i
    for i, layer in enumerate(lrp_model.layers):
        if isinstance(layer, LRPLayer):
            return i
    raise RuntimeError("Kein Rückwärtspfad im LRP-Modell gefunden")


def layer_rule_tag(layer) -> str:
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
        if getattr(layer, "ignore_bias", False):
            parts.append("ignore_bias")
        return ", ".join(parts) if parts else "LRP-0"
    if isinstance(layer, PoolingLRPLayer):
        return str(getattr(layer, "strategy", "?"))
    return "—"


def forward_anchor_labels(keras_model) -> list[str]:
    """Forward-Namen der Convs/Pools/Dense, Reihenfolge Ausgabe→Eingabe (wie LRP)."""
    labels: list[str] = []
    for layer in keras_model.layers:
        t = type(layer).__name__
        short = layer.name.replace("sfcn-reg_", "")
        if t == "Dense":
            labels.append(f"dense:{short}")
        elif t == "GlobalAveragePooling3D":
            labels.append(f"gap:{short}")
        elif t in ("AveragePooling3D", "MaxPooling3D"):
            labels.append(f"maxpool:{short}" if "Max" in t else f"avgpool:{short}")
        elif t == "Conv3D":
            # block-0_conv → conv1-ähnlich; top_conv separat
            labels.append(f"conv:{short}")
    return labels[::-1]


def collect_layer_relevance(
    lrp_model,
    x: np.ndarray,
    *,
    keras_model,
    key_layers_only: bool = True,
) -> pd.DataFrame:
    """ΣR je Schicht ab Mask-Lambda.

    `key_layers_only=True`: nur Mask + StandardLRP + Pooling (lesbarer Plot,
    weniger RAM). Vollständige Tabelle: `key_layers_only=False`.
    """
    start = find_backward_start(lrp_model)
    all_layers = list(lrp_model.layers[start:])
    if key_layers_only:
        layers = [
            lyr
            for lyr in all_layers
            if isinstance(lyr, (StandardLRPLayer, PoolingLRPLayer))
            or (isinstance(lyr, Lambda) and "output_mask" in lyr.name)
        ]
    else:
        layers = all_layers

    # Ein Output nach dem anderen — vermeidet Multi-Output-OOM bei 3D-MRT.
    rows = []
    anchors = forward_anchor_labels(keras_model)
    anchor_i = 0
    for layer in layers:
        probe = KerasModel(lrp_model.input, layer.output)
        R = np.asarray(probe.predict(x, verbose=0))
        sum_r = float(np.sum(R))
        sum_pos = float(np.sum(R[R > 0])) if R.size else 0.0
        sum_neg = float(np.sum(R[R < 0])) if R.size else 0.0
        idx = list(lrp_model.layers).index(layer)

        if isinstance(layer, (StandardLRPLayer, PoolingLRPLayer)) and anchor_i < len(anchors):
            friendly = anchors[anchor_i]
            anchor_i += 1
        elif isinstance(layer, Lambda) and "output_mask" in layer.name:
            friendly = "mask:output_logit"
        else:
            friendly = type(layer).__name__.replace("LRP", "")

        rows.append(
            {
                "layer_idx": idx,
                "label": friendly,
                "name": layer.name,
                "type": type(layer).__name__,
                "rule": layer_rule_tag(layer),
                "shape": tuple(int(s) for s in R.shape),
                "sum_R": sum_r,
                "sum_pos": sum_pos,
                "sum_neg": sum_neg,
                "sum_abs": float(np.sum(np.abs(R))),
            }
        )
        del probe, R

    df = pd.DataFrame(rows)
    df["delta_sum_R"] = df["sum_R"].diff()
    r0 = float(df["sum_R"].iloc[0]) if len(df) else float("nan")
    df["ratio_to_start"] = df["sum_R"] / r0 if r0 else np.nan
    return df


def plot_layerwise_sum_R(
    df: pd.DataFrame,
    *,
    title: str,
    save_path: Path | None = None,
    show_inline: bool = True,
) -> None:
    """Balkendiagramm ΣR mit Layer-Label, Regel und Zahlenwert."""
    fig, ax = plt.subplots(figsize=(max(10, 0.7 * len(df) + 4), 5.5))
    x = np.arange(len(df))
    bars = ax.bar(x, df["sum_R"], color="C0", alpha=0.85, width=0.75)
    r0 = float(df["sum_R"].iloc[0])
    r1 = float(df["sum_R"].iloc[-1])
    ax.axhline(r0, color="C1", ls="--", lw=1.2, label=f"Start ΣR = {r0:.3g}")
    ax.axhline(r1, color="C2", ls=":", lw=1.2, label=f"Ende ΣR = {r1:.3g}")

    tick_labels = []
    for _, row in df.iterrows():
        rule = row["rule"] if row["rule"] != "—" else ""
        tick_labels.append(f"{row['label']}\n{rule}" if rule else str(row["label"]))
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(r"$\sum R$")
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    ymax = float(np.nanmax(np.abs(df["sum_R"].to_numpy()))) or 1.0
    for bar, val in zip(bars, df["sum_R"].to_numpy()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01 * ymax * np.sign(bar.get_height() or 1.0),
            f"{val:.3g}",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=7,
            rotation=90,
        )
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print("Plot:", save_path)
    if show_inline:
        display(fig)
    plt.close(fig)


def run_layerwise_for_model(
    *,
    tag: str,
    keras_model,
    lrp_strategy,
) -> list[pd.DataFrame]:
    """Erstes Subject je Dataset: LRP bauen, ΣR sammeln, plotten."""
    lrp_here = LRP(
        keras_model,
        layer=len(keras_model.layers) - 1,
        idx=0,
        strategy=lrp_strategy,
    )
    frames: list[pd.DataFrame] = []
    for dataset_id in DATASETS:
        df_labels = dataset_labels[dataset_id]
        if df_labels.empty:
            print(f"[{tag}/{dataset_id}] keine Labels.")
            continue
        row = df_labels.iloc[0]
        sid = str(row["participant_id"])
        path = str(row["filepath"])
        y_true = float(row[pred_var])
        if not Path(path).is_file():
            print(f"[{tag}/{dataset_id}/{sid}] Volume fehlt: {path}")
            continue

        vol = load_volume(path)
        x = np.expand_dims(vol, 0)
        y_pred = float(np.squeeze(keras_model.predict(x, verbose=0)))
        print(
            f"\n=== [{tag}] {dataset_id}/{sid}  true={y_true:.1f}  pred={y_pred:.1f} ==="
        )

        df = collect_layer_relevance(lrp_here, x, keras_model=keras_model)
        df.insert(0, "model_tag", tag)
        df.insert(1, "dataset_id", dataset_id)
        df.insert(2, "subject_id", sid)
        df.insert(3, "y_true", y_true)
        df.insert(4, "y_pred", y_pred)
        frames.append(df)

        r0 = float(df["sum_R"].iloc[0])
        r1 = float(df["sum_R"].iloc[-1])
        kept = (100.0 * r1 / r0) if r0 else float("nan")
        print(df[["label", "type", "rule", "sum_R", "delta_sum_R", "ratio_to_start"]].to_string(
            index=False, float_format=lambda v: f"{v:10.4f}"
        ))
        print(f"Bilanz: Start={r0:.4f}  Ende={r1:.4f}  erhalten={kept:.2f}%")

        csv_path = LAYERWISE_OUT_DIR / f"sum_R_{tag}_{dataset_id}_{sid}.csv"
        df.to_csv(csv_path, index=False)
        print("CSV:", csv_path)

        plot_layerwise_sum_R(
            df,
            title=(
                f"{tag}  |  {dataset_id}  {sid}\n"
                f"true={y_true:.0f}  pred={y_pred:.0f}  |  "
                f"ΣR start→end {r0:.3g}→{r1:.3g} ({kept:.1f}% erhalten)"
            ),
            save_path=LAYERWISE_OUT_DIR / f"sum_R_{tag}_{dataset_id}_{sid}.png",
            show_inline=SHOW_PLOTS_INLINE,
        )
    return frames


# Dieselbe Composite-Strategie wie in A.5.
layerwise_strategy = strategy

print("Teil C — Original-Modell (A)")
frames_a = run_layerwise_for_model(
    tag="A_original_model",
    keras_model=model,
    lrp_strategy=layerwise_strategy,
)

print("\nTeil C — Jitter-Modell (B)")
jitter_model_c = _ensure_jitter_model()
frames_b = run_layerwise_for_model(
    tag="B_jittered_model",
    keras_model=jitter_model_c,
    lrp_strategy=layerwise_strategy,
)

all_frames = frames_a + frames_b
if all_frames:
    summary = pd.concat(all_frames, ignore_index=True)
    summary_path = LAYERWISE_OUT_DIR / "sum_R_all_models_datasets.csv"
    summary.to_csv(summary_path, index=False)
    print("\nGesamttabelle:", summary_path)

    # Kompakter Vergleich: Erhaltungsquote je Modell × Dataset
    cmp_rows = []
    for fr in all_frames:
        r0 = float(fr["sum_R"].iloc[0])
        r1 = float(fr["sum_R"].iloc[-1])
        cmp_rows.append(
            {
                "model_tag": fr["model_tag"].iloc[0],
                "dataset_id": fr["dataset_id"].iloc[0],
                "subject_id": fr["subject_id"].iloc[0],
                "y_pred": float(fr["y_pred"].iloc[0]),
                "sum_R_start": r0,
                "sum_R_end": r1,
                "pct_kept": (100.0 * r1 / r0) if r0 else float("nan"),
            }
        )
    cmp_df = pd.DataFrame(cmp_rows)
    display(cmp_df.round(3))
    cmp_df.to_csv(LAYERWISE_OUT_DIR / "sum_R_conservation_summary.csv", index=False)

# %% [markdown]
# ## C. Nachlese der Plots
#
# Nach dem Lauf der Code-Zelle oben solltest du pro Modell × Dataset einen Plot und eine
# CSV unter `output/notebooks/analysis_LRP_for_right_thalamus_volume_based_on_CNN_prediction/layerwise_relevance/`
# haben.
#
# **So liest du die Abbildung:**
#
# 1. **x-Achse:** Forward-Layer-Label (`dense:predictions`, `gap:top_pool`,
#    `conv:block-k_conv`, `maxpool:…`) plus LRP-Regel (`ε=…`, `α=2,β=1`, `flat`).
# 2. **Balkenhöhe / Zahl:** $\sum R$ in dieser Schicht (auch als Annotation am Balken).
# 3. **Orange gestrichelt / grün punktiert:** Start-ΣR (maskiertes Logit) vs. Ende-ΣR
#    (letzte dargestellte Schicht Richtung Input).
#
# **Was die Bilanz bedeutet:**
#
# * `erhalten ≈ 100%` → Composite-LRP konserviert die Masse gut für dieses Subject.
# * Deutlicher Abfall schon am **Dense/ε** ist erwartet (Stabilisierungsterm).
# * Weitere Sprünge an **flat**-Convs nahe dem Input sind ebenfalls typisch für die
#   Strategie aus A.5; Größenordnungs-Sprünge wären ein Warnsignal.
# * Vergleich **A_original_model** vs. **B_jittered_model**: ähnliche Kurvenform bei
#   gleichem Explainer spricht dafür, dass beide Modelle denselben LRP-Pfad korrekt
#   durchlaufen — Unterschiede in der *Höhe* folgen aus unterschiedlichen $f(x)$.
