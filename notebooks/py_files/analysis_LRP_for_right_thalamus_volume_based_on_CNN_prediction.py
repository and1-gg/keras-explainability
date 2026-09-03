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
# # LRP-Analyse: rechtes Thalamus-Volumen (CNN)
#
# Dieses Notebook prüft, **wo** ein auf `Right-Whole_thalamus` trainiertes 3D-CNN
# seine Vorhersage herleitet (Layer-wise Relevance Propagation).
#
# Ablauf:
#
# 1. `N_SUBJECTS` Holdout-Fälle aus **IXI** und **UKB** laden
# 2. Vorhersagen + LRP-Heatmaps berechnen
# 3. FreeSurfer-`aseg` als Thalamus-Maske nach MNI152 bringen
# 4. Anteil der Relevanz **im Thalamus** vs. außerhalb auswerten
# 5. `N_PLOT_SUBJECTS` Fälle pro Dataset plotten
# 6. Interaktiver 3D-Plot (Gehirn + beide Thalamus-Masken + unnormierte LRP)
#    für das **erste** Subject von IXI und UKB
#

# %% [markdown]
# ## 1. Imports
#

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
# ## 2. Konfiguration
#
# - `N_SUBJECTS`: wie viele Fälle **pro Dataset** berechnet werden (Heatmap + Maske).
# - `N_PLOT_SUBJECTS`: wie viele davon geplottet werden (PNG + optional inline).
#   Darf höchstens `N_SUBJECTS` sein, Default ist 2.
#

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
# ## 3. Repo-Pfade (keras-explainability + pybrainmetrics)
#

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
# ## 4. Labels laden
#
# UKB kommt aus dem offiziellen **predict-Split** (Holdout, nicht Training).
# IXI aus `subjects_dl_input.tsv` bzw. FreeSurfer-Thalamus-Stats.
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
# ## 5. Modell, Volume-Loader, LRP
#
# Single-Device-Modell (keine `MirroredStrategy`). LRP-Composite für SFCN:
# zwei `flat`-Schichten, vier αβ-Schichten, ε am Dense-Ausgang.
# Relevanz außerhalb des Gehirns (`voxel==0`) wird auf 0 gesetzt.
#

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


def mask_explanation(volume: np.ndarray, explanation: np.ndarray) -> np.ndarray:
    x = volume.squeeze()
    expl = explanation.squeeze().astype(np.float32)
    return expl * (x != 0).astype(np.float32)


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
# ## 6. Thalamus-Maske (`aseg` → MNI152 → Crop)
#
# FreeSurfer-Labels: links=10, rechts=49. Crop wie Training-FOV `167×212×160`.
# Bereits vorhandene Masken werden übersprungen.
#

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
# ## 7. Heatmaps + Masken für alle Subjects
#
# Pro Dataset `N_SUBJECTS` Fälle. Overlay-Plots (LRP **unnormiert**, rechter Thalamus grün, linker Thalamus lila) für die ersten `N_PLOT_SUBJECTS`.
# Schnitte fest: sagittal `x=70`, koronal `y=104`, axial `z=78`.
#

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
        preds_by_dataset[dataset_id].append(
            {"subject_id": sid, pred_var: y_true, "prediction": y_pred}
        )
        R_masked = mask_explanation(vol, lrp(np.expand_dims(vol, 0))[0].numpy())
        nii_path = heatmaps_dir / sid / f"lrp_heatmap_{dataset_id}_{sid}.nii.gz"
        nii_path.parent.mkdir(parents=True, exist_ok=True)
        save_heatmap_nifti(R_masked, path, nii_path)
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
                R_masked,
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
        #         R_masked,
        #         title=f"{dataset_id}  {sid}  Slice-Zoom um LRP-Peak",
        #         save_path=plot_dir / f"lrp_slices_{sid}.png",
        #         show_inline=SHOW_PLOTS_INLINE,
        #     )

print(f"\nHeatmaps: {len(saved_niftis)}  |  Masken-Fehler: {len(mask_errors)}")
for m in mask_errors:
    print(" -", m)


# %% [markdown]
# ## 8. True vs. Predicted
#
# Scatter und Pearson-r / MAE pro Dataset (`n = N_SUBJECTS`).
#

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
# ## 9. Relevanz im Thalamus
#
# Anteil der |LRP|-Summe in linker / rechter Thalamus-Maske vs. außerhalb.
# Kompakte Voxel-Statistik statt einer Tabelle aller ~5,7 Mio. Voxel.
#

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
# ## 10. Interaktiver 3D-Plot (erstes Subject je Dataset)
#
# Für das **erste** IXI- und UKB-Subject: Gehirnkontur plus beide FreeSurfer-Thalamus-Masken
# (`aseg` 10 links / 49 rechts) und die **unnormierte** LRP-Heatmap.
#
# Maus: drehen, Scrollrad: zoomen, Shift+Ziehen: verschieben. Farben wie in den 2D-Overlays
# (lila / grün). LRP-Punkte sind die Voxel mit dem größten \(|R|\) (sonst ~2 Mio. Punkte).
#

# %%
import plotly.graph_objects as go
from IPython.display import HTML, display
from scipy.ndimage import binary_erosion

# Für JupyterHub *und* nbconvert-HTML:
# fig.show()/notebook-Renderer → oft height:100% ohne Elternhöhe → leerer Export.
# Stattdessen fig.to_html(...) als text/html ausgeben (feste Pixelhöhe).
_PLOTLY_JS_DONE = False

# Farben analog zu plot_lrp_overlay (Abschnitt 7)
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
    title = (
        f"<b>{dataset_id.upper()} · {subject_id}</b><br>"
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

