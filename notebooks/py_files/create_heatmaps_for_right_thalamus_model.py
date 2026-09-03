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
import shutil
import subprocess
import sys
from dataclasses import dataclass
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
DATASET_DIRS = {
    "ixi": DATA_ROOT / "ixi",
    #"ds003114_OvarianHormones": DATA_ROOT / "ds003114_OvarianHormones",
    #"ds004711_AgeRisk": DATA_ROOT / "ds004711_AgeRisk",
    "ukb": DATA_ROOT / "ukb",
}

# Alle Datensätze, für die Heatmaps + Thalamus-Masken erzeugt werden.
DATASETS = [
    "ixi",
    "ukb",
    #"ds003114_OvarianHormones",
    #"ds004711_AgeRisk",
]

N_SUBJECTS = 5
PRED_BATCH_SIZE = 8
# Wie viele Subjects *pro Dataset* eine LRP-Figur bekommen (PNG + optional inline).
# 1 reicht zum Spot-Check; höhere Werte blähen die .ipynb auf → 413 beim Speichern.
N_HEATMAP_PLOTS = 1
# Inline-Plots in der Notebook-Ausgabe einbetten? PNG wird immer geschrieben.
# False vermeidet "413 Request Entity Too Large" beim Speichern der .ipynb.
SHOW_HEATMAP_PLOTS_INLINE = False
# |LRP| > eps gilt als „relevant“ für Thalamus-Overlap (0 = jedes != 0 Voxel).
LRP_RELEVANCE_EPS = 0.0
SHOW_OVERLAP_PLOTS_INLINE = True

# Nur für UKB: Holdout-Subjects aus dem offiziellen predict-Split (nicht train.tsv).
# Die ersten N_SUBJECTS Zeilen sind disjoint zum Training (siehe pyment-and1).
UKB_HOLDOUT_PREDICT_TSV = Path(
    "~/git-repos/pyment-and1/training_runs/input_files/mri/"
    "right_whole_thalamus/predict.tsv"
).expanduser().resolve()

MNI152_1MM = Path("/usr/local/fsl/data/standard/MNI152_T1_1mm.nii.gz")

MODEL_PATH = RUN_DIR / "model.keras"
CONFIG_PATH = RUN_DIR / "config.yaml"

unknown = [d for d in DATASETS if d not in DATASET_DIRS]
if unknown:
    raise ValueError(
        f"Unbekannte DATASETS={unknown!r}. Erlaubt: {list(DATASET_DIRS)}"
    )
if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(CONFIG_PATH)
if not MNI152_1MM.is_file():
    raise FileNotFoundError(f"MNI152-Referenz fehlt: {MNI152_1MM}")

print("RUN_DIR: ", RUN_DIR)
print("DATASETS:")
for ds in DATASETS:
    print(f"  {ds} → {DATASET_DIRS[ds]}")


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


def print_subject_filepaths(df: pd.DataFrame, dataset_id: str) -> None:
    """Subject-ID und vollständigen Pfad zu cropped.nii.gz ausgeben."""
    id_col = "participant_id" if "participant_id" in df.columns else "subject-id"
    print(f"[{dataset_id}] Subjects + cropped.nii.gz (n={len(df)}):")
    for _, row in df.iterrows():
        print(f"  {row[id_col]}  {row['filepath']}")


cfg = OmegaConf.load(CONFIG_PATH)
pred_var = cfg.data.prediction_variable

dataset_labels: dict[str, pd.DataFrame] = {}
for dataset_id in DATASETS:
    ukb_labels_file = UKB_HOLDOUT_PREDICT_TSV if dataset_id == "ukb" else None
    df = load_dataset_labels(
        DATASET_DIRS[dataset_id],
        pred_var,
        N_SUBJECTS,
        labels_file=ukb_labels_file,
    )
    print_subject_filepaths(df, dataset_id)
    labels_tsv = RUN_DIR / f"{dataset_id}_predict_labels_n{len(df)}.tsv"
    df.to_csv(labels_tsv, sep="\t", index=False)
    dataset_labels[dataset_id] = df
    print(f"[{dataset_id}] labels → {labels_tsv}")

with open_dict(cfg):
    cfg.paths.csv_dir = str(RUN_DIR)
    # Platzhalter; Prediction-CSV wird pro Dataset geschrieben, Modell braucht sie nicht.
    first_ds = DATASETS[0]
    cfg.data.predict_labels_file = str(
        RUN_DIR / f"{first_ds}_predict_labels_n{len(dataset_labels[first_ds])}.tsv"
    )
    if "prediction" not in cfg.training:
        cfg.training.prediction = {}
    cfg.training.prediction.batch_size = int(PRED_BATCH_SIZE)

print("pred_var:", pred_var)

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
# Volume-Loader (gleiche Pipeline wie Training/Prediction).
# Normierung: cfg.preprocessing.normalization_factor (= 1.0 für dieses Modell),
# NICHT sigma=255 wie im Brain-Age-Demo-Notebook.

from pybrainmetrics.data.dataset import (
    _load_single_volume,
    _load_single_volume_native,
)

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
    show_inline: bool = True,
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
    if show_inline:
        display(fig)
    plt.close(fig)
    return expl, peak


# %%
# aseg → MNI152 → Thalamus-Maske (links+rechts) → Crop auf Modell-FOV.
# Nutzt flirt / fslmaths; Zwischenprodukte unter heatmaps/<ds>/<sid>/.
#
# FSL-Env (entspricht: export FSLDIR=…; source $FSLDIR/etc/fslconf/fsl.sh).
# Ohne FSLOUTPUTTYPE scheitert flirt mit:
#   "Environment variable FSLOUTPUTTYPE is not set!"
# FreeSurfer/conda-compscy werden hier nicht gebraucht (nur FSL-Binaries).


def _require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{what} fehlt: {path}")
    return path.resolve()


def setup_fsl_environment(
    fsldir: str | Path | None = None,
) -> dict[str, str]:
    """Setzt FSLDIR, FSLOUTPUTTYPE und PATH für flirt/fslmaths (auch in Subprocesses)."""
    env = os.environ
    root = Path(fsldir or env.get("FSLDIR") or "/usr/local/fsl").expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"FSLDIR existiert nicht: {root}")

    env["FSLDIR"] = str(root.resolve())
    if not env.get("FSLOUTPUTTYPE"):
        # wie in $FSLDIR/etc/fslconf/fsl.sh
        env["FSLOUTPUTTYPE"] = "NIFTI_GZ"

    path_prepend: list[str] = []
    share_bin = root / "share" / "fsl" / "bin"
    fsl_bin = root / "bin"
    if share_bin.is_dir():
        path_prepend.append(str(share_bin))
    if fsl_bin.is_dir():
        path_prepend.append(str(fsl_bin))
    current = env.get("PATH", "")
    for p in reversed(path_prepend):
        if p and p not in current.split(os.pathsep):
            current = p + os.pathsep + current
    env["PATH"] = current

    # Optional: FreeSurfer-Home setzen (ohne Setup-Script; flirt braucht es nicht).
    if not env.get("FREESURFER_HOME"):
        fs_home = Path("/usr/local/freesurfer")
        if fs_home.is_dir():
            env["FREESURFER_HOME"] = str(fs_home)

    print(
        f"FSLDIR={env['FSLDIR']}  FSLOUTPUTTYPE={env['FSLOUTPUTTYPE']}  "
        f"FREESURFER_HOME={env.get('FREESURFER_HOME', '—')}"
    )
    return dict(env)


def _fsl_bin(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    fsldir = Path(os.environ.get("FSLDIR", "/usr/local/fsl"))
    for sub in ("share/fsl/bin", "bin"):
        cand = fsldir / sub / name
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError(
        f"FSL-Tool {name!r} nicht gefunden. PATH prüfen oder FSL installieren."
    )


def _run_cmd(cmd: list[str], *, label: str) -> None:
    print(f"  $ {' '.join(cmd)}")
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            env=os.environ,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        detail = stderr or stdout or str(exc)
        raise RuntimeError(f"{label} fehlgeschlagen:\n{detail}") from exc


_ = setup_fsl_environment()


@dataclass
class AsegThalamusMaskPipeline:
    """Erzeugt eine gecroppte Thalamus-aseg-Maske in MNI152 für ein Subject."""

    dataset_id: str
    dataset_dir: Path
    heatmaps_dir: Path
    mni152_ref: Path = MNI152_1MM
    # Gleiches Crop wie Training-FOV (167×212×160); kein /255 — Maske bleibt 0/1.
    crop_slices: tuple[slice, slice, slice] = (
        slice(6, 173),
        slice(2, 214),
        slice(0, 160),
    )

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
        # Bevorzugt mri/, Fallback ohne mri/ (wie in manchen Skripten skizziert).
        aseg_candidates = [
            mri / "aseg_reoriented.nii.gz",
            self.dataset_dir / "recon" / subject_id / "aseg_reoriented.nii.gz",
        ]
        aseg = next((p for p in aseg_candidates if p.is_file()), None)
        if aseg is None:
            raise FileNotFoundError(
                f"[{self.dataset_id}/{subject_id}] aseg_reoriented fehlt. "
                f"Geprüft: {aseg_candidates}"
            )
        return brainmask, aseg.resolve()

    def crop_nifti(self, input_path: Path, output_path: Path) -> None:
        _require_file(input_path, "Crop-Input")
        img = nib.load(str(input_path))
        cropped = img.slicer[self.crop_slices]
        data = np.asarray(cropped.get_fdata(), dtype=np.float32)
        out = nib.Nifti1Image(data, cropped.affine, cropped.header)
        out.header.set_data_dtype(np.float32)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(out, str(output_path))

    def run(self, subject_id: str) -> Path:
        """Vollständige Pipeline; liefert Pfad zur finalen gecroppten Maske."""
        _require_file(self.mni152_ref, "MNI152-Referenz")
        flirt = _fsl_bin("flirt")
        fslmaths = _fsl_bin("fslmaths")

        brainmask, aseg = self.resolve_inputs(subject_id)
        work = self.subject_work_dir(subject_id)
        work.mkdir(parents=True, exist_ok=True)
        self.heatmaps_dir.mkdir(parents=True, exist_ok=True)

        mni152_out = work / "mni152.nii.gz"
        xfm = work / "T1_to_mni152.mat"
        aseg_mni = work / "aseg_mni152.nii.gz"
        left = work / "aseg_mni152_left_thalamus.nii.gz"
        right = work / "aseg_mni152_right_thalamus.nii.gz"
        both = work / "aseg_mni152_thalamus_mask.nii.gz"
        both_cropped_work = work / "aseg_mni152_thalamus_mask_cropped.nii.gz"
        final = self.final_mask_path(subject_id)

        # 1) Trafo brainmask → MNI152 (dof 6)
        _run_cmd(
            [
                flirt,
                "-in", str(brainmask),
                "-out", str(mni152_out),
                "-ref", str(self.mni152_ref),
                "-dof", "6",
                "-omat", str(xfm),
            ],
            label=f"flirt brainmask→MNI152 ({subject_id})",
        )
        _require_file(xfm, "Trafo-Matrix T1_to_mni152.mat")

        # 2) aseg mit derselben Trafo (nearestneighbour)
        _run_cmd(
            [
                flirt,
                "-in", str(aseg),
                "-out", str(aseg_mni),
                "-ref", str(self.mni152_ref),
                "-dof", "6",
                "-applyxfm",
                "-init", str(xfm),
                "-interp", "nearestneighbour",
            ],
            label=f"flirt aseg→MNI152 ({subject_id})",
        )
        _require_file(aseg_mni, "aseg_mni152")

        # 3) Hemisphären-Masken (FreeSurfer: links=10, rechts=49) + vereinigen
        _run_cmd(
            [
                fslmaths, str(aseg_mni),
                "-thr", "10", "-uthr", "10", "-bin", str(left),
            ],
            label=f"fslmaths left thalamus ({subject_id})",
        )
        _run_cmd(
            [
                fslmaths, str(aseg_mni),
                "-thr", "49", "-uthr", "49", "-bin", str(right),
            ],
            label=f"fslmaths right thalamus ({subject_id})",
        )
        _run_cmd(
            [
                fslmaths, str(left),
                "-add", str(right),
                "-bin", str(both),
            ],
            label=f"fslmaths combine thalamus ({subject_id})",
        )
        _require_file(both, "aseg_mni152_thalamus_mask")

        # 4) Crop auf Modell-FOV; finales Deliverable unter heatmaps/<dataset-id>/
        self.crop_nifti(both, both_cropped_work)
        self.crop_nifti(both, final)
        # optionale Hemisphären-Crops (wie in der Bash-Skizze)
        self.crop_nifti(left, work / "aseg_mni152_left_thalamus_cropped.nii.gz")
        self.crop_nifti(right, work / "aseg_mni152_right_thalamus_cropped.nii.gz")

        return _require_file(final, "finale Thalamus-Maske")


# %%
# Pro Dataset: LRP-Heatmaps + aseg-Thalamus-Masken für alle N_SUBJECTS.

saved_niftis: list[Path] = []
saved_masks: list[Path] = []
mask_errors: list[str] = []
preds_by_dataset: dict[str, list[dict[str, object]]] = {}

for dataset_id in DATASETS:
    df = dataset_labels[dataset_id]
    heatmaps_dir = RUN_DIR / "heatmaps" / dataset_id
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    target_dir = (
        keras_xai_root
        / "output"
        / "notebooks"
        / "create_heatmaps_for_right_thalamus_model"
        / dataset_id
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    mask_pipeline = AsegThalamusMaskPipeline(
        dataset_id=dataset_id,
        dataset_dir=DATASET_DIRS[dataset_id],
        heatmaps_dir=heatmaps_dir,
    )

    print(f"\n======== {dataset_id} ========")
    print(f"Subjekte LRP + Maske: n={len(df)}")
    print(f"NIfTI-Heatmaps → {heatmaps_dir}/<subject-id>/")
    print(
        f"Thalamus-Masken → {heatmaps_dir}/<subject-id>/"
        f"<subject-id>_aseg_thalamus_mask_mni152_cropped.nii.gz"
    )
    print(
        f"PNG-Vorschau → {target_dir} "
        f"(erste {min(N_HEATMAP_PLOTS, len(df))}, "
        f"inline={SHOW_HEATMAP_PLOTS_INLINE})"
    )

    preds_by_dataset[dataset_id] = []

    for i, (_, row) in enumerate(
        tqdm(df.iterrows(), total=len(df), desc=f"LRP+mask [{dataset_id}]")
    ):
        sid = str(row["participant_id"])
        path = str(row["filepath"])
        y_true = float(row[pred_var])

        if not Path(path).is_file():
            msg = f"[{dataset_id}/{sid}] cropped.nii.gz fehlt: {path}"
            print("FEHLER:", msg)
            mask_errors.append(msg)
            continue

        vol = load_volume(path)
        X = np.expand_dims(vol, axis=0)
        y_pred = float(np.squeeze(model.predict(X, verbose=0)))
        preds_by_dataset[dataset_id].append(
            {
                "subject_id": sid,
                pred_var: y_true,
                "prediction": y_pred,
            }
        )
        R = lrp(X)[0].numpy()
        R_masked = mask_explanation(vol, R)

        subject_dir = heatmaps_dir / sid
        subject_dir.mkdir(parents=True, exist_ok=True)
        nii_path = subject_dir / f"lrp_heatmap_{dataset_id}_{sid}.nii.gz"
        save_heatmap_nifti(R_masked, path, nii_path)
        saved_niftis.append(nii_path)

        print(
            f"[{dataset_id}] {sid}: true={y_true:.1f}  pred={y_pred:.1f}  "
            f"sum(R)={float(np.sum(R_masked)):.3f}  →  {nii_path}"
        )

        try:
            mask_path = mask_pipeline.run(sid)
            saved_masks.append(mask_path)
            print(f"[{dataset_id}] {sid}: Thalamus-Maske → {mask_path}")
        except (FileNotFoundError, RuntimeError) as exc:
            msg = f"[{dataset_id}/{sid}] Maske fehlgeschlagen: {exc}"
            print("FEHLER:", msg)
            mask_errors.append(msg)

        if i < N_HEATMAP_PLOTS:
            plot_lrp_heatmap(
                vol,
                R,
                title=(
                    f"dataset={dataset_id}  {sid}  "
                    f"true={y_true:.1f}  pred={y_pred:.1f}"
                ),
                save_path=target_dir / f"lrp_heatmap_{sid}.png",
                show_inline=SHOW_HEATMAP_PLOTS_INLINE,
            )

print(f"\nFertig. {len(saved_niftis)} NIfTI-Heatmaps, {len(saved_masks)} Thalamus-Masken.")
if mask_errors:
    print(f"Warnung: {len(mask_errors)} Masken-/Input-Fehler:")
    for m in mask_errors:
        print(" -", m)

# %% [markdown]
# ## Korrelation True vs. Pred (pro Dataset)
#
# Scatter-Plots für echte vs. vorhergesagte Thalamus-Volumen — jeweils für die konfigurierten `DATASETS` und `N_SUBJECTS`.

# %%
for dataset_id, rows in preds_by_dataset.items():
    if not rows:
        print(f"[{dataset_id}] Keine Vorhersagen für Scatter-Plot.")
        continue

    preds_df = pd.DataFrame(rows)
    print(f"\n[{dataset_id}] true vs. pred Thalamus-Volumen (n={len(preds_df)}, N_SUBJECTS={N_SUBJECTS}):")
    display(preds_df[["subject_id", pred_var, "prediction"]])

    y_true = preds_df[pred_var].astype(float).to_numpy()
    y_pred = preds_df["prediction"].astype(float).to_numpy()

    r_val, _ = pearsonr(y_true, y_pred)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    print(
        f"[{dataset_id}] n={len(preds_df)}   Pearson r={r_val:.4f}   MAE={mae:.1f}"
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="none")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel(f"true {pred_var}")
    ax.set_ylabel("prediction")
    ax.set_title(f"{dataset_id}  r={r_val:.3f}  MAE={mae:.1f}")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    scatter_path = RUN_DIR / f"scatter_true_vs_pred_{dataset_id}_n{len(preds_df)}.png"
    fig.savefig(scatter_path, dpi=120)
    print("Scatter:", scatter_path)

    target_dir = (
        keras_xai_root
        / "output"
        / "notebooks"
        / "create_heatmaps_for_right_thalamus_model"
        / dataset_id
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_dir / f"scatter_true_vs_pred_n{len(preds_df)}.png", dpi=120)

    display(fig)
    plt.close(fig)

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

# %matplotlib inline

mask_path = Path(
    "/mnt/users/andreasre/data/nn-trainings/mri/Right-Whole_thalamus/"
    "training_run_21h19m18s_20aug2026/heatmaps/ukb/5614724_20252_2_0/"
    #"5614724_20252_2_0_aseg_thalamus_mask_mni152_cropped.nii.gz"
    "aseg_mni152_right_thalamus_cropped.nii.gz"
)

img = nib.load(str(mask_path))
mask = np.asarray(img.get_fdata(), dtype=np.float32).squeeze()
mask = (mask > 0).astype(np.float32)

print("shape:", mask.shape)
print("voxels in mask:", int(mask.sum()))

# Schnitt durch das Zentrum der Maske
coords = np.argwhere(mask > 0)
if len(coords) == 0:
    raise ValueError("Maske ist leer.")
center = coords.mean(axis=0).astype(int)
cx, cy, cz = center

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle(mask_path.name, fontsize=10)

axes[0].imshow(np.rot90(mask[cx]), cmap="gray", vmin=0, vmax=1)
axes[0].set_title(f"sagittal (x={cx})")
axes[0].axis("off")

axes[1].imshow(np.rot90(mask[:, cy]), cmap="gray", vmin=0, vmax=1)
axes[1].set_title(f"koronal (y={cy})")
axes[1].axis("off")

axes[2].imshow(mask[:, :, cz], cmap="gray", vmin=0, vmax=1)
axes[2].set_title(f"axial (z={cz})")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# %%
img.get_fdata().min()

# %%
# Overlap: relevante LRP-Voxel ∩ rechter Thalamus (aseg label 49, gecroppt).
# Liest gespeicherte NIfTIs — kann unabhängig vom LRP-Loop erneut laufen.


def compute_lrp_thalamus_overlap(
    heatmap_path: Path,
    right_thalamus_path: Path,
    *,
    relevance_eps: float = LRP_RELEVANCE_EPS,
) -> dict[str, float | int | str]:
    """Overlap-Metriken zwischen LRP-Heatmap und rechter Thalamus-Maske."""
    if not heatmap_path.is_file():
        raise FileNotFoundError(f"LRP-Heatmap fehlt: {heatmap_path}")
    if not right_thalamus_path.is_file():
        raise FileNotFoundError(f"Thalamus-Maske fehlt: {right_thalamus_path}")

    heat = np.asarray(
        nib.load(str(heatmap_path)).get_fdata(), dtype=np.float32
    ).squeeze()
    mask = np.asarray(
        nib.load(str(right_thalamus_path)).get_fdata(), dtype=np.float32
    ).squeeze()
    if heat.shape != mask.shape:
        raise ValueError(
            f"Shape-Mismatch Heatmap {heat.shape} vs Maske {mask.shape} "
            f"({heatmap_path.name})"
        )

    thalamus = mask > 0
    relevant = np.abs(heat) > relevance_eps

    n_relevant = int(relevant.sum())
    n_thalamus = int(thalamus.sum())
    n_in_thalamus = int((relevant & thalamus).sum())
    n_outside = n_relevant - n_in_thalamus

    abs_in = float(np.abs(heat[relevant & thalamus]).sum()) if n_in_thalamus else 0.0
    abs_total = float(np.abs(heat[relevant]).sum()) if n_relevant else 0.0

    return {
        "subject_id": heatmap_path.parent.name,
        "heatmap_path": str(heatmap_path),
        "mask_path": str(right_thalamus_path),
        "n_relevant_voxels": n_relevant,
        "n_thalamus_voxels": n_thalamus,
        "n_relevant_in_thalamus": n_in_thalamus,
        "n_relevant_outside_thalamus": n_outside,
        "frac_relevant_in_thalamus": (
            n_in_thalamus / n_relevant if n_relevant > 0 else np.nan
        ),
        "frac_thalamus_with_relevant_lrp": (
            n_in_thalamus / n_thalamus if n_thalamus > 0 else np.nan
        ),
        "frac_abs_relevance_in_thalamus": (
            abs_in / abs_total if abs_total > 0 else np.nan
        ),
        "sum_abs_relevance_in_thalamus": abs_in,
        "sum_abs_relevance_total": abs_total,
    }


def collect_overlap_stats(
    dataset_id: str,
    df: pd.DataFrame,
    *,
    heatmaps_dir: Path,
    relevance_eps: float = LRP_RELEVANCE_EPS,
) -> pd.DataFrame:
    rows: list[dict] = []
    for _, row in df.iterrows():
        sid = str(row["participant_id"])
        subject_dir = heatmaps_dir / sid
        heatmap_path = subject_dir / f"lrp_heatmap_{dataset_id}_{sid}.nii.gz"
        mask_path = subject_dir / "aseg_mni152_right_thalamus_cropped.nii.gz"
        try:
            stats = compute_lrp_thalamus_overlap(
                heatmap_path, mask_path, relevance_eps=relevance_eps
            )
            stats["dataset_id"] = dataset_id
            rows.append(stats)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[{dataset_id}/{sid}] Overlap übersprungen: {exc}")
    return pd.DataFrame(rows)


def plot_dataset_thalamus_overlap(
    overlap_df: pd.DataFrame,
    *,
    dataset_id: str,
    save_path: Path | None = None,
    show_inline: bool = SHOW_OVERLAP_PLOTS_INLINE,
) -> None:
    """Pro Dataset: Anteil relevanter LRP-Voxel im rechten Thalamus je Subject."""
    if overlap_df.empty:
        print(f"[{dataset_id}] Keine Overlap-Daten zum Plotten.")
        return

    df = overlap_df.sort_values("frac_relevant_in_thalamus", ascending=True).copy()
    subjects = df["subject_id"].astype(str).tolist()
    x = np.arange(len(subjects))

    fig, axes = plt.subplots(1, 3, figsize=(18, max(4.5, 0.28 * len(subjects) + 2)))
    fig.suptitle(
        f"{dataset_id}: LRP-Relevanz vs. rechter Thalamus "
        f"(|LRP| > {LRP_RELEVANCE_EPS})",
        fontsize=12,
    )

    # 1) Anteil relevanter Voxel im Thalamus
    ax = axes[0]
    pct = 100.0 * df["frac_relevant_in_thalamus"].to_numpy(dtype=float)
    colors = plt.cm.YlOrRd(np.clip(pct / max(pct.max(), 1.0), 0.2, 1.0))
    ax.barh(x, pct, color=colors, edgecolor="0.3", linewidth=0.4)
    ax.set_yticks(x)
    ax.set_yticklabels(subjects, fontsize=8)
    ax.set_xlabel("% relevanter LRP-Voxel im rechten Thalamus")
    ax.set_title("Voxel-Anteil")
    med = float(np.nanmedian(pct))
    ax.axvline(med, color="0.2", ls="--", lw=1, label=f"Median {med:.1f}%")
    ax.legend(fontsize=8, loc="lower right")

    # 2) Absolute Voxel-Zahlen (in vs. außerhalb)
    ax = axes[1]
    in_vox = df["n_relevant_in_thalamus"].to_numpy(dtype=float)
    out_vox = df["n_relevant_outside_thalamus"].to_numpy(dtype=float)
    ax.barh(x, in_vox, color="#2ca02c", label="in Thalamus", edgecolor="0.3", lw=0.4)
    ax.barh(x, out_vox, left=in_vox, color="#bdbdbd", label="außerhalb", edgecolor="0.3", lw=0.4)
    ax.set_yticks(x)
    ax.set_yticklabels(subjects, fontsize=8)
    ax.set_xlabel("Anzahl relevanter Voxel")
    ax.set_title("Voxel-Anzahl")
    ax.legend(fontsize=8, loc="lower right")

    # 3) Anteil der |LRP|-Summe im Thalamus
    ax = axes[2]
    pct_abs = 100.0 * df["frac_abs_relevance_in_thalamus"].to_numpy(dtype=float)
    ax.barh(x, pct_abs, color="#1f77b4", edgecolor="0.3", linewidth=0.4)
    ax.set_yticks(x)
    ax.set_yticklabels(subjects, fontsize=8)
    ax.set_xlabel("% der |LRP|-Summe im rechten Thalamus")
    ax.set_title("Relevanz-Masse")
    med_abs = float(np.nanmedian(pct_abs))
    ax.axvline(med_abs, color="0.2", ls="--", lw=1, label=f"Median {med_abs:.1f}%")
    ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[{dataset_id}] Overlap-Plot → {save_path}")
    if show_inline:
        display(fig)
    plt.close(fig)


overlap_by_dataset: dict[str, pd.DataFrame] = {}

for dataset_id in DATASETS:
    df = dataset_labels[dataset_id]
    heatmaps_dir = RUN_DIR / "heatmaps" / dataset_id
    target_dir = (
        keras_xai_root
        / "output"
        / "notebooks"
        / "create_heatmaps_for_right_thalamus_model"
        / dataset_id
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    overlap_df = collect_overlap_stats(dataset_id, df, heatmaps_dir=heatmaps_dir)
    overlap_by_dataset[dataset_id] = overlap_df

    if overlap_df.empty:
        print(f"\n[{dataset_id}] Keine Overlap-Statistiken.")
        continue

    tsv_path = RUN_DIR / f"{dataset_id}_lrp_thalamus_overlap.tsv"
    overlap_df.to_csv(tsv_path, sep="\t", index=False)

    summary = overlap_df[
        [
            "subject_id",
            "n_relevant_voxels",
            "n_relevant_in_thalamus",
            "frac_relevant_in_thalamus",
            "frac_abs_relevance_in_thalamus",
        ]
    ].copy()
    summary["pct_in_thalamus"] = 100.0 * summary["frac_relevant_in_thalamus"]
    print(f"\n[{dataset_id}] Overlap (n={len(summary)}) → {tsv_path}")
    display(summary.sort_values("pct_in_thalamus", ascending=False).head(10))

    plot_dataset_thalamus_overlap(
        overlap_df,
        dataset_id=dataset_id,
        save_path=target_dir / "lrp_thalamus_overlap_by_subject.png",
        show_inline=SHOW_OVERLAP_PLOTS_INLINE,
    )

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

base = Path(
    "/mnt/users/andreasre/data/nn-trainings/mri/Right-Whole_thalamus/"
    "training_run_21h19m18s_20aug2026/heatmaps/ukb/5614724_20252_2_0"
)
heatmap_path = base / "lrp_heatmap_ukb_5614724_20252_2_0.nii.gz"
mask_path = base / "aseg_mni152_right_thalamus_cropped.nii.gz"

heat = np.asarray(nib.load(str(heatmap_path)).get_fdata(), dtype=np.float32).squeeze()
mask = np.asarray(nib.load(str(mask_path)).get_fdata(), dtype=np.float32).squeeze()
mask = (mask > 0).astype(np.float32)

assert heat.shape == mask.shape, (heat.shape, mask.shape)

# Schnitt durch das Masken-Zentrum
coords = np.argwhere(mask > 0)
cx, cy, cz = coords.mean(axis=0).astype(int)

# Heatmap auf [-1, 1] für die Farbskala (falls nötig)
vmax = float(np.nanmax(np.abs(heat))) or 1.0
heat_n = np.clip(heat / vmax, -1.0, 1.0)

slices = [
    (np.rot90(heat_n[cx]), np.rot90(mask[cx]), f"sagittal (x={cx})"),
    (np.rot90(heat_n[:, cy]), np.rot90(mask[:, cy]), f"koronal (y={cy})"),
    (heat_n[:, :, cz], mask[:, :, cz], f"axial (z={cz})"),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
fig.suptitle("LRP-Heatmap + rechter Thalamus (leicht transparent)", fontsize=11)

for ax, (h_slc, m_slc, title) in zip(axes, slices):
    im = ax.imshow(h_slc, cmap="seismic", vmin=-1, vmax=1)
    # Maske nur dort sichtbar, wo Maske==1; leicht transparent
    ax.imshow(
        np.ma.masked_where(m_slc == 0, m_slc),
        cmap="Greens",
        alpha=0.35,
        vmin=0,
        vmax=1,
    )
    ax.set_title(title)
    ax.axis("off")

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
cbar.set_label("LRP-Relevanz (normiert)")
plt.tight_layout()
plt.show()


# %% [markdown]
# # vergleich summer LRP relevanzen der heatmap vs thalamus volumen (truth und predicted) / eigentlich sollte relevanzkonservierung gelten von layer zu layer

# %%
# Erstes Subject je Dataset: true/pred Thalamus-Volumen + Summe der LRP-Heatmap.
import nibabel as nib

for dataset_id in DATASETS:
    df = dataset_labels[dataset_id]
    if df.empty:
        print(f"[{dataset_id}] Keine Subjects (N_SUBJECTS={N_SUBJECTS}).")
        continue

    row = df.iloc[0]
    sid = str(row["participant_id"])
    y_true = float(row[pred_var])

    y_pred: float | None = None
    for rec in preds_by_dataset.get(dataset_id, []):
        if str(rec["subject_id"]) == sid:
            y_pred = float(rec["prediction"])
            break
    if y_pred is None and preds_by_dataset.get(dataset_id):
        y_pred = float(preds_by_dataset[dataset_id][0]["prediction"])

    heatmap_path = (
        RUN_DIR
        / "heatmaps"
        / dataset_id
        / sid
        / f"lrp_heatmap_{dataset_id}_{sid}.nii.gz"
    )
    if heatmap_path.is_file():
        heat = np.asarray(nib.load(str(heatmap_path)).get_fdata(), dtype=np.float32).squeeze()
        heat_sum = float(np.sum(heat))
        heat_sum_abs = float(np.sum(abs(heat)))
    else:
        heat_sum = float("nan")
        heat_sum_abs = float("nan")
        print(f"[{dataset_id}] Warnung: Heatmap fehlt → {heatmap_path}")

    print(f"\n[{dataset_id}] erstes Subject: {sid} (N_SUBJECTS={N_SUBJECTS})")
    print(f"  true {pred_var}: {y_true:.1f}")
    if y_pred is not None:
        print(f"  prediction:    {y_pred:.1f}")
    else:
        print("  prediction:    (LRP-Zelle noch nicht ausgeführt?)")
    print(f"  sum(heatmap):  {heat_sum:.3f}")
    print(f"  sum(abs(heatmap)):  {heat_sum_abs:.3f}")

# %% [markdown]
# # liste aller relevanzen aller voxel

# %%
import nibabel as nib

for dataset_id in DATASETS:
    df = dataset_labels[dataset_id]
    if df.empty:
        print(f"[{dataset_id}] Keine Subjects.")
        continue

    sid = str(df.iloc[0]["participant_id"])
    heatmap_path = (
        RUN_DIR
        / "heatmaps"
        / dataset_id
        / sid
        / f"lrp_heatmap_{dataset_id}_{sid}.nii.gz"
    )
    if not heatmap_path.is_file():
        print(f"[{dataset_id}] Heatmap fehlt → {heatmap_path}")
        continue

    heat = np.asarray(nib.load(str(heatmap_path)).get_fdata(), dtype=np.float32).ravel()
    total = float(np.sum(heat))

    relevance_table = pd.DataFrame(
        {
            "voxel_nr": np.arange(1, len(heat) + 1, dtype=np.int64),
            "relevanz": heat,
            "anteil_pct": np.where(
                total != 0.0,
                100.0 * heat / total,
                0.0,
            ),
        }
    )

    print(
        f"\n[{dataset_id}] {sid}: Relevanz je Voxel "
        f"(n={len(relevance_table)}, sum={total:.6f})"
    )
    display(relevance_table)

    # optional: als TSV speichern (empfohlen bei ~5,7 Mio. Voxeln)
    tsv_path = RUN_DIR / f"relevance_voxels_{dataset_id}_{sid}.tsv"
    relevance_table.to_csv(tsv_path, sep="\t", index=False)
    print("gespeichert:", tsv_path)

# %%
import nibabel as nib

# Mehr Nachkommastellen in der Notebook-Anzeige
pd.set_option("display.float_format", lambda x: f"{x:.12f}")
pd.set_option("display.max_rows", 20)  # head + ... + tail

for dataset_id in DATASETS:
    df = dataset_labels[dataset_id]
    if df.empty:
        continue

    sid = str(df.iloc[0]["participant_id"])
    heatmap_path = (
        RUN_DIR / "heatmaps" / dataset_id / sid
        / f"lrp_heatmap_{dataset_id}_{sid}.nii.gz"
    )
    if not heatmap_path.is_file():
        print(f"[{dataset_id}] Heatmap fehlt → {heatmap_path}")
        continue

    heat = np.asarray(nib.load(str(heatmap_path)).get_fdata(), dtype=np.float32).ravel()
    total = float(np.sum(heat))

    relevance_table = pd.DataFrame(
        {
            "voxel_nr": np.arange(1, len(heat) + 1, dtype=np.int64),
            "relevanz": heat,
            "anteil_pct": np.where(total != 0.0, 100.0 * heat / total, 0.0),
        }
    )

    print(f"\n[{dataset_id}] {sid}: Relevanz je Voxel (n={len(relevance_table)}, sum={total:.12f})")
    display(relevance_table)

    # TSV mit voller Genauigkeit (nicht gerundet)
    tsv_path = RUN_DIR / f"relevance_voxels_{dataset_id}_{sid}.tsv"
    relevance_table.to_csv(tsv_path, sep="\t", index=False, float_format="%.12f")
    print("gespeichert:", tsv_path)

# %%
heat = np.asarray(nib.load(str(heatmap_path)).get_fdata(), dtype=np.float32).ravel()

print("sum:              ", float(np.sum(heat)))
print("!= 0:             ", int(np.count_nonzero(heat)))
print("== 0:             ", int(np.sum(heat == 0)))
print("max:              ", float(np.max(heat)))
print("min (non-zero):   ", float(heat[heat != 0].min()) if np.any(heat != 0) else None)
print("mean (non-zero):  ", float(heat[heat != 0].mean()) if np.any(heat != 0) else None)

# %%
import nibabel as nib

pd.set_option("display.float_format", lambda x: f"{x:.12f}")
pd.set_option("display.max_rows", 110)

for dataset_id in DATASETS:
    df = dataset_labels[dataset_id]
    if df.empty:
        print(f"[{dataset_id}] Keine Subjects.")
        continue

    sid = str(df.iloc[0]["participant_id"])
    heatmap_path = (
        RUN_DIR
        / "heatmaps"
        / dataset_id
        / sid
        / f"lrp_heatmap_{dataset_id}_{sid}.nii.gz"
    )
    if not heatmap_path.is_file():
        print(f"[{dataset_id}] Heatmap fehlt → {heatmap_path}")
        continue

    heat = np.asarray(nib.load(str(heatmap_path)).get_fdata(), dtype=np.float32).ravel()
    total_abs = float(np.sum(np.abs(heat)))  # nur für Prozent-Nenner

    relevance_table = pd.DataFrame(
        {
            "voxel_nr": np.arange(1, len(heat) + 1, dtype=np.int64),
            "relevanz": heat,  # ← Vorzeichen bleibt (+/-)
            "anteil_pct": np.where(  # ← nur hier abs()
                total_abs != 0.0,
                100.0 * np.abs(heat) / total_abs,
                0.0,
            ),
        }
    )

    relevance_table = relevance_table.sort_values(
        "anteil_pct", ascending=False
    ).reset_index(drop=True)
    relevance_table.insert(0, "rang", np.arange(1, len(relevance_table) + 1))

    print(
        f"\n[{dataset_id}] {sid}: sortiert nach anteil_pct "
        f"(n={len(relevance_table)}, sum={float(np.sum(heat)):.12f}, "
        f"sum|R|={total_abs:.12f})"
    )

    n_show = 50
    preview = pd.concat(
        [relevance_table.head(n_show), relevance_table.tail(n_show)],
        ignore_index=True,
    )

    display(
        preview.style.format(
            {
                "relevanz": "{:.12f}",      # z.B. -0.003 oder +0.003
                "anteil_pct": "{:.4f}%",    # immer >= 0, z.B. 5.0000%
            }
        )
    )
    print(f"... {len(relevance_table) - 2 * n_show} Zeilen dazwischen ausgeblendet ...")

    tsv_path = RUN_DIR / f"relevance_voxels_sorted_{dataset_id}_{sid}.tsv"
    relevance_table.to_csv(tsv_path, sep="\t", index=False, float_format="%.12f")
    print("gespeichert:", tsv_path)

# %% [markdown]
# # anteil der summen der relevanz voxel innerhalb des thalamus am prädiziertem gesamtthalamus

# %%
import nibabel as nib

def _load_nii(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32).squeeze()


def _pred_for_subject(dataset_id: str, sid: str) -> float | None:
    for rec in preds_by_dataset.get(dataset_id, []):
        if str(rec["subject_id"]) == sid:
            return float(rec["prediction"])
    return None


def _fmt_pred(x: float | None) -> str:
    return f"{x:.1f}" if x is not None else "nan"


thalamus_relevance_rows: list[dict[str, object]] = []

for dataset_id in DATASETS:
    df = dataset_labels[dataset_id]

    print(
        f"\n======== {dataset_id}: Relevanz in linker/rechter Thalamus "
        f"(n={len(df)}) ========"
    )

    for _, row in df.iterrows():
        sid = str(row["participant_id"])
        y_true = float(row[pred_var])
        y_pred = _pred_for_subject(dataset_id, sid)

        subject_dir = RUN_DIR / "heatmaps" / dataset_id / sid
        heatmap_path = subject_dir / f"lrp_heatmap_{dataset_id}_{sid}.nii.gz"
        left_mask_path = subject_dir / "aseg_mni152_left_thalamus_cropped.nii.gz"
        right_mask_path = subject_dir / "aseg_mni152_right_thalamus_cropped.nii.gz"

        if not heatmap_path.is_file():
            print(f"[{dataset_id}/{sid}] Heatmap fehlt → {heatmap_path}")
            continue
        if not left_mask_path.is_file() or not right_mask_path.is_file():
            print(
                f"[{dataset_id}/{sid}] Thalamus-Maske fehlt "
                f"(left={left_mask_path.is_file()}, right={right_mask_path.is_file()})"
            )
            continue

        heat = _load_nii(heatmap_path)
        left_mask = _load_nii(left_mask_path) > 0
        right_mask = _load_nii(right_mask_path) > 0

        if heat.shape != left_mask.shape or heat.shape != right_mask.shape:
            print(f"[{dataset_id}/{sid}] Shape-Mismatch: heat={heat.shape}")
            continue

        sum_total = float(np.sum(heat))
        sum_abs_total = float(np.sum(np.abs(heat)))

        sum_left = float(np.sum(heat[left_mask]))
        sum_right = float(np.sum(heat[right_mask]))
        sum_abs_left = float(np.sum(np.abs(heat[left_mask])))
        sum_abs_right = float(np.sum(np.abs(heat[right_mask])))
        sum_abs_outside = sum_abs_total - sum_abs_left - sum_abs_right

        pct_abs_left = 100.0 * sum_abs_left / sum_abs_total if sum_abs_total > 0 else np.nan
        pct_abs_right = 100.0 * sum_abs_right / sum_abs_total if sum_abs_total > 0 else np.nan
        pct_abs_outside = 100.0 * sum_abs_outside / sum_abs_total if sum_abs_total > 0 else np.nan

        pct_of_pred_right = 100.0 * sum_right / y_pred if y_pred not in (None, 0.0) else np.nan
        pct_of_pred_left = 100.0 * sum_left / y_pred if y_pred not in (None, 0.0) else np.nan

        print(
            f"[{dataset_id}/{sid}]  "
            f"true={y_true:.1f}  pred={_fmt_pred(y_pred)}\n"
            f"  sum(R) gesamt:             {sum_total:+.6e}\n"
            f"  sum(R) linker Thalamus:    {sum_left:+.6e}   "
            f"({pct_abs_left:.4f}% von |R| gesamt)\n"
            f"  sum(R) rechter Thalamus:   {sum_right:+.6e}   "
            f"({pct_abs_right:.4f}% von |R| gesamt)\n"
            f"  sum(|R|) außerhalb beider: {sum_abs_outside:.6e}   "
            f"({pct_abs_outside:.4f}% von |R| gesamt)\n"
            f"  sum(R_right) / prediction: {pct_of_pred_right:.4f}%\n"
            f"  sum(R_left)  / prediction: {pct_of_pred_left:.4f}%"
        )

        thalamus_relevance_rows.append(
            {
                "dataset_id": dataset_id,
                "subject_id": sid,
                "true_volume": y_true,
                "prediction": y_pred,
                "sum_R_total": sum_total,
                "sum_abs_R_total": sum_abs_total,
                "sum_R_left_thalamus": sum_left,
                "sum_R_right_thalamus": sum_right,
                "pct_abs_left_of_total_R": pct_abs_left,
                "pct_abs_right_of_total_R": pct_abs_right,
                "pct_abs_outside_both_of_total_R": pct_abs_outside,
                "pct_sum_R_right_of_prediction": pct_of_pred_right,
                "pct_sum_R_left_of_prediction": pct_of_pred_left,
            }
        )

thalamus_relevance_df = pd.DataFrame(thalamus_relevance_rows)
display(thalamus_relevance_df)

tsv_path = RUN_DIR / "lrp_relevance_left_right_thalamus_by_subject.tsv"
thalamus_relevance_df.to_csv(tsv_path, sep="\t", index=False, float_format="%.12e")
print("\ngespeichert:", tsv_path)

# %%
