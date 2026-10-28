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
# # LRP mit drei Referenzwerten: `Right-Whole_thalamus` (UKB)
#
# Nur **Modellfall (1)** aus
# `check_LRP_heatmaps_for_3_different_right_thalamus_ukb_models.ipynb`:
# SFCN auf **unveränderten** MRI-Scans (`cropped.nii.gz`), Label
# `Right-Whole_thalamus` (Volumen des rechten Thalamus).
#
# Vergleich dreier LRP-Heatmaps für dasselbe Subject:
#
# 1. **klassisch** — impliziter Referenzwert $\widetilde{y}=0$ im Netzraum
# 2. **Letzgus et al.** mit $\widetilde{y}=$ **Mittelwert der Trainingslabels**
#    → „warum weicht dieses Subject vom Trainingsmittel ab?“
# 3. **Letzgus et al.** mit $\widetilde{y}=$ **Groundtruth** des Subjects
#    → „warum weicht die Vorhersage vom wahren Wert ab?“
#    (Umstrukturierung nach Abschn. IV-B, arXiv:2112.11407)
#
# Ablauf: Vorhersagen für `N_SUBJ_PRED` Holdout-Subjects → Scatter/MAE/r →
# Tabelle → für Subject `IDX_PRED` Input mit Thalamus-Maske und die drei
# unnormierten LRP-Heatmaps, beschriftet mit Subject-ID und $\widetilde{y}$.
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
# - `N_SUBJ_PRED`: Anzahl Holdout-Subjects (erste Zeilen der `predict.tsv`).
# - `IDX_PRED`: Index des Subjects für die Heatmaps. Muss
#   `0 <= IDX_PRED < N_SUBJ_PRED` erfüllen.
# - `REFERENCE_ASYMMETRY`: $\kappa$ der Flooding-Regel (0 = einseitig,
#   0.25 = Variante aus Abschn. VI-A des Papers, robuster wenn
#   $f(x)<\widetilde{y}$).
#

# %%
N_SUBJ_PRED = 3
IDX_PRED = 1
REFERENCE_ASYMMETRY = 0.25

PRED_BATCH_SIZE = 8
SAGITTAL_X = 70
SHOW_PLOTS_INLINE = True
COLOR_RIGHT = (0.15, 0.65, 0.25, 0.50)  # grün, alpha=0.5
PRED_VAR = "Right-Whole_thalamus"

if not (0 <= int(IDX_PRED) < int(N_SUBJ_PRED)):
    raise ValueError(
        f"IDX_PRED={IDX_PRED} muss kleiner als N_SUBJ_PRED={N_SUBJ_PRED} sein "
        f"(gültig: 0 .. {N_SUBJ_PRED - 1})."
    )

# (1) unveränderte MRI (dataset ukb, original scans)
RUN_DIR = Path(
    "/mnt/ceph2/dl_project/data/nn-trainings/mri/"
    "Right-Whole_thalamus/"
    "training_run_21h19m18s_20aug2026"
).resolve()
PREDICT_TSV = Path(
    "/mnt/users/andreasre/git-repos/pyment-and1/training_runs/input_files/mri/"
    "right_whole_thalamus/volume/original_scans/predict.tsv"
).resolve()
TRAIN_TSV = Path(
    "/mnt/users/andreasre/git-repos/pyment-and1/training_runs/input_files/mri/"
    "right_whole_thalamus/volume/original_scans/train.tsv"
).resolve()
MASK_TMPL = (
    "/mnt/ceph2/dl_project/data/mri-scans/not_altered/ukb/recon/"
    "{sid}/mri/aseg_mni152_right_thalamus_cropped.nii.gz"
)
MASK_FALLBACKS = [
    "/mnt/ceph2/dl_project/data/mri-scans/not_altered/ukb/recon/"
    "{sid}/mri/aseg_mni152_right_thalamus.nii.gz",
]

for p, name in (
    (RUN_DIR / "model.keras", "model.keras"),
    (RUN_DIR / "config.yaml", "config.yaml"),
    (PREDICT_TSV, "predict.tsv"),
    (TRAIN_TSV, "train.tsv"),
):
    if not Path(p).is_file():
        raise FileNotFoundError(f"{name} fehlt: {p}")

print(f"N_SUBJ_PRED          = {N_SUBJ_PRED}")
print(f"IDX_PRED             = {IDX_PRED}")
print(f"REFERENCE_ASYMMETRY  = {REFERENCE_ASYMMETRY}")
print(f"PRED_VAR             = {PRED_VAR}")
print(f"run                  = {RUN_DIR.name}")
print(f"predict.tsv          = {PREDICT_TSV}")
print(f"train.tsv            = {TRAIN_TSV}")


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
from explainability.model.utils.reference import find_linear_dense_head  # noqa: E402

print("pyment-and1:  ", PYMENT_AND1_ROOT)
print("keras-xai:     ", keras_xai_root)
print("pybrainmetrics:", pybm_src)
print("sys.path[0:2]: ", sys.path[:2])
print("TensorFlow:    ", tf.__version__)


# %% [markdown]
# ## D. Labels laden, Subject-Pool und Trainings-Referenzwert
#
# Holdout-Labels kommen aus `predict.tsv`. Der Referenzwert $\widetilde{y}$
# ist der Mittelwert der **wahren** `Right-Whole_thalamus`-Volumina
# in der **Trainings**-TSV — nicht die Vorhersage und nicht der Holdout-Wert
# des einzelnen Subjects.
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


def load_label_tsv(tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t")
    return _normalize_label_columns(df, PRED_VAR)


labels_pred = load_label_tsv(PREDICT_TSV)
labels_train = load_label_tsv(TRAIN_TSV)
y_ref_orig = float(labels_train[PRED_VAR].mean())

subject_ids = labels_pred["participant_id"].astype(str).head(int(N_SUBJ_PRED)).tolist()
lookup = {str(row["participant_id"]): row for _, row in labels_pred.iterrows()}
subject_id_plot = subject_ids[int(IDX_PRED)]

print(f"Trainings-n              = {len(labels_train)}")
print(f"y_ref (Trainingsmittel)  = {y_ref_orig:.6g}   ← Referenzwert auf der Label-Skala")
print(f"Subject-Pool (n={len(subject_ids)}): {subject_ids}")
print(f"Plot-Subject IDX_PRED={IDX_PRED}: {subject_id_plot}")


# %% [markdown]
# ## E. Modell, Volume-Loader und LRP-Strategie
#
# Das `model.keras` von Lauf (1) wird geladen. Der NIfTI-Loader folgt
# `config.yaml` (`nifti-native`). LRP-Composite für SFCN wie im
# Volumen-Notebook: zwei `flat`-Schichten, vier αβ-Schichten (α=1, β=0),
# ε am Dense-Ausgang.
#
# Ist `training.normalisation.use: true` mit `z_score`, liegen Netzausgabe
# und damit auch $\widetilde{y}$ im z-Raum. Dann gilt
# $\widetilde{y}_{\mathrm{Netz}} = (\bar{y}_{\mathrm{Train}}-\mu)/\sigma$.
# Heatmaps werden hinterher mit $\sigma$ auf die Volumenskala
# zurückskaliert (`R · σ`), analog zum alten Notebook.
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
        {"alpha": 1, "beta": 0},
        {"alpha": 1, "beta": 0},
        {"alpha": 1, "beta": 0},
        {"alpha": 1, "beta": 0},
        {"epsilon": 0.25},
    ],
)


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


def to_network_space(y_orig: float, params: dict[str, float | bool]) -> float:
    """Map a label-scale value into the space of the network output."""
    if not params["use"]:
        return float(y_orig)
    return (float(y_orig) - float(params["mean"])) / float(params["std"])


model, cfg, delta = load_keras_model(RUN_DIR)
load_vol = make_load_volume(cfg)
zscore_params = zscore_inverse_params(RUN_DIR)
LAYER_IDX = find_linear_dense_head(model)
y_ref_net = to_network_space(y_ref_orig, zscore_params)

print(
    f"Modell geladen  Δw={delta:.3g}  loader={cfg.data.loader}  "
    f"label={cfg.data.prediction_variable}"
)
print(f"Dense-Kopf      layer={LAYER_IDX}  name={model.layers[LAYER_IDX].name}")
if zscore_params["use"]:
    print(
        f"z-score aktiv   μ={zscore_params['mean']:.6g}  "
        f"σ={zscore_params['std']:.6g}"
    )
    print(
        f"y_ref Netzraum  = {y_ref_net:.6g}  "
        f"((Trainingsmittel − μ) / σ)"
    )
    if abs(y_ref_net) < 1e-3:
        print(
            "Hinweis: Bei z-Score-Labels ist das Trainingsmittel im Netzraum "
            "≈ 0, also derselbe Output-Offset wie bei klassischem LRP. "
            "Die Heatmaps unterscheiden sich trotzdem, weil Letzgus et al. "
            "die letzte versteckte Schicht umstrukturieren (Flooding)."
        )
else:
    print("keine Label-z-score-Normalisierung")
    print(f"y_ref Netzraum  = {y_ref_net:.6g}  (identisch zur Label-Skala)")


# %% [markdown]
# ## F. Vorhersagen für `N_SUBJ_PRED` Subjects
#
# Das Modell sagt für die Holdout-Subjects das Volumen des rechten
# Thalamus voraus. Wahre Werte kommen aus `predict.tsv`
# (Originalskala). Die Netzausgabe wird bei aktivem z-Score mit
# $y = z\cdot\sigma + \mu$ zurückgerechnet.
#

# %%
def predict_subjects(sids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sid in sids:
        row = lookup[sid]
        path = Path(str(row["filepath"]))
        y_true = float(row[PRED_VAR])
        if not path.is_file():
            raise FileNotFoundError(f"[{sid}] Volume fehlt: {path}")
        vol = load_vol(str(path))
        y_pred_net = float(np.squeeze(model.predict(np.expand_dims(vol, 0), verbose=0)))
        y_pred = float(np.squeeze(inverse_zscore_values(y_pred_net, zscore_params)))
        rows.append(
            {
                "subject_id": sid,
                "filepath": str(path),
                "y_true": y_true,
                "y_pred": y_pred,
                "y_pred_net": y_pred_net,
                "y_ref_orig": y_ref_orig,
                "delta_vs_ref": y_pred - y_ref_orig,
            }
        )
    return pd.DataFrame(rows)


preds_df = predict_subjects(subject_ids)
y_true_all = preds_df["y_true"].to_numpy(dtype=float)
y_pred_all = preds_df["y_pred"].to_numpy(dtype=float)
r_val = float(pearsonr(y_true_all, y_pred_all)[0]) if len(preds_df) >= 2 else float("nan")
mae = float(np.mean(np.abs(y_true_all - y_pred_all)))
print(f"n={len(preds_df)}  MAE={mae:.4f}  r={r_val:.3f}")
print(f"y_ref (Trainingsmittel) = {y_ref_orig:.6g}")
display(preds_df.round(6))


# %% [markdown]
# ## G. Scatter-Plot
#
# Jeder Punkt ist ein Holdout-Subject. Subject `IDX_PRED` ist rot umrandet.
# Die Diagonale ist die ideale Vorhersage. Die horizontale Linie markiert
# den Trainingsmittel-Referenzwert.
#

# %%
fig, ax = plt.subplots(figsize=(5.2, 5.0))
ax.scatter(y_true_all, y_pred_all, alpha=0.85, s=55, label="Holdout")
lo = float(min(y_true_all.min(), y_pred_all.min(), y_ref_orig))
hi = float(max(y_true_all.max(), y_pred_all.max(), y_ref_orig))
pad = 0.05 * (hi - lo + 1e-6)
ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="Identität")
ax.axhline(
    y_ref_orig,
    color="C1",
    ls=":",
    lw=1.4,
    label=f"y_ref (Mittelwert Trainingslabels) = {y_ref_orig:.4f}",
)
ax.axvline(y_ref_orig, color="C1", ls=":", lw=1.4)
ax.scatter(
    [y_true_all[IDX_PRED]],
    [y_pred_all[IDX_PRED]],
    s=140,
    facecolors="none",
    edgecolors="C3",
    linewidths=2,
    label=f"IDX_PRED={IDX_PRED}  {subject_id_plot}",
)
ax.set_xlabel("wahres Volumen")
ax.set_ylabel("prädiziertes Volumen")
ax.set_title(
    f"Wahr vs. prädiziert  ·  {PRED_VAR}\n"
    f"n={N_SUBJ_PRED}  MAE={mae:.4f}  r={r_val:.3f}  ·  "
    f"y_ref (Mittelwert Trainingslabels) = {y_ref_orig:.4f}"
)
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="best", fontsize=8)
fig.tight_layout()
if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)


# %% [markdown]
# ## H. Input-Bild und Thalamus-Maske für `IDX_PRED`
#
# Sagittaler Schnitt `x=70` des Input-Volumes aus der `predict.tsv`.
# Die rechte Thalamus-Maske wird als grüne Umrandung gezeichnet.
#

# %%
def resolve_mask_path(sid: str) -> Path:
    candidates = [Path(MASK_TMPL.format(sid=sid))]
    candidates.extend(Path(t.format(sid=sid)) for t in MASK_FALLBACKS)
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"Keine Thalamus-Maske für [{sid}]. Geprüft:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32).squeeze()


def sagittal_slc(vol: np.ndarray, cx: int) -> np.ndarray:
    return np.rot90(vol[cx])


row_plot = lookup[subject_id_plot]
vol_path = Path(str(row_plot["filepath"]))
mask_path = resolve_mask_path(subject_id_plot)
if not vol_path.is_file():
    raise FileNotFoundError(vol_path)
plot_vol = load_nifti(vol_path)
plot_mask = load_nifti(mask_path)
if plot_vol.shape != plot_mask.shape:
    raise ValueError(
        f"Shape-Mismatch [{subject_id_plot}]: "
        f"vol {plot_vol.shape} vs mask {plot_mask.shape}"
    )
plot_right = plot_mask > 0
pred_row = preds_df.set_index("subject_id").loc[subject_id_plot]
plot_true = float(pred_row["y_true"])
plot_pred = float(pred_row["y_pred"])
cx = int(np.clip(SAGITTAL_X, 0, plot_vol.shape[0] - 1))

print(f"volume: {vol_path}")
print(f"mask:   {mask_path}")
print(
    f"subject={subject_id_plot}  true={plot_true:.4f}  pred={plot_pred:.4f}  "
    f"y_ref={y_ref_orig:.4f}  shape={plot_vol.shape}  sagittal x={cx}"
)


# %% [markdown]
# ## I. Drei LRP-Heatmaps neu berechnen (unnormiert)
#
# Für Subject `IDX_PRED` werden drei Heatmaps bei jedem Lauf neu berechnet.
#
# - **klassisch:** `LRP(..., reference_value=None)` — implizit $\widetilde{y}=0$
#   im Netzraum.
# - **Letzgus, Trainingsmittel:** `reference_value=y_ref_net` mit
#   $\widetilde{y}=$ Mittelwert der Trainingslabels.
# - **Letzgus, Groundtruth:** `reference_value` = wahrer Labelwert des Subjects
#   (Netzraum) — erklärt $f(x)-y_{\mathrm{true}}$.
#
# War `normalisation.use: true`, wird jede Heatmap mit dem Trainings-σ
# multipliziert (`R · σ`). μ wird nicht voxelweise addiert. Danach gilt
# näherungsweise $\sum R_{\mathrm{klassisch}}\cdot\sigma \approx y_{\mathrm{pred}}-\mu$
# und $\sum R_{\mathrm{ref}}\cdot\sigma \approx y_{\mathrm{pred}}-\widetilde{y}$.
#

# %%
def run_lrp(vol: np.ndarray, *, reference_value: float | None) -> np.ndarray:
    kwargs = dict(
        model=model,
        layer=LAYER_IDX,
        idx=0,
        strategy=LRP_STRATEGY,
    )
    if reference_value is not None:
        kwargs["reference_value"] = float(reference_value)
        kwargs["reference_asymmetry"] = float(REFERENCE_ASYMMETRY)
    lrp = LRP(**kwargs)
    R = np.asarray(lrp(np.expand_dims(vol, 0))[0].numpy(), dtype=np.float64).squeeze()
    if zscore_params["use"]:
        R = R * float(zscore_params["std"])
    return np.asarray(R, dtype=np.float32)


y_ref_gt_orig = float(plot_true)
y_ref_gt_net = to_network_space(y_ref_gt_orig, zscore_params)

vol_net = load_vol(str(vol_path))
t0 = time.perf_counter()
heat_classic = run_lrp(vol_net, reference_value=None)
t_classic = time.perf_counter() - t0
t1 = time.perf_counter()
heat_ref_train = run_lrp(vol_net, reference_value=y_ref_net)
t_ref_train = time.perf_counter() - t1
t2 = time.perf_counter()
heat_ref_gt = run_lrp(vol_net, reference_value=y_ref_gt_net)
t_ref_gt = time.perf_counter() - t2

heatmaps = {
    "classic": {
        "R": heat_classic,
        "y_ref_orig": 0.0,
        "y_ref_net": 0.0,
        "label": "klassisch (y_ref = 0, implizit)",
        "seconds": t_classic,
    },
    "paper_train": {
        "R": heat_ref_train,
        "y_ref_orig": y_ref_orig,
        "y_ref_net": y_ref_net,
        "label": (
            f"Letzgus et al.  ·  y_ref (Mittelwert Trainingslabels) "
            f"= {y_ref_orig:.4f}"
        ),
        "seconds": t_ref_train,
    },
    "paper_gt": {
        "R": heat_ref_gt,
        "y_ref_orig": y_ref_gt_orig,
        "y_ref_net": y_ref_gt_net,
        "label": (
            f"Letzgus et al.  ·  y_ref (Groundtruth) "
            f"= {y_ref_gt_orig:.4f}"
        ),
        "seconds": t_ref_gt,
    },
}

for key, meta in heatmaps.items():
    R = meta["R"]
    sum_R = float(np.sum(R))
    sum_R_thal = float(np.sum(R[plot_right]))
    pct_thal = 100.0 * sum_R_thal / sum_R if abs(sum_R) > 0 else float("nan")
    meta["sum_R"] = sum_R
    meta["sum_R_thal"] = sum_R_thal
    meta["pct_thal"] = pct_thal
    scale_note = (
        f"  · R·σ (σ={float(zscore_params['std']):.6g})"
        if zscore_params["use"]
        else "  · keine z-score-Skalierung"
    )
    print(
        f"[{key}] {meta['label']}\n"
        f"        shape={R.shape}  |R|_max={float(np.nanmax(np.abs(R))):.4g}  "
        f"ΣR={sum_R:.4g}  Thalamus={pct_thal:.1f}%  "
        f"Dauer={meta['seconds']:.2f} s{scale_note}"
    )

t_all = t_classic + t_ref_train + t_ref_gt
print(f"\nDauer alle Heatmaps: {t_all:.2f} s ({t_all / 60.0:.2f} min)")


# %% [markdown]
# ## J. Figur: 3 Zeilen × 2 Spalten
#
# - Zeile 1: Input-MRT | klassisch ($\widetilde{y}=0$)
# - Zeile 2: Letzgus, Trainingsmittel | Letzgus, Groundtruth
# - Zeile 3: Diff (Trainingsmittel − 0) | Diff (Groundtruth − 0)
#

# %%
COLOR_OUTLINE = COLOR_RIGHT[:3]
pos = plot_vol[plot_vol > 0]
vmax_i = float(np.percentile(pos, 99.5)) if pos.size else 1.0
r_slc = sagittal_slc(plot_right.astype(np.float32), cx) > 0

fig, axes = plt.subplots(3, 2, figsize=(12.0, 15.5))
fig.suptitle(
    (
        f"Subject {subject_id_plot}  ·  IDX_PRED={IDX_PRED}/{N_SUBJ_PRED}  ·  "
        f"sagittal x={cx}  ·  {PRED_VAR}\n"
        f"wahr={plot_true:.4f}  präd={plot_pred:.4f}  ·  "
        f"y_ref (Mittelwert Trainingslabels) = {y_ref_orig:.4f}  ·  "
        f"y_ref (Groundtruth) = {y_ref_gt_orig:.4f}"
    ),
    fontsize=11,
)


def _draw_contour(ax):
    if np.any(r_slc):
        ax.contour(
            r_slc.astype(np.float32),
            levels=[0.5],
            colors=[COLOR_OUTLINE],
            linewidths=1.5,
        )


def _draw_lrp_panel(ax, key: str):
    meta = heatmaps[key]
    heat = meta["R"]
    abs_max = float(np.nanmax(np.abs(heat))) or 1.0
    vmax_r = float(np.nanpercentile(np.abs(heat), 99.5)) or abs_max
    im = ax.imshow(
        sagittal_slc(heat, cx),
        cmap="RdBu_r",
        vmin=-vmax_r,
        vmax=vmax_r,
    )
    _draw_contour(ax)
    ax.set_title(
        f"LRP  ·  {meta['label']}\n"
        f"Subject {subject_id_plot}\n"
        f"wahr={plot_true:.4f}  präd={plot_pred:.4f}  "
        f"|R|_max={abs_max:.3g}\n"
        f"Thalamus = {meta['pct_thal']:.1f}% der Gesamtrelevanz  ·  "
        f"ΣR={meta['sum_R']:.3g}",
        fontsize=8.5,
    )
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("LRP-Relevanz")
    print(
        f"[{key}] Thalamus-Anteil = {meta['pct_thal']:.2f}%  "
        f"(ΣR_Thalamus={meta['sum_R_thal']:.4g} / ΣR={meta['sum_R']:.4g})  "
        f"y_ref={meta['y_ref_orig']:.6g}"
    )


def _draw_diff_panel(ax, diff: np.ndarray, title_prefix: str):
    abs_max_d = float(np.nanmax(np.abs(diff))) or 1.0
    vmax_d = float(np.nanpercentile(np.abs(diff), 99.5)) or abs_max_d
    im_d = ax.imshow(
        sagittal_slc(diff, cx),
        cmap="RdBu_r",
        vmin=-vmax_d,
        vmax=vmax_d,
    )
    _draw_contour(ax)
    ax.set_title(
        f"{title_prefix}\n"
        f"Subject {subject_id_plot}\n"
        f"|ΔR|_max={abs_max_d:.3g}  ΣΔR={float(np.sum(diff)):.3g}",
        fontsize=8.5,
    )
    ax.axis("off")
    cbar = fig.colorbar(im_d, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δ LRP-Relevanz")


# Zeile 1: Input | klassisch (y_ref = 0)
ax = axes[0, 0]
im0 = ax.imshow(sagittal_slc(plot_vol, cx), cmap="gray", vmin=0.0, vmax=vmax_i)
_draw_contour(ax)
ax.set_title(
    f"Input  ·  Subject {subject_id_plot}\n"
    f"wahr={plot_true:.4f}  präd={plot_pred:.4f}",
    fontsize=9,
)
ax.axis("off")
cbar = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("MRI-Intensität")

_draw_lrp_panel(axes[0, 1], "classic")

# Zeile 2: Trainingsmittel | Groundtruth
_draw_lrp_panel(axes[1, 0], "paper_train")
_draw_lrp_panel(axes[1, 1], "paper_gt")

# Zeile 3: Differenzen jeweils minus y_ref = 0
_draw_diff_panel(
    axes[2, 0],
    heatmaps["paper_train"]["R"] - heatmaps["classic"]["R"],
    (
        f"Differenz  ·  y_ref (Mittelwert Trainingslabels) "
        f"= {y_ref_orig:.4f}  −  y_ref = 0"
    ),
)
_draw_diff_panel(
    axes[2, 1],
    heatmaps["paper_gt"]["R"] - heatmaps["classic"]["R"],
    (
        f"Differenz  ·  y_ref (Groundtruth) "
        f"= {y_ref_gt_orig:.4f}  −  y_ref = 0"
    ),
)

fig.legend(
    handles=[
        Patch(
            facecolor="none",
            edgecolor=COLOR_OUTLINE,
            linewidth=1.5,
            label="rechter Thalamus",
        )
    ],
    loc="lower center",
    ncol=1,
    frameon=False,
    fontsize=9,
)
fig.tight_layout(rect=[0, 0.03, 1, 0.93])

if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)

print(
    f"Heatmap-Berechnung: klassisch {t_classic:.2f} s, "
    f"y_ref (Trainingsmittel) {t_ref_train:.2f} s, "
    f"y_ref (Groundtruth) {t_ref_gt:.2f} s  "
    f"(Subject {subject_id_plot})."
)
