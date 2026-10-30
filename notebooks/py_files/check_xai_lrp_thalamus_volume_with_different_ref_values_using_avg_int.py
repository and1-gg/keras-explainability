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
# # LRP mit drei Referenzwerten: `average-intensity_right-whole-thalamus` (UKB)
#
# Nur **Modellfall (1)** aus
# `check_LRP_heatmaps_for_average_intensity_right_thalamus_ukb_models.ipynb`:
# SFCN auf **unveränderten** MRI-Scans (`cropped.nii.gz`), Label
# `average-intensity_right-whole-thalamus`.
#
# Vergleich von LRP-Heatmaps für dasselbe Subject mit verschiedenen
# $\widetilde{y}$:
#
# 1. **klassisch** wie im alten Notebook — impliziter Referenzwert
#    $\widetilde{y}=0$ im Netzraum (Letzgus et al. 2022, arXiv:2112.11407)
# 2. **Umstrukturierung** nach Letzgus et al., Abschn. IV-B — das Netz wird
#    ohne Neutraining so umgeschrieben, dass $g(x)=f(x)-\widetilde{y}$ erklärt
#    wird. Als $\widetilde{y}$ dient der **Mittelwert der wahren Labels** in
#    der Trainings-TSV (Populationstypische mittlere Thalamus-Intensität).
# 3. **Ground Truth** — dieselbe Umstrukturierung, aber mit
#    $\widetilde{y}=y_{\mathrm{true}}$ des Plot-Subjects. Die Erklärung
#    beantwortet dann „warum weicht die Vorhersage vom wahren Label ab?“
#    (Residuum $f(x)-y_{\mathrm{true}}$).
#
# Referenz (2) hat dieselbe Einheit wie die Vorhersage und beantwortet
# „warum weicht dieses Subject vom Trainingsmittel ab?“, statt „warum ist
# die Intensität nicht 0?“.
#
# Ablauf: Vorhersagen für `N_SUBJ_PRED` Holdout-Subjects → Scatter/MAE/r →
# Tabelle → für Subject `IDX_PRED` Input mit Thalamus-Maske und die
# unnormierten LRP-Heatmaps (klassisch, Trainingsmittel, Ground Truth),
# beschriftet mit Subject-ID und $\widetilde{y}$. Anschließend Vergleich
# mit einem **direkten TensorFlow-Port** der Funktionen `find_a_ref` /
# `restructure_model` aus
# [sltzgs/xai-regression](https://github.com/sltzgs/xai-regression/blob/main/restructuring_implementation.ipynb).
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
# - `PLOT_LANG`: Plot-Beschriftungen `"de"` (Deutsch) oder `"en"` (Englisch).
#

# %%
N_SUBJ_PRED = 3
IDX_PRED = 1
REFERENCE_ASYMMETRY = 0.25

PRED_BATCH_SIZE = 8
SAGITTAL_X = 70
SHOW_PLOTS_INLINE = True
COLOR_RIGHT = (0.15, 0.65, 0.25, 0.50)  # grün, alpha=0.5
PRED_VAR = "average-intensity_right-whole-thalamus"
PLOT_LANG = "de"  # "de" | "en"

PLOT_LABELS = {
    "de": {
        "scatter_title": (
            "Wahr vs. prädiziert  ·  {pred_var}\n"
            "n={n}  MAE={mae:.4f}  r={r:.3f}  ·  "
            "y_ref (Training) = {y_ref:.4f}"
        ),
        "xlabel_true": "wahre Intensität",
        "ylabel_pred": "prädizierte Intensität",
        "identity": "Identität",
        "true_abbr": "wahr",
        "pred_abbr": "präd",
        "y_ref_training": "y_ref Training",
        "classic_label": "klassisch (y_ref = 0, implizit)",
        "paper_label": "Letzgus et al. (y_ref = {y_ref:.4f})",
        "gt_label": "Ground Truth (y_ref = y_true = {y_ref:.4f})",
        "thal_pct": "Thalamus = {pct:.1f}% der Gesamtrelevanz",
        "cbar_intensity": "Intensität",
        "cbar_lrp": "LRP-Relevanz",
        "cbar_delta_lrp": "Δ LRP-Relevanz",
        "legend_right_thal": "rechter Thalamus",
        "diff": "Differenz",
        "diff_train_vs_zero": "Differenz  ·  (y_ref={y_ref:.4f}) − (y_ref=0)",
        "diff_gt_vs_zero": (
            "Differenz  ·  (Ground Truth y_ref={y_ref:.4f}) − (y_ref=0)"
        ),
        "classic_expl": "klassisch (explainability, y_ref=0)",
        "diff_letzgus_vs_expl": (
            "Differenz  ·  (Letzgus-TF-Port) − (explainability)"
        ),
    },
    "en": {
        "scatter_title": (
            "True vs. predicted  ·  {pred_var}\n"
            "n={n}  MAE={mae:.4f}  r={r:.3f}  ·  "
            "y_ref (training) = {y_ref:.4f}"
        ),
        "xlabel_true": "true intensity",
        "ylabel_pred": "predicted intensity",
        "identity": "identity",
        "true_abbr": "true",
        "pred_abbr": "pred",
        "y_ref_training": "y_ref training",
        "classic_label": "classic (y_ref = 0, implicit)",
        "paper_label": "Letzgus et al. (y_ref = {y_ref:.4f})",
        "gt_label": "ground truth (y_ref = y_true = {y_ref:.4f})",
        "thal_pct": "Thalamus = {pct:.1f}% of total relevance",
        "cbar_intensity": "Intensity",
        "cbar_lrp": "LRP relevance",
        "cbar_delta_lrp": "Δ LRP relevance",
        "legend_right_thal": "right thalamus",
        "diff": "Difference",
        "diff_train_vs_zero": "Difference  ·  (y_ref={y_ref:.4f}) − (y_ref=0)",
        "diff_gt_vs_zero": (
            "Difference  ·  (ground-truth y_ref={y_ref:.4f}) − (y_ref=0)"
        ),
        "classic_expl": "classic (explainability, y_ref=0)",
        "diff_letzgus_vs_expl": (
            "Difference  ·  (Letzgus TF port) − (explainability)"
        ),
    },
}
if PLOT_LANG not in PLOT_LABELS:
    raise ValueError(
        f"PLOT_LANG={PLOT_LANG!r} ungültig (erlaubt: {sorted(PLOT_LABELS)})"
    )
PL = PLOT_LABELS[PLOT_LANG]


if not (0 <= int(IDX_PRED) < int(N_SUBJ_PRED)):
    raise ValueError(
        f"IDX_PRED={IDX_PRED} muss kleiner als N_SUBJ_PRED={N_SUBJ_PRED} sein "
        f"(gültig: 0 .. {N_SUBJ_PRED - 1})."
    )

# (1) unveränderte MRI (dataset ukb, original scans)
RUN_DIR = Path(
    "/mnt/ceph2/dl_project/data/nn-trainings/mri/"
    "average-intensity_right-whole-thalamus/"
    "training_run_09h17m23s_23sep2026"
).resolve()
PREDICT_TSV = Path(
    "/mnt/users/andreasre/git-repos/pyment-and1/training_runs/input_files/mri/"
    "right_whole_thalamus/average_intensity/original_scans/predict.tsv"
).resolve()
TRAIN_TSV = Path(
    "/mnt/users/andreasre/git-repos/pyment-and1/training_runs/input_files/mri/"
    "right_whole_thalamus/average_intensity/original_scans/train.tsv"
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
print(f"PLOT_LANG            = {PLOT_LANG}")
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
# ist der Mittelwert der **wahren** `average-intensity_right-whole-thalamus`
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
# `config.yaml` (`nifti-native`). LRP-Composite für SFCN wie im alten
# Intensitäts-Notebook: zwei `flat`-Schichten, vier αβ-Schichten (α=1, β=0),
# ε am Dense-Ausgang.
#
# Ist `training.normalisation.use: true` mit `z_score`, liegen Netzausgabe
# und damit auch $\widetilde{y}$ im z-Raum. Dann gilt
# $\widetilde{y}_{\mathrm{Netz}} = (\bar{y}_{\mathrm{Train}}-\mu)/\sigma$.
# Heatmaps werden hinterher mit $\sigma$ auf die Intensitätsskala
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

#LRP_STRATEGY = LRP_STRATEGY_a2b1
LRP_STRATEGY = LRP_STRATEGY_a1b0

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
# Das Modell sagt für die Holdout-Subjects die mittlere Intensität des
# rechten Thalamus voraus. Wahre Werte kommen aus `predict.tsv`
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
ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label=PL["identity"])
ax.axhline(y_ref_orig, color="C1", ls=":", lw=1.4, label=f"y_ref = {y_ref_orig:.4f}")
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
ax.set_xlabel(PL["xlabel_true"])
ax.set_ylabel(PL["ylabel_pred"])
ax.set_title(
    PL["scatter_title"].format(
        pred_var=PRED_VAR,
        n=N_SUBJ_PRED,
        mae=mae,
        r=r_val,
        y_ref=y_ref_orig,
    )
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
# ## I. Zwei LRP-Heatmaps neu berechnen (unnormiert)
#
# Für Subject `IDX_PRED` werden beide Heatmaps bei jedem Lauf neu berechnet.
#
# - **klassisch:** `LRP(..., reference_value=None)` — implizit $\widetilde{y}=0$
#   im Netzraum.
# - **Letzgus et al.:** `LRP(..., reference_value=y_ref_net)` — Umstrukturierung
#   der letzten Schichten (Flooding + verdreifachte ReLU), ohne Neutraining.
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


vol_net = load_vol(str(vol_path))
t0 = time.perf_counter()
heat_classic = run_lrp(vol_net, reference_value=None)
t_classic = time.perf_counter() - t0
t1 = time.perf_counter()
heat_ref = run_lrp(vol_net, reference_value=y_ref_net)
t_ref = time.perf_counter() - t1

heatmaps = {
    "classic": {
        "R": heat_classic,
        "y_ref_orig": 0.0,
        "y_ref_net": 0.0,
        "label": PL["classic_label"],
        "seconds": t_classic,
    },
    "paper": {
        "R": heat_ref,
        "y_ref_orig": y_ref_orig,
        "y_ref_net": y_ref_net,
        "label": PL["paper_label"].format(y_ref=y_ref_orig),
        "seconds": t_ref,
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

print(
    f"\nDauer beide Heatmaps: {t_classic + t_ref:.2f} s "
    f"({(t_classic + t_ref) / 60.0:.2f} min)"
)



# %% [markdown]
# ## J. Figur: Input + klassische LRP + Referenz-LRP + Differenz
#
# Alle Panels nennen Subject-ID und den verwendeten Referenzwert.
#
# - oben links: Input, sagittal `x=70`, Thalamus-Umrandung
# - oben rechts: klassische LRP ($\widetilde{y}=0$)
# - unten links: LRP nach Letzgus et al. ($\widetilde{y}=$ Trainingsmittel)
# - unten rechts: Differenz (Letzgus − klassisch)
#

# %%
COLOR_OUTLINE = COLOR_RIGHT[:3]
pos = plot_vol[plot_vol > 0]
vmax_i = float(np.percentile(pos, 99.5)) if pos.size else 1.0
r_slc = sagittal_slc(plot_right.astype(np.float32), cx) > 0

fig, axes = plt.subplots(2, 2, figsize=(11.2, 10.6))
fig.suptitle(
    (
        f"Subject {subject_id_plot}  ·  IDX_PRED={IDX_PRED}/{N_SUBJ_PRED}  ·  "
        f"sagittal x={cx}  ·  {PRED_VAR}\n"
        f"{PL['true_abbr']}={plot_true:.4f}  {PL['pred_abbr']}={plot_pred:.4f}  ·  "
        f"{PL['y_ref_training']} = {y_ref_orig:.4f}"
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


ax = axes[0, 0]
im0 = ax.imshow(sagittal_slc(plot_vol, cx), cmap="gray", vmin=0.0, vmax=vmax_i)
_draw_contour(ax)
ax.set_title(
    f"Input  ·  Subject {subject_id_plot}\n"
    f"{PL['true_abbr']}={plot_true:.4f}  {PL['pred_abbr']}={plot_pred:.4f}",
    fontsize=9,
)
ax.axis("off")
cbar = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(PL["cbar_intensity"])

panel_specs = [
    (axes[0, 1], "classic"),
    (axes[1, 0], "paper"),
]
ims_lrp = []
for ax, key in panel_specs:
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
        f"{PL['true_abbr']}={plot_true:.4f}  {PL['pred_abbr']}={plot_pred:.4f}  "
        f"|R|_max={abs_max:.3g}\n"
        f"{PL['thal_pct'].format(pct=meta['pct_thal'])}  ·  "
        f"ΣR={meta['sum_R']:.3g}",
        fontsize=9,
    )
    ax.axis("off")
    ims_lrp.append(im)
    print(
        f"[{key}] Thalamus-Anteil = {meta['pct_thal']:.2f}%  "
        f"(ΣR_Thalamus={meta['sum_R_thal']:.4g} / ΣR={meta['sum_R']:.4g})  "
        f"y_ref={meta['y_ref_orig']:.6g}"
    )

for ax, im in zip((axes[0, 1], axes[1, 0]), ims_lrp):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(PL["cbar_lrp"])

ax = axes[1, 1]
diff = heatmaps["paper"]["R"] - heatmaps["classic"]["R"]
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
    f"{PL['diff_train_vs_zero'].format(y_ref=y_ref_orig)}\n"
    f"Subject {subject_id_plot}\n"
    f"|ΔR|_max={abs_max_d:.3g}  ΣΔR={float(np.sum(diff)):.3g}",
    fontsize=9,
)
ax.axis("off")
cbar = fig.colorbar(im_d, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(PL["cbar_delta_lrp"])

fig.legend(
    handles=[
        Patch(
            facecolor="none",
            edgecolor=COLOR_OUTLINE,
            linewidth=1.5,
            label=PL["legend_right_thal"],
        )
    ],
    loc="lower center",
    ncol=1,
    frameon=False,
    fontsize=9,
)
fig.tight_layout(rect=[0, 0.05, 1, 0.90])

if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)

print(
    f"Heatmap-Berechnung: klassisch {t_classic:.2f} s, "
    f"Letzgus {t_ref:.2f} s  (Subject {subject_id_plot})."
)


# %% [markdown]
# ## K. LRP mit Ground-Truth-Referenzwert
#
# Dritte Variante nach Letzgus et al.: $\widetilde{y}$ wird auf den **wahren
# Label-Wert** des Plot-Subjects gesetzt (`y_true` aus `predict.tsv`), nicht
# auf das Trainingsmittel.
#
# Damit erklärt LRP $g(x)=f(x)-y_{\mathrm{true}}$, also das Residuum der
# Vorhersage gegenüber der Ground Truth. Im Plot ist der Referenzwert
# explizit als **Ground Truth** beschriftet.
#
# - **klassisch:** $\widetilde{y}=0$ (wie in Sektion I/J)
# - **Ground Truth:** $\widetilde{y}=y_{\mathrm{true}}$ des Subjects `IDX_PRED`

# %%
y_ref_gt_orig = float(plot_true)
y_ref_gt_net = to_network_space(y_ref_gt_orig, zscore_params)

t2 = time.perf_counter()
heat_gt = run_lrp(vol_net, reference_value=y_ref_gt_net)
t_gt = time.perf_counter() - t2

heatmaps_gt = {
    "classic": heatmaps["classic"],
    "ground_truth": {
        "R": heat_gt,
        "y_ref_orig": y_ref_gt_orig,
        "y_ref_net": y_ref_gt_net,
        "label": PL["gt_label"].format(y_ref=y_ref_gt_orig),
        "seconds": t_gt,
    },
}

for key, meta in heatmaps_gt.items():
    if key == "classic":
        continue
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
    print(
        f"        y_ref Label={y_ref_gt_orig:.6g}  "
        f"y_ref Netzraum={y_ref_gt_net:.6g}  "
        f"Residuum pred−true = {plot_pred - plot_true:.6g}"
    )

fig, axes = plt.subplots(2, 2, figsize=(11.2, 10.6))
fig.suptitle(
    (
        f"Subject {subject_id_plot}  ·  IDX_PRED={IDX_PRED}/{N_SUBJ_PRED}  ·  "
        f"sagittal x={cx}  ·  {PRED_VAR}\n"
        f"{PL['true_abbr']}={plot_true:.4f}  {PL['pred_abbr']}={plot_pred:.4f}  ·  "
        f"y_ref = Ground Truth = {y_ref_gt_orig:.4f}"
    ),
    fontsize=11,
)

ax = axes[0, 0]
im0 = ax.imshow(sagittal_slc(plot_vol, cx), cmap="gray", vmin=0.0, vmax=vmax_i)
_draw_contour(ax)
ax.set_title(
    f"Input  ·  Subject {subject_id_plot}\n"
    f"{PL['true_abbr']}={plot_true:.4f}  {PL['pred_abbr']}={plot_pred:.4f}",
    fontsize=9,
)
ax.axis("off")
cbar = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(PL["cbar_intensity"])

panel_specs_gt = [
    (axes[0, 1], "classic"),
    (axes[1, 0], "ground_truth"),
]
ims_lrp_gt = []
for ax, key in panel_specs_gt:
    meta = heatmaps_gt[key]
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
        f"{PL['true_abbr']}={plot_true:.4f}  {PL['pred_abbr']}={plot_pred:.4f}  "
        f"|R|_max={abs_max:.3g}\n"
        f"{PL['thal_pct'].format(pct=meta['pct_thal'])}  ·  "
        f"ΣR={meta['sum_R']:.3g}",
        fontsize=9,
    )
    ax.axis("off")
    ims_lrp_gt.append(im)
    print(
        f"[{key}] Thalamus-Anteil = {meta['pct_thal']:.2f}%  "
        f"(ΣR_Thalamus={meta['sum_R_thal']:.4g} / ΣR={meta['sum_R']:.4g})  "
        f"y_ref={meta['y_ref_orig']:.6g}"
    )

for ax, im in zip((axes[0, 1], axes[1, 0]), ims_lrp_gt):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(PL["cbar_lrp"])

ax = axes[1, 1]
diff_gt = heatmaps_gt["ground_truth"]["R"] - heatmaps_gt["classic"]["R"]
abs_max_d = float(np.nanmax(np.abs(diff_gt))) or 1.0
vmax_d = float(np.nanpercentile(np.abs(diff_gt), 99.5)) or abs_max_d
im_d = ax.imshow(
    sagittal_slc(diff_gt, cx),
    cmap="RdBu_r",
    vmin=-vmax_d,
    vmax=vmax_d,
)
_draw_contour(ax)
ax.set_title(
    f"{PL['diff_gt_vs_zero'].format(y_ref=y_ref_gt_orig)}\n"
    f"Subject {subject_id_plot}\n"
    f"|ΔR|_max={abs_max_d:.3g}  ΣΔR={float(np.sum(diff_gt)):.3g}",
    fontsize=9,
)
ax.axis("off")
cbar = fig.colorbar(im_d, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(PL["cbar_delta_lrp"])

fig.legend(
    handles=[
        Patch(
            facecolor="none",
            edgecolor=COLOR_OUTLINE,
            linewidth=1.5,
            label=PL["legend_right_thal"],
        )
    ],
    loc="lower center",
    ncol=1,
    frameon=False,
    fontsize=9,
)
fig.tight_layout(rect=[0, 0.05, 1, 0.90])

if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)

print(
    f"Heatmap-Berechnung: Ground-Truth-Referenz {t_gt:.2f} s  "
    f"(Subject {subject_id_plot}, y_ref = y_true = {y_ref_gt_orig:.4f})."
)


# %% [markdown]
# ## L. Vergleich: Letzgus-TF-Port vs. `explainability`-LRP
#
# Direkter TensorFlow-/Keras-Port der Idee aus
# [restructuring_implementation.ipynb](https://github.com/sltzgs/xai-regression/blob/main/restructuring_implementation.ipynb)
# (`find_a_ref`, `restructure_model`), **korrigiert** auf die Mathematik aus
# Letzgus et al. 2022 (Abschn. IV-B) bzw. `explainability.model.utils.reference`.
#
# Früherer Bug in diesem Notebook: `a_ref` wurde aus **post-GAP**-Aktivierungen
# geschätzt und in die **Conv-Biases** gebacken. Bei Topologie
# `Conv → ReLU → GAP → Dense` gilt aber
# $\mathrm{GAP}(\mathrm{ReLU}(z-a_{\mathrm{ref}}))\neq\mathrm{GAP}(\mathrm{ReLU}(z))-a_{\mathrm{ref}}$.
# Die Paper-Identität braucht Offsets auf **ReLU-/Pre-Aktivierungsebene**.
#
# | | Dieser Port (korrigiert) | `explainability` (`reference_value=…`) |
# |--|--|--|
# | Flooding | Bisektion mit $\kappa$ (`REFERENCE_ASYMMETRY`), optional Uniform-Schritt | Bisektion mit $\kappa$ in `ReferenceShift` |
# | $a_{\mathrm{ref}}$ | fest pro Sample, Form = ReLU-Units; Offset per `Add` auf $z$ | sample-abhängig in `ReferenceShift` |
# | Pre-Bias nach Umbau | $(b,-b,-b)$, Offsets **nicht** in Conv-Bias | $(b,-b,-b)$ + `ReferenceShift` |
# | Dense-Bias | Original bleibt | Original bleibt |
# | LRP | auf umgebautes Netz, `ignore_bias` auf verdreifachter Pre-Schicht | intern `ignore_bias=True` am Top |
#
# Gemeinsame Referenz: Ground Truth $y_{\mathrm{true}}$ des Plot-Subjects (wie Sektion K), nicht das Trainingsmittel.
# Heatmaps unnormiert, bei z-Score mit $\sigma$ skaliert.
#

# %%
# ---------------------------------------------------------------------------
# Korrigierter TensorFlow-Port von Letzgus et al. (Abschn. IV-B)
#   Inspiration: https://github.com/sltzgs/xai-regression
#                restructuring_implementation.ipynb  (find_a_ref, restructure_model)
#   Referenz-Implementierung (korrekt): explainability.model.utils.reference
#
# Bugfix (dieser Notebook-Port):
#   - a_ref aus post-GAP-Aktivierungen + Einbacken in Conv-Biases
#     → bricht relu(z)-a_ref = … bei Conv→ReLU→GAP
#   - Dense-Bias auf 0 gesetzt (inkonsistent zur Library-Konvention
#     sum(a_ref*c)=y_ref bei erhaltendem Dense-Bias)
#   - LRP ohne ignore_bias auf der verdreifachten Pre-Schicht
# ---------------------------------------------------------------------------

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Activation,
    Add,
    Dense,
    Dropout,
    Flatten,
)
from explainability.layers.layer import StandardLRPLayer
from explainability.model.utils import fuse_batchnorm, remove_activation
from explainability.model.utils.reference import (
    _fold_norms,
    _strip_range_clip,
    _walk_top_block,
    _dense_input_shape,
    find_linear_dense_head,
    output_coefficients,
)

# Flooding-Parameter: κ wie in Sektion B / explainability-Pfad.
LETZGUS_ASYMMETRY = float(REFERENCE_ASYMMETRY)
LETZGUS_BISECTION_STEPS = 60
# Optionaler Uniform-Schritt-Port (PyTorch-Notebook); Default = Bisektion.
LETZGUS_USE_UNIFORM_STEP = False
LETZGUS_STEP_WIDTH = 0.005
LETZGUS_MAX_IT = 100_000
LETZGUS_NORMALIZE_TOP = False  # rescale_top bleibt optional aus


def rescale_top_weights(W_in, b_in, W_out, b_out):
    """Port von ``rescale_top`` (utils_.py): skaliert die letzte Schicht auf ±1.

    Multipliziert eingehende Gewichte/Bias mit |W_out| und ersetzt W_out durch
    sign(W_out). Optional; hier standardmäßig nicht verwendet
    (``LETZGUS_NORMALIZE_TOP=False``).
    """
    W_in = np.asarray(W_in, dtype=np.float64)
    b_in = np.asarray(b_in, dtype=np.float64).reshape(-1)
    W_out = np.asarray(W_out, dtype=np.float64)
    b_out = np.asarray(b_out, dtype=np.float64).reshape(-1)
    w_abs = np.abs(W_out.reshape(-1))
    if W_in.ndim == 2:
        W_in_new = W_in * w_abs[None, :]
    else:
        W_in_new = W_in * w_abs.reshape((1,) * (W_in.ndim - 1) + (-1,))
    b_in_new = b_in * w_abs
    W_out_new = np.sign(W_out)
    return (
        W_in_new.astype(np.float32),
        b_in_new.astype(np.float32),
        W_out_new.astype(np.float32),
        b_out.astype(np.float32),
    )


def find_a_ref_bisection(
    a: np.ndarray,
    c: np.ndarray,
    y_ref: float,
    *,
    asymmetry: float = LETZGUS_ASYMMETRY,
    bisection_steps: int = LETZGUS_BISECTION_STEPS,
    verbose: bool = True,
) -> np.ndarray:
    """Flooding-Regel per Bisektion (Letzgus et al. / ``ReferenceShift``).

    Sucht ``a_ref = relu(a - t * d)`` so dass ``sum(a_ref * c) ≈ y_ref``.

    Parameters
    ----------
    a :
        ReLU-Aktivierungen der letzten versteckten Schicht, Form wie die
        ReLU-Units (ohne Batch), z. B. ``(D, H, W, C)`` oder ``(C,)``.
    c :
        Effektive Ausgangsgewichte je ReLU-Unit (gleiche Form wie ``a``),
        inkl. GAP-Normierung ``w / V`` — siehe ``output_coefficients``.
    y_ref :
        Ziel-Referenzwert im Netzraum. Konvention wie in ``explainability``:
        ``sum(a_ref * c) = y_ref`` (Dense-Bias bleibt separat erhalten).
    asymmetry :
        κ der Flooding-Richtung: Units mit Evidenz Richtung Ziel bekommen
        ``d=+1``, gegenläufige Units ``d=-κ`` (0 = einseitig).
    bisection_steps :
        Anzahl Bisektionsschritte für das Skalar ``t``.
    verbose :
        Fortschritt / Erreichbarkeit loggen.

    Returns
    -------
    a_ref : ndarray
        Referenz-Aktivierungen, gleiche Form wie ``a``.
    """
    a = np.asarray(a, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if a.shape != c.shape:
        raise ValueError(f"a-Form {a.shape} ≠ c-Form {c.shape}")
    if asymmetry < 0:
        raise ValueError("asymmetry muss >= 0 sein")

    y_ref = float(y_ref)
    g0 = float(np.sum(a * c))
    # +1: Evidenz entfernen (y zu hoch), -1: Evidenz hinzufügen.
    direction = 1.0 if g0 >= y_ref else -1.0
    cd = c * direction
    d = np.where(cd > 0, 1.0, np.where(cd < 0, -float(asymmetry), 0.0))

    def h(t: float) -> float:
        g = float(np.sum(np.maximum(0.0, a - t * d) * c))
        return (g - y_ref) * direction

    hi = float(np.max(a)) + 1.0
    if asymmetry > 0:
        for _ in range(64):
            if h(hi) <= 0:
                break
            hi *= 2.0
    if h(hi) > 0:
        raise RuntimeError(
            f"y_ref={y_ref} ist mit Flooding (κ={asymmetry}) nicht erreichbar "
            f"(g0={g0:.6g}, h(hi)={h(hi):.6g})."
        )

    lo = 0.0
    for _ in range(int(bisection_steps)):
        mid = 0.5 * (lo + hi)
        if h(mid) > 0:
            lo = mid
        else:
            hi = mid
    t = 0.5 * (lo + hi)
    a_ref = np.maximum(0.0, a - t * d).astype(np.float32)
    g_ref = float(np.sum(a_ref.astype(np.float64) * c))
    if verbose:
        print(
            f"find_a_ref_bisection: t={t:.6g}  g0={g0:.6g}  "
            f"sum(a_ref*c)={g_ref:.6g}  y_ref={y_ref:.6g}  κ={asymmetry}"
        )
    return a_ref


def find_a_ref_uniform_step(
    a: np.ndarray,
    c: np.ndarray,
    y_ref: float,
    *,
    step_width: float = LETZGUS_STEP_WIDTH,
    max_it: int = LETZGUS_MAX_IT,
    verbose: bool = True,
) -> np.ndarray:
    """Flooding mit uniformem Schritt (Stil des PyTorch-Notebooks).

    Im Gegensatz zum alten Notebook-Bug arbeitet diese Variante auf denselben
    ReLU-Units und Koeffizienten ``c`` wie die Bisektion
    (``sum(a_ref * c) → y_ref``), nicht auf post-GAP-Vektoren.
    """
    a_ref = np.asarray(a, dtype=np.float64).copy()
    c = np.asarray(c, dtype=np.float64)
    if a_ref.shape != c.shape:
        raise ValueError(f"a-Form {a_ref.shape} ≠ c-Form {c.shape}")
    y_ref = float(y_ref)
    y_t = float(np.sum(a_ref * c))
    if verbose:
        print(f"y = {np.round(y_t, 2)} / y_ref = {y_ref}")

    update = np.full_like(a_ref, float(step_width))
    counter = 0
    if y_t > y_ref:
        while y_t > y_ref:
            a_ref = np.maximum(0.0, a_ref - update)
            y_t = float(np.sum(a_ref * c))
            counter += 1
            if verbose and (counter % 500 == 0 or counter == 1):
                print(f"iteration {counter} - y_t: {y_t:.6g}", end="\r")
            if counter > max_it:
                print(
                    f"\n! y_ref {y_ref} nicht in {int(max_it)} Iterationen erreicht"
                )
                break
    else:
        while y_t < y_ref:
            a_ref = np.maximum(0.0, a_ref + update)
            y_t = float(np.sum(a_ref * c))
            counter += 1
            if verbose and (counter % 500 == 0 or counter == 1):
                print(f"iteration {counter} - y_t: {y_t:.6g}", end="\r")
            if counter > max_it:
                print(
                    f"\n! y_ref {y_ref} nicht in {int(max_it)} Iterationen erreicht"
                )
                break
    if verbose:
        print(
            f"\nfind_a_ref_uniform_step: {counter} Iterationen  "
            f"sum(a_ref*c)={y_t:.6g}  y_ref={y_ref:.6g}"
        )
    return a_ref.astype(np.float32)


def find_a_ref_tf(
    a: np.ndarray,
    c: np.ndarray,
    y_ref: float,
    **kwargs,
) -> np.ndarray:
    """Dispatch: Bisektion (Default) oder Uniform-Schritt.

    Siehe ``find_a_ref_bisection`` / ``find_a_ref_uniform_step``.
    """
    if LETZGUS_USE_UNIFORM_STEP:
        return find_a_ref_uniform_step(a, c, y_ref, **kwargs)
    return find_a_ref_bisection(a, c, y_ref, **kwargs)


def _prepare_head_model(base_model, layer_idx: int):
    """Bereitet den Regressionskopf wie ``LRP`` vor.

    Schneidet auf den linearen Dense-Kopf, entfernt sigmoid/softmax,
    fusioniert BatchNorm und entfernt Range-Clipping. Ergebnis ist das
    Modell, auf dem ``restructure_for_reference`` / dieser Port arbeitet.
    """
    m = Model(base_model.input, base_model.layers[int(layer_idx)].output)
    m = remove_activation(m, ["sigmoid", "softmax"])
    m = fuse_batchnorm(m)
    m = _strip_range_clip(m)
    return m


def _relu_activation_model(head_model):
    """Baut ein Modell, das die ReLU-Aktivierungen der letzten Hidden-Schicht liefert.

    Returns
    -------
    act_model :
        ``x → relu(pre(x))`` (nach optional gefalteter BN).
    dense, chain, relu, pre, norms, kernel, bias :
        Top-Block-Metadaten bzw. gefaltete Pre-Gewichte.
    """
    dense, chain, relu, pre, norms = _walk_top_block(head_model)
    pre_weights = pre.get_weights()
    kernel = pre_weights[0]
    bias = (
        pre_weights[1]
        if pre.use_bias
        else np.zeros(kernel.shape[-1], dtype=kernel.dtype)
    )
    kernel, bias = _fold_norms(kernel, bias, norms)

    x = pre.input
    if isinstance(pre, Dense):
        pre_lin = Dense(
            pre.units, activation=None, use_bias=True, name="letzgus_pre_lin"
        )
        z = pre_lin(x)
    else:
        pre_lin = pre.__class__(
            pre.filters,
            kernel_size=pre.kernel_size,
            strides=pre.strides,
            padding=pre.padding,
            dilation_rate=pre.dilation_rate,
            activation=None,
            use_bias=True,
            name="letzgus_pre_lin",
        )
        z = pre_lin(x)
    a = Activation("relu", name="letzgus_relu")(z)
    act_model = Model(head_model.input, a, name="letzgus_relu_act")
    pre_lin.set_weights([kernel, bias])
    return act_model, dense, chain, relu, pre, norms, kernel, bias


def restructure_model_letzgus_tf(
    head_model,
    a_ref: np.ndarray,
    *,
    normalize_top: bool = False,
    name: str = "letzgus_tf",
):
    """Port von ``restructure_model``: verdreifacht die letzte Hidden-Schicht.

    Korrekte Umsetzung der Paper-Identität
    ``relu(z) - a_ref = relu(z - a_ref) + relu(-z) - relu(-z + a_ref)``:

    1. Pre-Gewichte verdreifachen: ``(W, -W, -W)``, Bias ``(b, -b, -b)``.
    2. Offsets ``(-a_ref, 0, +a_ref)`` per ``Add`` auf die Pre-Aktivierungen
       (gleiche Form wie die ReLU-Units — **nicht** post-GAP in Conv-Bias).
    3. Ausgangsgewichte ``(w, w, -w)``; **Dense-Bias bleibt erhalten**
       (Konvention ``sum(a_ref * c) = y_ref``).

    Parameters
    ----------
    head_model :
        Vorbereiteter Kopf (siehe ``_prepare_head_model``).
    a_ref :
        Referenz-Aktivierungen, Form = Pre-/ReLU-Output ohne Batch.
    normalize_top :
        Optional ``rescale_top`` vor dem Umbau.
    name :
        Namenspräfix neuer Schichten.

    Returns
    -------
    Model
        ``g(x) ≈ f(x) - y_ref`` für das Sample, zu dem ``a_ref`` gehört.
    """
    dense, chain, relu, pre, norms = _walk_top_block(head_model)
    pre_weights = pre.get_weights()
    kernel = pre_weights[0]
    bias = (
        pre_weights[1]
        if pre.use_bias
        else np.zeros(kernel.shape[-1], dtype=np.float32)
    )
    kernel, bias = _fold_norms(kernel, bias, norms)
    w_out, *b_out_list = dense.get_weights()
    b_out = b_out_list[0] if b_out_list else np.zeros(dense.units, dtype=np.float32)

    a_ref = np.asarray(a_ref, dtype=np.float32)
    unit_shape = tuple(int(s) for s in pre.output.shape[1:])
    if tuple(a_ref.shape) != unit_shape:
        raise ValueError(
            f"a_ref-Form {a_ref.shape} ≠ Pre-/ReLU-Form {unit_shape}. "
            "a_ref muss auf ReLU-Unit-Ebene liegen (nicht post-GAP)."
        )

    if normalize_top:
        in_shape = _dense_input_shape(dense, chain)
        w_eff = w_out[:, 0].reshape(in_shape)
        if w_eff.ndim > 1:
            w_ch = w_eff.reshape(-1, unit_shape[-1]).mean(axis=0)
        else:
            w_ch = w_eff.reshape(-1)
        # rescale nur kanalweise am Pre-Layer (Dense-Fall: units)
        if kernel.ndim == 2:
            kernel, bias, w_ch, b_out = rescale_top_weights(
                kernel, bias, w_ch, b_out
            )
            w_out = w_ch.reshape(-1, 1).astype(np.float32)
        else:
            # Conv: |c| aus effektiven Kanalgewichten (Mittel über Raum)
            c_ch = w_eff.reshape(-1, unit_shape[-1]).mean(axis=0)
            kernel, bias, c_ch, b_out = rescale_top_weights(
                kernel, bias, c_ch, b_out
            )
            n_spatial = len(unit_shape) - 1
            positions = int(np.prod(unit_shape[:-1]))
            w_spatial = np.broadcast_to(
                c_ch.reshape((1,) * n_spatial + (unit_shape[-1],)),
                unit_shape,
            )
            w_out = w_spatial.reshape(-1, 1).astype(np.float32)

    # Schritt 2 (Paper): verdreifachte lineare Pre-Schicht ohne a_ref im Bias.
    kernel_tri = np.concatenate([kernel, -kernel, -kernel], axis=-1)
    bias_tri = np.concatenate([bias, -bias, -bias], axis=-1)

    x_in = pre.input
    if isinstance(pre, Dense):
        pre_tri = Dense(
            3 * pre.units,
            activation=None,
            use_bias=True,
            name=f"{name}_pre",
        )
    else:
        pre_tri = pre.__class__(
            3 * pre.filters,
            kernel_size=pre.kernel_size,
            strides=pre.strides,
            padding=pre.padding,
            dilation_rate=pre.dilation_rate,
            activation=None,
            use_bias=True,
            name=f"{name}_pre",
        )
    z_tri = pre_tri(x_in)
    pre_tri.set_weights([kernel_tri, bias_tri])

    # Offsets auf Pre-Aktivierungsebene (sample-feste Konstante).
    # AddLRP reicht bei konstantem Operanden die Relevanz an z_tri durch.
    shift = np.concatenate(
        [-a_ref, np.zeros_like(a_ref), a_ref], axis=-1
    ).astype(np.float32)
    z_shift = Add(name=f"{name}_shift")(
        [z_tri, tf.constant(shift[None, ...], dtype=tf.float32)]
    )
    h = Activation("relu", name=f"{name}_relu")(z_shift)

    for layer in chain:
        if isinstance(layer, Dropout):
            continue
        if isinstance(layer, Flatten):
            h = Flatten(name=f"{name}_{layer.name}")(h)
        else:
            pool_kwargs = {"name": f"{name}_{layer.name}"}
            if getattr(layer, "keepdims", None) is not None:
                pool_kwargs["keepdims"] = layer.keepdims
            df = getattr(layer, "data_format", None)
            if df is not None:
                pool_kwargs["data_format"] = df
            h = layer.__class__(**pool_kwargs)(h)

    # Ausgang: (w, w, −w), Dense-Bias unverändert.
    in_shape = _dense_input_shape(dense, chain)
    w = w_out.reshape(in_shape + (dense.units,))
    w_tri = np.concatenate([w, w, -w], axis=len(in_shape) - 1)
    w_tri = w_tri.reshape((-1, dense.units))
    dense_tri = Dense(
        dense.units,
        activation=None,
        use_bias=True,
        name=f"{name}_dense",
    )
    out = dense_tri(h)
    dense_tri.set_weights([w_tri, np.asarray(b_out, dtype=np.float32)])

    return Model(head_model.input, out, name=f"{name}_{head_model.name}")


def _enable_ignore_bias_on_pre(lrp_model, pre_name_prefix: str = "letzgus_tf_pre"):
    """Setzt ``ignore_bias=True`` auf der verdreifachten Pre-Schicht.

    Entspricht Letzgus et al., Sec. VI-A / dem Sonderpfad in
    ``LayerwiseRelevancePropagator`` bei ``reference_value``: Biases der
    Flooding-Schicht werden in der FC-/Conv-LRP ignoriert, damit LRP-0 den
    Umbau nicht „zurückrechnet“. Die Referenz steckt in den verdreifachten
    ReLU-Aktivierungen.
    """
    n = 0
    for lyr in lrp_model.layers:
        if not isinstance(lyr, StandardLRPLayer):
            continue
        base = getattr(lyr, "layer", None)
        if base is None:
            continue
        if str(base.name).startswith(pre_name_prefix):
            lyr.ignore_bias = True
            n += 1
    if n == 0:
        raise RuntimeError(
            f"Keine StandardLRPLayer mit layer.name-Prefix "
            f"{pre_name_prefix!r} gefunden — ignore_bias nicht gesetzt."
        )
    return n


# --- Anwenden auf Subject IDX_PRED mit y_ref = Ground Truth (Netzraum) ---
y_ref_compare_orig = float(plot_true)
y_ref_compare = float(to_network_space(y_ref_compare_orig, zscore_params))
head_model = _prepare_head_model(model, LAYER_IDX)
act_model, dense_h, chain_h, relu_h, pre_h, norms_h, kern_h, bias_h = (
    _relu_activation_model(head_model)
)
coefficients = output_coefficients(head_model, idx=0)

vol_batch = np.expand_dims(vol_net, 0)
a_relu = np.asarray(act_model.predict(vol_batch, verbose=0), dtype=np.float64)
# Batch-Dimension entfernen → Form der ReLU-Units
a_units = np.squeeze(a_relu, axis=0)
if a_units.shape != coefficients.shape:
    raise ValueError(
        f"ReLU-Aktivierungsform {a_units.shape} ≠ Koeffizienten {coefficients.shape}"
    )

print("Letzgus-TF-Port (korrigiert): Flooding auf ReLU-Units")
print(f"  pre={pre_h.name} ({pre_h.__class__.__name__})  dense={dense_h.name}")
print(
    f"  a_relu shape={a_units.shape}  c shape={coefficients.shape}  "
    f"|a|_1={float(np.sum(a_units)):.4g}"
)
print(
    f"  Flooding={'uniform step' if LETZGUS_USE_UNIFORM_STEP else 'bisection'}  "
    f"κ={LETZGUS_ASYMMETRY}"
)

a_ref_letzgus = find_a_ref_tf(
    a_units,
    coefficients,
    y_ref=y_ref_compare,
    verbose=True,
)

model_letzgus = restructure_model_letzgus_tf(
    head_model,
    a_ref_letzgus,
    normalize_top=LETZGUS_NORMALIZE_TOP,
    name="letzgus_tf",
)

y_orig = float(np.squeeze(head_model.predict(vol_batch, verbose=0)))
y_let = float(np.squeeze(model_letzgus.predict(vol_batch, verbose=0)))
resid = y_orig - y_ref_compare
print(
    f"\nCheck Umbau: f(x)={y_orig:.6g}  g_letzgus(x)={y_let:.6g}  "
    f"f−y_ref={resid:.6g}  "
    f"|g − (f−y_ref)|={abs(y_let - resid):.3g}"
)
if abs(y_let - resid) > 1e-3 * max(1.0, abs(resid)):
    print(
        "WARNUNG: g(x) weicht deutlich von f(x)−y_ref ab — Umbau prüfen."
    )

# LRP auf Letzgus-umgebautes Netz (ohne reference_value — Umbau steckt schon drin).
# Wichtig: ignore_bias auf verdreifachter Pre-Schicht (wie Library-Pfad).
layer_letzgus = find_linear_dense_head(model_letzgus)
t_l0 = time.perf_counter()
lrp_letzgus = LRP(
    model_letzgus,
    layer=layer_letzgus,
    idx=0,
    strategy=LRP_STRATEGY,
)
n_ignore = _enable_ignore_bias_on_pre(lrp_letzgus, "letzgus_tf_pre")
print(f"ignore_bias gesetzt auf {n_ignore} Pre-LRP-Schicht(en)")

R_letzgus = np.asarray(
    lrp_letzgus(vol_batch)[0].numpy(), dtype=np.float64
).squeeze()
if zscore_params["use"]:
    R_letzgus = R_letzgus * float(zscore_params["std"])
R_letzgus = np.asarray(R_letzgus, dtype=np.float32)
t_letzgus = time.perf_counter() - t_l0

# Referenz aus Sektion K: explainability mit Ground-Truth-Referenz
R_expl = np.asarray(heatmaps_gt["ground_truth"]["R"], dtype=np.float32)
R_classic = np.asarray(heatmaps["classic"]["R"], dtype=np.float32)


def _lrp_stats(R, mask):
    """ΣR, Thalamus-Anteil und |R|_max für eine Heatmap."""
    s = float(np.sum(R))
    st = float(np.sum(R[mask]))
    pct = 100.0 * st / s if abs(s) > 0 else float("nan")
    return s, st, pct, float(np.nanmax(np.abs(R)))


sum_l, sum_l_th, pct_l, amax_l = _lrp_stats(R_letzgus, plot_right)
sum_e, sum_e_th, pct_e, amax_e = _lrp_stats(R_expl, plot_right)
diff_le = R_letzgus - R_expl
corr = float(
    np.corrcoef(R_letzgus.ravel(), R_expl.ravel())[0, 1]
) if R_letzgus.size > 1 else float("nan")

print(
    f"\n[letzgus_tf]  ΣR={sum_l:.4g}  Thalamus={pct_l:.1f}%  "
    f"|R|_max={amax_l:.4g}  Dauer={t_letzgus:.2f} s"
)
print(
    f"[explainability ground_truth]  ΣR={sum_e:.4g}  Thalamus={pct_e:.1f}%  "
    f"|R|_max={amax_e:.4g}"
)
print(
    f"Vergleich Letzgus-TF vs explainability:  "
    f"Pearson-r={corr:.4f}  ΣΔR={float(np.sum(diff_le)):.4g}  "
    f"|ΔR|_max={float(np.nanmax(np.abs(diff_le))):.4g}"
)

# --- Figur ---
fig, axes = plt.subplots(2, 2, figsize=(11.2, 10.6))
fig.suptitle(
    (
        f"Vergleich Letzgus-TF-Port (korrigiert) vs. explainability  ·  "
        f"Subject {subject_id_plot}  ·  sagittal x={cx}\n"
        f"y_ref = Ground Truth = {y_ref_compare_orig:.4f}  "
        f"(Netzraum {y_ref_compare:.4f})  ·  "
        f"{PL['true_abbr']}={plot_true:.4f}  {PL['pred_abbr']}={plot_pred:.4f}"
    ),
    fontsize=11,
)


def _panel(ax, heat, title):
    """Ein Heatmap-Panel mit RdBu_r und Thalamus-Kontur."""
    abs_max = float(np.nanmax(np.abs(heat))) or 1.0
    vmax_r = float(np.nanpercentile(np.abs(heat), 99.5)) or abs_max
    im = ax.imshow(
        sagittal_slc(heat, cx),
        cmap="RdBu_r",
        vmin=-vmax_r,
        vmax=vmax_r,
    )
    _draw_contour(ax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(PL["cbar_lrp"])
    return im


_panel(
    axes[0, 0],
    R_classic,
    (
        f"{PL['classic_expl']}\n"
        f"Subject {subject_id_plot}  ΣR={heatmaps['classic']['sum_R']:.3g}  "
        f"Thalamus={heatmaps['classic']['pct_thal']:.1f}%"
    ),
)
_panel(
    axes[0, 1],
    R_expl,
    (
        f"explainability  ·  Letzgus-Umbau (Bisektion, κ={REFERENCE_ASYMMETRY})\n"
        f"y_ref = Ground Truth = {y_ref_compare_orig:.4f}\n"
        f"ΣR={sum_e:.3g}  Thalamus={pct_e:.1f}%  |R|_max={amax_e:.3g}"
    ),
)
flood_label = (
    f"uniform step={LETZGUS_STEP_WIDTH}"
    if LETZGUS_USE_UNIFORM_STEP
    else f"Bisektion, κ={LETZGUS_ASYMMETRY}"
)
_panel(
    axes[1, 0],
    R_letzgus,
    (
        f"Letzgus-TF-Port  ·  find_a_ref + restructure_model\n"
        f"y_ref = Ground Truth = {y_ref_compare_orig:.4f}  ({flood_label})\n"
        f"ΣR={sum_l:.3g}  Thalamus={pct_l:.1f}%  |R|_max={amax_l:.3g}"
    ),
)

abs_max_d = float(np.nanmax(np.abs(diff_le))) or 1.0
vmax_d = float(np.nanpercentile(np.abs(diff_le), 99.5)) or abs_max_d
im_d = axes[1, 1].imshow(
    sagittal_slc(diff_le, cx),
    cmap="RdBu_r",
    vmin=-vmax_d,
    vmax=vmax_d,
)
_draw_contour(axes[1, 1])
axes[1, 1].set_title(
    (
        f"{PL['diff_letzgus_vs_expl']}\n"
        f"Subject {subject_id_plot}  Pearson-r={corr:.4f}\n"
        f"|ΔR|_max={abs_max_d:.3g}  ΣΔR={float(np.sum(diff_le)):.3g}"
    ),
    fontsize=9,
)
axes[1, 1].axis("off")
cbar = fig.colorbar(im_d, ax=axes[1, 1], fraction=0.046, pad=0.04)
cbar.set_label(PL["cbar_delta_lrp"])

fig.legend(
    handles=[
        Patch(
            facecolor="none",
            edgecolor=COLOR_OUTLINE,
            linewidth=1.5,
            label=PL["legend_right_thal"],
        )
    ],
    loc="lower center",
    ncol=1,
    frameon=False,
    fontsize=9,
)
fig.tight_layout(rect=[0, 0.05, 1, 0.90])
if SHOW_PLOTS_INLINE:
    display(fig)
plt.close(fig)

