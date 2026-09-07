#!/usr/bin/env python3
"""Interaktive 3D-LRP-Plots analog A.10, aber mit dem Jitter-Modell.

Für das erste Holdout-Subject von IXI und UKB:
  1. Vorhersage + LRP mit dem auf gejitterten Volumes trainierten Modell
  2. Heatmap als NIfTI speichern
  3. HTML-3D-Plot (Gehirn + beide Thalamus-Masken + unnormierte LRP)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from omegaconf import OmegaConf, open_dict
from scipy.ndimage import binary_erosion

ROOT = Path("/mnt/users/andreasre/git-repos/keras-explainability")
PYBM = Path("/mnt/users/andreasre/git-repos/pyment-and1/src")
for p in (ROOT, PYBM):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from explainability import LRP, LRPStrategy  # noqa: E402
from pybrainmetrics.data.dataset import (  # noqa: E402
    _load_single_volume,
    _load_single_volume_native,
)
from pybrainmetrics.modeling.train import _build_single_device_model  # noqa: E402

# --- Pfade / Konfiguration ---------------------------------------------------
ORIG_RUN_DIR = Path(
    "/mnt/users/andreasre/data/nn-trainings/mri/Right-Whole_thalamus/"
    "training_run_21h19m18s_20aug2026"
)
JITTER_RUN_DIR = Path(
    "/mnt/users/andreasre/data/nn-trainings/mri/Right-Whole_thalamus/"
    "training_run_05h09m52s_04sep2026"
)
UKB_HOLDOUT_PREDICT_TSV = Path(
    "/mnt/users/andreasre/git-repos/pyment-and1/training_runs/input_files/mri/"
    "right_whole_thalamus/predict.tsv"
)
DATA_ROOT = Path("/mnt/ceph/data")
DATASET_DIRS = {"ixi": DATA_ROOT / "ixi", "ukb": DATA_ROOT / "ukb"}
DATASETS = ["ixi", "ukb"]

OUT_DIR = (
    ROOT
    / "output"
    / "notebooks"
    / "analysis_LRP_for_right_thalamus_volume_based_on_CNN_prediction"
)
HEATMAPS_DIR = ORIG_RUN_DIR / "heatmaps_jitter_model"
DOC_DIR = ROOT / "doc" / "notebooks" / "analysis_LRP_for_right_thalamus"

COLOR_LEFT = "rgb(153, 51, 204)"
COLOR_RIGHT = "rgb(38, 166, 64)"
COLOR_BRAIN = "rgb(170, 170, 170)"
MAX_BRAIN_POINTS = 8_000
MAX_LRP_POINTS = 10_000


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


def load_first_subject(dataset_id: str, pred_var: str) -> pd.Series:
    """Erstes Subject wie im Notebook (A.4 / N_SUBJECTS-Reihenfolge)."""
    # Bevorzugt die bereits vom Notebook geschriebenen Label-TSVs (identische Reihenfolge).
    for n in (5, 1):
        tsv = ORIG_RUN_DIR / f"{dataset_id}_predict_labels_n{n}.tsv"
        if tsv.is_file():
            df = pd.read_csv(tsv, sep=None, engine="python")
            df = _normalize_label_columns(df, pred_var)
            return df.iloc[0]

    if dataset_id == "ukb":
        df = pd.read_csv(UKB_HOLDOUT_PREDICT_TSV, sep=None, engine="python")
        df = _normalize_label_columns(df, pred_var)
        return df.dropna(subset=[pred_var, "filepath"]).iloc[0]

    dataset_dir = DATASET_DIRS[dataset_id]
    for name in ("subjects_dl_input.tsv", "participants_dl_input.tsv"):
        cand = dataset_dir / name
        if cand.is_file():
            df = pd.read_csv(cand, sep=None, engine="python")
            df = _normalize_label_columns(df, pred_var)
            return df.dropna(subset=[pred_var, "filepath"]).iloc[0]

    vol_path = dataset_dir / "T1stats" / "ThalamicNuclei.volumes.txt_concat.stats"
    vols = pd.read_csv(vol_path, sep=r"\s+")
    rows = []
    for _, row in vols.iterrows():
        sid = str(row["Subject"])
        cropped = dataset_dir / "recon" / sid / "mri" / "cropped.nii.gz"
        if cropped.is_file():
            rows.append(
                {
                    "filepath": str(cropped),
                    "participant_id": sid,
                    "subject-id": sid,
                    "Right-Whole_thalamus": float(row["Right-Whole_thalamus"]),
                }
            )
    df = _normalize_label_columns(pd.DataFrame(rows), pred_var)
    return df.iloc[0]


def load_volume(path: str, loader: str, norm_factor: float) -> np.ndarray:
    if loader == "nifti-native":
        vol = _load_single_volume_native(path, norm_factor)
    else:
        vol = _load_single_volume(path, norm_factor)
    if vol.ndim == 3:
        vol = np.expand_dims(vol, axis=-1)
    return vol.astype(np.float32)


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine=ref.affine, header=header), str(out_path))


def _load_vol(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32).squeeze()


def _surface_xyz(mask: np.ndarray, *, step: int = 1, max_points: int | None = None):
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


def _fix_plotly_umd(html: str) -> str:
    return html.replace(
        "root.moduleName = factory();",
        "root.Plotly = factory();",
        1,
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
    save_html: Path,
    model_label: str = "Jitter-Modell",
) -> Path:
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
        f"<b>{dataset_id.upper()} · {subject_id}</b> · {model_label}<br>"
        f"3D-Gehirn mit FreeSurfer-Thalamus-Masken und unnormierter LRP-Relevanz<br>"
        f"true {pred_var_name} = {true_s} · Vorhersage = {pred_s} · "
        f"Top-{hv.size} LRP-Voxel nach |R|"
    )

    fig = go.Figure()
    if bx.size:
        fig.add_trace(
            go.Scatter3d(
                x=bx, y=by, z=bz, mode="markers",
                name="Gehirnkontur (cropped T1)",
                marker=dict(size=1.4, color=COLOR_BRAIN, opacity=0.12),
                hoverinfo="skip", legendgroup="brain",
            )
        )
    if lx.size >= 4:
        fig.add_trace(
            go.Mesh3d(
                x=lx, y=ly, z=lz, alphahull=0,
                name="linker Thalamus (aseg 10)",
                color=COLOR_LEFT, opacity=0.28, flatshading=True,
                hovertemplate="linker Thalamus<br>x=%{x:.0f} y=%{y:.0f} z=%{z:.0f}<extra></extra>",
                legendgroup="left", showlegend=True,
            )
        )
    if rx.size >= 4:
        fig.add_trace(
            go.Mesh3d(
                x=rx, y=ry, z=rz, alphahull=0,
                name="rechter Thalamus (aseg 49)",
                color=COLOR_RIGHT, opacity=0.28, flatshading=True,
                hovertemplate="rechter Thalamus<br>x=%{x:.0f} y=%{y:.0f} z=%{z:.0f}<extra></extra>",
                legendgroup="right", showlegend=True,
            )
        )
    if hv.size:
        fig.add_trace(
            go.Scatter3d(
                x=hx, y=hy, z=hz, mode="markers",
                name="LRP-Relevanz (unnormiert)",
                marker=dict(
                    size=2.2, color=hv, colorscale="RdBu_r",
                    cmin=-vmax, cmax=vmax, opacity=0.75,
                    colorbar=dict(
                        title=dict(text="LRP-Relevanz<br>(unnormiert)", side="right"),
                        thickness=18, len=0.65, x=1.02,
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
            title=dict(text="Legende"), itemsizing="constant",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.15)", borderwidth=1,
            x=0.02, y=0.98, xanchor="left", yanchor="top",
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
        width=980, height=780,
        margin=dict(l=0, r=80, t=90, b=10),
        hovermode="closest",
    )

    save_html.parent.mkdir(parents=True, exist_ok=True)
    save_html.write_text(
        _fix_plotly_umd(fig.to_html(include_plotlyjs=True, full_html=True)),
        encoding="utf-8",
    )
    print("3D-HTML:", save_html)
    return save_html


def main() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    jitter_cfg = OmegaConf.load(JITTER_RUN_DIR / "config.yaml")
    pred_var = jitter_cfg.data.prediction_variable
    with open_dict(jitter_cfg):
        jitter_cfg.paths.csv_dir = str(JITTER_RUN_DIR)
        if "prediction" not in jitter_cfg.training:
            jitter_cfg.training.prediction = {}
        jitter_cfg.training.prediction.batch_size = 1

    model = _build_single_device_model(jitter_cfg)
    w0 = model.get_weights()[0].copy()
    model.load_weights(str(JITTER_RUN_DIR / "model.keras"))
    delta = float(np.mean(np.abs(model.get_weights()[0] - w0)))
    if delta < 1e-9:
        raise RuntimeError("Jitter-Modell-Gewichte wurden nicht geladen.")
    print(f"Gewichtsdelta: {delta:.3g}")

    norm_factor = float(jitter_cfg.preprocessing.normalization_factor)
    loader = str(getattr(jitter_cfg.data, "loader", "nifti-nibabel")).lower()

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
    print(f"LRP-Schichten: {len(lrp.layers)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for dataset_id in DATASETS:
        row = load_first_subject(dataset_id, pred_var)
        sid = str(row["participant_id"])
        t1_path = Path(str(row["filepath"]))
        y_true = float(row[pred_var])
        if not t1_path.is_file():
            raise FileNotFoundError(f"[{dataset_id}/{sid}] Volume fehlt: {t1_path}")

        # Masken aus Teil A wiederverwenden (identische Anatomie).
        mask_dir = ORIG_RUN_DIR / "heatmaps" / dataset_id / sid
        left_path = mask_dir / "aseg_mni152_left_thalamus_cropped.nii.gz"
        right_path = mask_dir / "aseg_mni152_right_thalamus_cropped.nii.gz"
        for p in (left_path, right_path):
            if not p.is_file():
                raise FileNotFoundError(f"[{dataset_id}/{sid}] Maske fehlt: {p}")

        print(f"\n=== {dataset_id}/{sid} ===")
        vol = load_volume(str(t1_path), loader, norm_factor)
        x = np.expand_dims(vol, 0)
        y_pred = float(np.squeeze(model.predict(x, verbose=0)))
        R_raw = lrp(x)[0].numpy()
        R_masked = mask_explanation(vol, R_raw)
        print(
            f"true={y_true:.1f}  pred={y_pred:.1f}  "
            f"sum_R_raw={float(np.sum(R_raw)):.1f}  sum_R_masked={float(np.sum(R_masked)):.1f}"
        )

        hm_path = HEATMAPS_DIR / dataset_id / sid / f"lrp_heatmap_jitter_model_{dataset_id}_{sid}.nii.gz"
        save_heatmap_nifti(R_masked, str(t1_path), hm_path)
        print("Heatmap:", hm_path)

        html_name = f"{dataset_id}_{sid}_thalamus_lrp_3d_jitter_model.html"
        html_out = OUT_DIR / html_name
        plot_thalamus_lrp_3d(
            dataset_id=dataset_id,
            subject_id=sid,
            volume=_load_vol(t1_path),
            heatmap=R_masked,
            left_mask=_load_vol(left_path),
            right_mask=_load_vol(right_path),
            y_true=y_true,
            y_pred=y_pred,
            pred_var_name=pred_var,
            save_html=html_out,
            model_label="Jitter-Modell (training_run_05h09m52s_04sep2026)",
        )
        # Kopie neben den Bericht, damit relative Links im README greifen.
        doc_copy = DOC_DIR / html_name
        doc_copy.write_bytes(html_out.read_bytes())
        print("Doc-Kopie:", doc_copy)

        results.append(
            {
                "dataset_id": dataset_id,
                "subject_id": sid,
                "y_true": y_true,
                "y_pred": y_pred,
                "heatmap": str(hm_path),
                "html": str(html_out),
            }
        )

    print("\n=== fertig ===")
    for r in results:
        print(
            f"{r['dataset_id']}/{r['subject_id']}: "
            f"true={r['y_true']:.1f} pred={r['y_pred']:.1f}\n  {r['html']}"
        )


if __name__ == "__main__":
    main()
