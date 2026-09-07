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

# %% [markdown]
# <a id="top"></a>
# # Relevanzerhaltung in LRP prüfen
#
# ## Worum geht es?
#
# Layer-wise Relevance Propagation (LRP) soll Relevanz **weder erzeugen noch vernichten**,
# sondern nur umverteilen. Die zentrale Bilanz lautet:
#
# $$\sum_j R_j^{(\ell)} \;=\; \sum_k R_k^{(\ell+1)} \qquad\text{für jede Schicht }\ell$$
#
# und am Ende idealerweise
#
# $$\sum_i R_i^{(\mathrm{input})} \;=\; f_c(x)$$
#
# also gleich dem **Logit** der erklärten Klasse $c$ (Softmax wird im Erklärer entfernt).
#
# Dieses Notebook prüft genau das am trainierten **3D-MNIST-CNN** aus
# [`Train_and_explain_3D_mnist_model`](Train_and_explain_3D_mnist_model.py):
#
# 1. Relevanzsumme **je Schicht** des Rückwärtspfads (Composite-LRP)
# 2. Derselbe Check mit **einer einheitlichen Regel** (LRP-αβ)
# 3. Schicht-zu-Schicht-Lecks und Strategievergleich
# 4. zusätzliche Plausibilitätschecks (Vorzeichen, Konzentration, Stichproben)
#
# ## Warum das wichtig ist
#
# Wenn die Summe über die Schichten hinweg springt, liegt oft eines davon vor:
#
# * Bias-Terme (Relevanz ohne Bildbezug wird verworfen)
# * ε-Stabilisierung (Nenner wird absichtlich vergrößert)
# * flat-/b-Regeln ($a\leftarrow 1$), die streng genommen nicht erhaltend sind
# * Implementierungsfehler in einer LRP-Schicht
#
# > **Faustregel:** Abweichungen von einigen Prozent bis wenigen Zehnteln sind bei Composite-LRP
# > normal. Sprünge um Größenordnungen deuten auf einen Bug hin.
#
# ## Ablauf
#
# ```text
#   Daten + fertiges 3D-CNN laden
#            │
#            ▼
#   Fall A: Composite-LRP  →  ΣR je Schicht
#            │
#            ▼
#   Fall B: einheitlich αβ (α=2,β=1)  →  ΣR je Schicht + Vergleich
#            │
#            ▼
#   Weitere Strategien, Stichproben, Konzentration
# ```
#
# ---
#
# <a id="toc"></a>
# ## Inhaltsverzeichnis
#
# | # | Abschnitt | Inhalt |
# |---|---|---|
# | 1 | [Setup, Daten, Modell](#sec-01) | 3D-MNIST, fertiges `.keras`-Modell |
# | 2 | [Composite-LRP wie im Vorbild](#sec-02) | Strategie, Zielklasse, Logit |
# | 3 | [Schichtweise Relevanzsummen](#sec-03) | Hauptdiagnose + Plots (Composite) |
# | 4 | [Lecks visualisieren](#sec-04) | ΔΣR und relative Änderung |
# | 5 | [Einheitliche Strategie: LRP-αβ](#sec-05) | eine Regel für alle Schichten |
# | 6 | [Strategievergleich](#sec-06) | Composite / ε / αβ / Bias-Flags |
# | 7 | [Zusatz: Vorzeichen & Konzentration](#sec-07) | pos/neg Masse, räumliche Schärfe |
# | 8 | [Zusatz: Stichproben & Klassen](#sec-08) | Erhaltungsquote über Beispiele |
# | 9 | [Fazit](#sec-09) | Was man mitnehmen sollte |

# %% [markdown]
# <a id="sec-01"></a>
# ## 1. Setup, Daten, Modell
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Wir nutzen dieselbe Datenquelle und dasselbe trainierte Netz wie das
# 3D-MNIST-Notebook. Neu trainiert wird hier nichts.

# %%
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ipynbname


def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / "explainability").is_dir():
            return candidate
    return p


repo_root = find_repo_root()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _process_ancestors(pid: int, levels: int = 5) -> list[int]:
    pids = []
    for _ in range(levels):
        pids.append(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            pid = int(stat[stat.rindex(")") + 1:].split()[1])
        except Exception:
            break
        if pid <= 1:
            break
    return pids


def find_notebook_name() -> str:
    try:
        return ipynbname.name()
    except Exception:
        pass

    for candidate in (os.environ.get("JPY_SESSION_NAME"), globals().get("__session__")):
        if candidate and str(candidate).endswith(".ipynb"):
            return Path(candidate).stem

    if "__file__" in globals():
        return Path(globals()["__file__"]).stem

    start_pid = int(os.environ.get("JPY_PARENT_PID") or os.getppid())
    for pid in _process_ancestors(start_pid):
        try:
            args = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        except Exception:
            continue
        for arg in args:
            if arg.endswith(".ipynb"):
                return Path(arg).stem

    return "check_relevance_conservation_in_LRP"


notebook_name = os.environ.get("NOTEBOOK_NAME") or find_notebook_name()
target_dir = repo_root / "output" / "notebooks" / notebook_name
target_dir.mkdir(parents=True, exist_ok=True)

print(f"Notebook-Name: {notebook_name}")
print(f"Zielordner:    {target_dir}")

data_path = repo_root / "data" / "3d-mnist" / "full_dataset_vectors.h5"
assert data_path.is_file(), (
    "3D-MNIST fehlt — bitte von https://www.kaggle.com/daavoo/3d-mnist laden "
    f"nach {data_path}"
)

with h5py.File(data_path, "r") as f:
    train_X = np.reshape(f["X_train"][:], (-1, 16, 16, 16, 1)).astype(np.float32)
    train_y = np.asarray(f["y_train"][:])
    test_X = np.reshape(f["X_test"][:], (-1, 16, 16, 16, 1)).astype(np.float32)
    test_y = np.asarray(f["y_test"][:])

train_X = train_X[:, ::-1, :, :]
test_X = test_X[:, ::-1, :, :]

from tensorflow.keras.models import load_model

MODEL_PATH = (
    repo_root
    / "output"
    / "notebooks"
    / "Train_and_explain_3D_mnist_model"
    / "100_epochs"
    / "3d_mnist_cnn.keras"
)
assert MODEL_PATH.is_file(), (
    f"Trainiertes Modell fehlt: {MODEL_PATH}\n"
    "Bitte zuerst Train_and_explain_3D_mnist_model ausführen."
)

model = load_model(MODEL_PATH)
print(f"Modell geladen: {MODEL_PATH}")
print(f"Schichten im CNN: {len(model.layers)}")
model.summary()

# %% [markdown]
# <a id="sec-02"></a>
# ## 2. Composite-LRP wie im Vorbild
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Dieselbe `LRPStrategy` wie in Abschnitt 6 von `Train_and_explain_3D_mnist_model`:
#
# | Eintrag | Regel | Schicht |
# |---|---|---|
# | 1 | `b=True`, α=1, β=0 | `conv1` |
# | 2–8 | α=2, β=1 | `conv2`…`conv8` |
# | 9–10 | ε = 0,25 | `dense`, `preds` |
#
# **Wichtig für die Diagnose:** `LRP` ist ein Keras-Modell, das den **Vorwärtspfad**
# (fused, Softmax entfernt) *und* den **Rückwärtspfad** enthält. Relevanz beginnt erst
# beim Mask-Lambda (`LRP_output_mask_lambda`). Alles davor sind Aktivierungen — deren
# Summe ist *kein* Conservation-Check.

# %%
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda

from explainability import LRP, LRPStrategy
from explainability.layers import PoolingLRPLayer, StandardLRPLayer
from explainability.layers.layer import LRPLayer

IMAGE_IDX = 654
sample = train_X[IMAGE_IDX : IMAGE_IDX + 1]
true_label = int(train_y[IMAGE_IDX])

probs = model.predict(sample, verbose=0)[0]
pred_label = int(np.argmax(probs))

print(f"Beispielindex: {IMAGE_IDX}")
print(f"Label:         {true_label}")
print(f"Vorhersage:    {pred_label}  (p={probs[pred_label]:.6f})")
print(f"Top-3:         {np.argsort(probs)[::-1][:3]}")

COMPOSITE_STRATEGY = LRPStrategy(
    layers=[
        {"b": True, "alpha": 1, "beta": 0},
        {"alpha": 2, "beta": 1},
        {"alpha": 2, "beta": 1},
        {"alpha": 2, "beta": 1},
        {"alpha": 2, "beta": 1},
        {"alpha": 2, "beta": 1},
        {"alpha": 2, "beta": 1},
        {"alpha": 2, "beta": 1},
        {"epsilon": 0.25},
        {"epsilon": 0.25},
    ]
)


def build_lrp(model, idx: int, strategy: LRPStrategy | None = None, **kwargs) -> LRP:
    return LRP(
        model,
        layer=len(model.layers) - 1,
        idx=idx,
        strategy=strategy,
        **kwargs,
    )


def find_backward_start(lrp: LRP) -> int:
    """Index der Mask-Lambda = Start der Relevanzpropagation."""
    for i, layer in enumerate(lrp.layers):
        if isinstance(layer, Lambda) and "output_mask" in layer.name:
            return i
    # Fallback: erste LRPLayer
    for i, layer in enumerate(lrp.layers):
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
        if layer.b:
            parts.append("b")
        if layer.flat:
            parts.append("flat")
        if layer.ignore_bias:
            parts.append("ignore_bias")
        if layer.adjust_epsilon:
            parts.append("adjust_ε")
        return ", ".join(parts) if parts else "LRP-0"
    if isinstance(layer, PoolingLRPLayer):
        return getattr(layer, "strategy", "?")
    return "—"


def collect_layer_relevance(lrp: LRP, x: np.ndarray) -> pd.DataFrame:
    """ΣR, pos/neg-Masse und Form für jede Schicht ab dem Mask-Lambda."""
    start = find_backward_start(lrp)
    layers = list(lrp.layers[start:])
    probe = Model(lrp.input, [layer.output for layer in layers])
    tensors = probe.predict(x, verbose=0)
    if not isinstance(tensors, list):
        tensors = [tensors]

    rows = []
    for offset, (layer, R) in enumerate(zip(layers, tensors)):
        R = np.asarray(R)
        idx = start + offset
        sum_r = float(np.sum(R))
        sum_pos = float(np.sum(R[R > 0])) if R.size else 0.0
        sum_neg = float(np.sum(R[R < 0])) if R.size else 0.0
        abs_r = np.abs(R).ravel()
        total_abs = float(abs_r.sum())
        if total_abs > 0:
            order = np.sort(abs_r)[::-1]
            cdf = np.cumsum(order) / total_abs
            n90 = int(np.searchsorted(cdf, 0.9) + 1)
            frac90 = n90 / abs_r.size
            entropy = float(
                -np.sum((order / total_abs) * np.log(order / total_abs + 1e-12))
            )
        else:
            n90, frac90, entropy = 0, np.nan, np.nan

        rows.append(
            {
                "layer_idx": idx,
                "name": layer.name,
                "type": type(layer).__name__,
                "rule": layer_rule_tag(layer),
                "shape": tuple(R.shape),
                "n_elements": int(R.size),
                "sum_R": sum_r,
                "sum_pos": sum_pos,
                "sum_neg": sum_neg,
                "sum_abs": float(np.sum(np.abs(R))),
                "max_abs": float(np.max(np.abs(R))) if R.size else 0.0,
                "frac_voxels_for_90pct_abs": frac90,
                "entropy_abs": entropy,
            }
        )

    df = pd.DataFrame(rows)
    df["delta_sum_R"] = df["sum_R"].diff()
    df["rel_change"] = df["sum_R"].pct_change()
    # Bezug: Startrelevanz (maskiertes Logit)
    r0 = df["sum_R"].iloc[0]
    df["ratio_to_start"] = df["sum_R"] / r0 if r0 != 0 else np.nan
    return df


lrp = build_lrp(model, pred_label, COMPOSITE_STRATEGY)
print(f"LRP-Schichten gesamt: {len(lrp.layers)}")
print(f"Rückwärtspfad ab Index: {find_backward_start(lrp)}")
print(f"Erklärer-Ausgabe ΣR: {float(np.sum(lrp.predict(sample, verbose=0))):.6f}")

# %% [markdown]
# <a id="sec-03"></a>
# ## 3. Schichtweise Relevanzsummen
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Für jede Schicht des Rückwärtspfads bauen wir ein Teilmodell
# `Model(lrp.input, layer.output)` (hier gebündelt als Multi-Output) und summieren
# die Relevanz. Idealerweise ist die Kurve **flach**.

# %%
df = collect_layer_relevance(lrp, sample)
csv_path = target_dir / "layerwise_relevance_sums_composite.csv"
df.to_csv(csv_path, index=False)
print(f"Tabelle gespeichert: {csv_path}\n")

display_cols = [
    "layer_idx",
    "type",
    "name",
    "rule",
    "sum_R",
    "delta_sum_R",
    "ratio_to_start",
    "shape",
]
print(df[display_cols].to_string(index=False, float_format=lambda v: f"{v:10.6f}"))

r_start = float(df["sum_R"].iloc[0])
r_end = float(df["sum_R"].iloc[-1])
print("\n--- Kurzbilanz ---")
print(f"Start (maskiertes Logit): {r_start:.6f}")
print(f"Ende  (Input-Relevanz):   {r_end:.6f}")
print(f"Erhalten:                 {100.0 * r_end / r_start:.2f} %" if r_start else "n/a")
print(f"Absolutverlust:           {r_end - r_start:.6f}")

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

ax = axes[0]
ax.plot(df["layer_idx"], df["sum_R"], "o-", color="C0", lw=2, markersize=5)
ax.axhline(r_start, color="C1", ls="--", lw=1.5, label=f"Start ΣR = {r_start:.3f}")
ax.axhline(r_end, color="C2", ls=":", lw=1.5, label=f"Ende ΣR = {r_end:.3f}")
ax.set_ylabel(r"$\sum R$")
ax.set_title(
    f"Relevanzsumme je LRP-Schicht — Beispiel {IMAGE_IDX}, "
    f"Klasse {pred_label} (Composite-LRP)"
)
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

# Marker für gewichtstragende / Pooling-Schichten
for _, row in df.iterrows():
    if row["type"] in {
        "DenseLRP",
        "Conv3DLRP",
        "Conv2DLRP",
        "MaxPoolingLRP",
        "AveragePoolingLRP",
    }:
        ax.annotate(
            row["type"].replace("LRP", ""),
            (row["layer_idx"], row["sum_R"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=7,
            rotation=45,
            alpha=0.8,
        )

ax = axes[1]
colors = ["C3" if (pd.notna(d) and abs(d) > 1e-3) else "0.6" for d in df["delta_sum_R"]]
ax.bar(df["layer_idx"], df["delta_sum_R"].fillna(0.0), color=colors, width=0.8)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Schichtindex im LRP-Modell")
ax.set_ylabel(r"$\Delta\sum R$ zur Vorgängerschicht")
ax.set_title("Schicht-zu-Schicht-Änderung (Lecks hervorgehoben)")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
fig_path = target_dir / "01_sum_R_per_layer.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Abbildung: {fig_path}")

# %% [markdown]
# ### Lesart
#
# * **Flache Abschnitte** (ReLU, NoOp, Pooling, Conv mit αβ): Relevanz bleibt erhalten.
# * **Sprünge an DenseLRP mit ε**: erwarteter Verlust durch Bias und/oder ε-Stabilisierung.
# * Die **b-Regel** an `conv1` darf die Summe leicht verschieben; große Sprünge wären verdächtig.

# %% [markdown]
# <a id="sec-04"></a>
# ## 4. Lecks im Detail
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Welche Schichten verursachen den größten Absolutverlust?

# %%
leak_df = (
    df.assign(abs_delta=lambda d: d["delta_sum_R"].abs())
    .sort_values("abs_delta", ascending=False)
    .head(10)
)

print("Top-10 Schichten nach |ΔΣR|:\n")
print(
    leak_df[
        ["layer_idx", "type", "name", "rule", "sum_R", "delta_sum_R", "ratio_to_start"]
    ].to_string(index=False, float_format=lambda v: f"{v:10.6f}")
)

fig, ax = plt.subplots(figsize=(12, 5))
plot_df = df.dropna(subset=["delta_sum_R"]).copy()
ax.plot(plot_df["layer_idx"], plot_df["ratio_to_start"], "o-", lw=2)
ax.axhline(1.0, color="C1", ls="--", label="100 % der Startrelevanz")
ax.set_xlabel("Schichtindex")
ax.set_ylabel(r"$\sum R \;/\; \sum R_{\mathrm{start}}$")
ax.set_title("Erhaltungsquote relativ zum maskierten Logit")
ax.set_ylim(0, max(1.05, float(plot_df["ratio_to_start"].max()) * 1.05))
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
fig_path = target_dir / "02_conservation_ratio_along_layers.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Abbildung: {fig_path}")

# %% [markdown]
# <a id="sec-05"></a>
# ## 5. Einheitliche Strategie: LRP-αβ für alle Schichten
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Warum überhaupt ein zweiter Fall?
#
# Composite-LRP mischt **drei** Mechanismen (`b`, αβ, ε). Wenn ΣR springt, ist unklar,
# ob das an der Implementierung einer Schicht liegt oder an der bewussten Regelwahl
# (ε und Bias „schlucken" Relevanz). Ein zweiter Lauf mit **derselben Regel überall**
# trennt diese Effekte: bleibe die Kurve flach, war das Composite-Leck regelbedingt;
# springe sie trotzdem, liegt ein Bug näher.
#
# ### Wahl der einheitlichen Regel: LRP-αβ mit $\alpha=2$, $\beta=1$
#
# Kandidaten für „eine Regel für alles":
#
# | Regel | Erhaltung | Stabilität | Eignung als *einzige* Regel |
# |---|---|---|---|
# | **LRP-0** ($\varepsilon=0$) | gut (ohne Bias) | schlecht — Nenner kann $\approx 0$ werden | zu rauschig über 10 Schichten |
# | **LRP-ε** ($\varepsilon=0{,}25$) | **absichtlich undicht** | gut | ungeeignet als Conservation-Referenz |
# | **LRP-ε + `adjust_epsilon`** | Quote $\approx 1$ erzwungen | gut | erzwingt Erhaltung nachträglich, testet die Propagation nicht echt |
# | **b / flat** ($a\leftarrow 1$) | nicht streng erhaltend | glättet | erzeugt Relevanz auf Null-Voxeln |
# | **LRP-αβ** ($\alpha=2$, $\beta=1$) | **ja**, solange $\alpha-\beta=1$ | sehr gut | klare Pro-/Contra-Trennung, Standard in der Literatur |
#
# **Entscheidung:** einheitlich **LRP-αβ mit $\alpha=2$, $\beta=1$**.
#
# Begründung in Kurzform:
#
# 1. **Conservation by design (im Idealfall).** Die Nebenbedingung $\alpha = \beta + 1$
#    (im Code erzwungen) soll positive und negative Beiträge so gewichten, dass die
#    Nettorelevanz erhalten bleibt — anders als bei ε, das den Nenner absichtlich aufbläht.
#    In der Praxis können Bias und die konkrete Dense-αβ-Implementierung das brechen;
#    genau deshalb messen wir schichtweise nach.
# 2. **Numerische Stabilität.** αβ trennt $z^{+}$ und $z^{-}$ und vermeidet die
#    Auslöschungs-Singularitäten von LRP-0; deshalb eignet sich die Regel auch für den
#    Klassifikationskopf, nicht nur für Conv-Blöcke.
# 3. **Fairer Vergleich zum Composite.** Im Composite stecken bereits sieben αβ-Einträge;
#    der einheitliche Lauf zeigt, was passiert, wenn man die beiden Dense-ε-Schichten und
#    die b-Regel an `conv1` *ebenfalls* durch αβ ersetzt — also den Einfluss genau dieser
#    Sonderregeln misst.
# 4. **Keine Nachskalierung.** Im Gegensatz zu `adjust_epsilon` wird die Summe nicht
#    nachträglich auf den Startwert zurückgezogen; Abweichungen bleiben sichtbar.
#
# Verbleibendes / typisches Leck: **Bias-Terme** und die αβ-Normalisierung in `DenseLRP`.
# Beides ist diagnostisch wertvoll — kein Grund, die Regel als Vergleichsfall zu verwerfen.
#
# ### API-Hinweis
#
# `LRPStrategy(layers=[...])` verlangt **einen Dict-Eintrag pro gewichtstragender Schicht**
# (hier 8× Conv3D + 2× Dense = 10). „Eine Strategie" heißt deshalb: **dieselbe Regel
# zehnmal**, nicht ein einzelnes Listenelement — sonst schlägt die Längenprüfung fehl.

# %%
UNIFORM_ALPHABETA_STRATEGY = LRPStrategy(
    layers=[{"alpha": 2, "beta": 1}] * 10
)

lrp_ab = build_lrp(model, pred_label, UNIFORM_ALPHABETA_STRATEGY)
df_ab = collect_layer_relevance(lrp_ab, sample)

csv_path_ab = target_dir / "layerwise_relevance_sums_uniform_alphabeta.csv"
df_ab.to_csv(csv_path_ab, index=False)

r_start_ab = float(df_ab["sum_R"].iloc[0])
r_end_ab = float(df_ab["sum_R"].iloc[-1])

print("Einheitliche LRP-αβ (α=2, β=1) — schichtweise ΣR:\n")
print(
    df_ab[
        [
            "layer_idx",
            "type",
            "name",
            "rule",
            "sum_R",
            "delta_sum_R",
            "ratio_to_start",
            "shape",
        ]
    ].to_string(index=False, float_format=lambda v: f"{v:10.6f}")
)

print("\n--- Kurzbilanz αβ ---")
print(f"Start (maskiertes Logit): {r_start_ab:.6f}")
print(f"Ende  (Input-Relevanz):   {r_end_ab:.6f}")
print(
    f"Erhalten:                 {100.0 * r_end_ab / r_start_ab:.2f} %"
    if r_start_ab
    else "n/a"
)
print(f"Absolutverlust:           {r_end_ab - r_start_ab:.6f}")
print(f"Tabelle: {csv_path_ab}")

# %%
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

ax = axes[0]
ax.plot(
    df["layer_idx"],
    df["sum_R"],
    "o--",
    color="0.55",
    lw=1.5,
    markersize=4,
    label=f"Composite (Ende ΣR={r_end:.3f})",
)
ax.plot(
    df_ab["layer_idx"],
    df_ab["sum_R"],
    "o-",
    color="C0",
    lw=2,
    markersize=5,
    label=f"einheitlich αβ (Ende ΣR={r_end_ab:.3f})",
)
ax.axhline(r_start_ab, color="C1", ls="--", lw=1.5, label=f"Start ΣR = {r_start_ab:.3f}")
ax.set_ylabel(r"$\sum R$")
ax.set_title(
    f"ΣR je Schicht — Composite vs. einheitlich αβ "
    f"(Beispiel {IMAGE_IDX}, Klasse {pred_label})"
)
ax.grid(True, alpha=0.3)
ax.legend(loc="best")

ax = axes[1]
ax.bar(
    df_ab["layer_idx"],
    df_ab["delta_sum_R"].fillna(0.0),
    color=[
        "C3" if (pd.notna(d) and abs(d) > 1e-3) else "0.6"
        for d in df_ab["delta_sum_R"]
    ],
    width=0.8,
)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Schichtindex im LRP-Modell")
ax.set_ylabel(r"$\Delta\sum R$ (nur αβ-Lauf)")
ax.set_title("Schicht-zu-Schicht-Änderung bei einheitlicher αβ-Regel")
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
fig_path = target_dir / "02b_sum_R_per_layer_uniform_alphabeta.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Abbildung: {fig_path}")

# %%
leak_df_ab = (
    df_ab.assign(abs_delta=lambda d: d["delta_sum_R"].abs())
    .sort_values("abs_delta", ascending=False)
    .head(10)
)
print("Top-10 |ΔΣR| bei einheitlicher αβ-Regel:\n")
print(
    leak_df_ab[
        ["layer_idx", "type", "name", "rule", "sum_R", "delta_sum_R", "ratio_to_start"]
    ].to_string(index=False, float_format=lambda v: f"{v:10.6f}")
)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(
    df["layer_idx"],
    df["ratio_to_start"],
    "o--",
    color="0.55",
    lw=1.5,
    label="Composite",
)
ax.plot(
    df_ab["layer_idx"],
    df_ab["ratio_to_start"],
    "o-",
    color="C0",
    lw=2,
    label="einheitlich αβ",
)
ax.axhline(1.0, color="C1", ls="--", label="100 % der Startrelevanz")
ax.set_xlabel("Schichtindex")
ax.set_ylabel(r"$\sum R \;/\; \sum R_{\mathrm{start}}$")
ax.set_title("Erhaltungsquote: Composite vs. einheitlich αβ")
ax.set_ylim(
    0,
    max(
        1.05,
        float(df["ratio_to_start"].max()),
        float(df_ab["ratio_to_start"].max()),
    )
    * 1.05,
)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
fig_path = target_dir / "02c_conservation_ratio_composite_vs_alphabeta.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Abbildung: {fig_path}")

print("\n--- Vergleich auf einen Blick ---")
print(
    f"Composite:     Start={r_start:.4f}  Ende={r_end:.4f}  "
    f"Quote={100.0 * r_end / r_start:.2f} %"
)
print(
    f"einheitlich αβ: Start={r_start_ab:.4f}  Ende={r_end_ab:.4f}  "
    f"Quote={100.0 * r_end_ab / r_start_ab:.2f} %"
)

# %% [markdown]
# ### Erwartete Lesart dieses Abschnitts
#
# * Die **Start-ΣR** (maskiertes Logit) ist in beiden Läufen gleich — dieselbe Klasse, dasselbe $x$.
# * Unter reinem αβ sollten **Pooling/ReLU/NoOp** flach bleiben; Sprünge markieren
#   gewichtstragende Schichten.
# * **Wichtiges Diagnose-Ergebnis:** In dieser Codebasis kann bereits die erste
#   `DenseLRP`-αβ-Schicht ΣR stark verändern (oft Richtung $\alpha\cdot R$, wenn der
#   negative Anteil / Bias die $\beta$-Kompensation nicht ausgleicht). Die Regel
#   $\alpha-\beta=1$ garantiert Erhaltung nur, wenn die positive und negative
#   Massen korrekt normalisiert werden — Bias im Nenner ohne zugehöriges Neuron
#   und die Dense-αβ-Implementierung können das brechen.
# * Composite wirkt hier oft „konservativer", weil ε Relevanz **dämpft**; αβ kann
#   sie dagegen **aufblasen**. Beides sind Regel-/Implementierungseffekte, keine
#   Garantie für eine flache Kurve.
# * Springen αβ *und* Composite in derselben späteren Conv-Schicht, verdient genau
#   diese Schicht eine Code-Prüfung.

# %% [markdown]
# <a id="sec-06"></a>
# ## 6. Strategievergleich
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Dieselbe Probe, dieselbe Zielklasse — unterschiedliche Regeln. Wir vergleichen nur
# Start-ΣR (maskiertes Logit), End-ΣR (Input) und die Erhaltungsquote. Zusätzlich
# prüfen wir `ignore_bias` und `adjust_epsilon`, die Lecks theoretisch schließen können.
# Die einheitliche αβ-Strategie aus Abschnitt 5 erscheint hier erneut im Überblick.

# %%
STRATEGIES = {
    "Composite (Notebook)": COMPOSITE_STRATEGY,
    "einheitlich αβ (Abschn. 5)": UNIFORM_ALPHABETA_STRATEGY,
    "nur ε=0.25": LRPStrategy(layers=[{"epsilon": 0.25}] * 10),
    "ε=0.25 + adjust_ε": LRPStrategy(
        layers=[{"epsilon": 0.25, "adjust_epsilon": True}] * 10
    ),
    "ε=0.25 + ignore_bias": LRPStrategy(
        layers=[{"epsilon": 0.25, "ignore_bias": True}] * 10
    ),
    "ε=0 + ignore_bias (nahe LRP-0)": LRPStrategy(
        layers=[{"epsilon": 0.0, "ignore_bias": True}] * 10
    ),
}


def summarize_strategy(name: str, strategy: LRPStrategy, x: np.ndarray, idx: int) -> dict:
    explainer = build_lrp(model, idx, strategy)
    layer_df = collect_layer_relevance(explainer, x)
    r0 = float(layer_df["sum_R"].iloc[0])
    r1 = float(layer_df["sum_R"].iloc[-1])
    # größter Einzelsprung
    biggest = layer_df.loc[layer_df["delta_sum_R"].abs().idxmax()]
    return {
        "strategy": name,
        "sum_R_start": r0,
        "sum_R_input": r1,
        "ratio": (r1 / r0) if r0 != 0 else np.nan,
        "abs_loss": r1 - r0,
        "worst_layer": f"{biggest['type']} ({biggest['name']})",
        "worst_delta": float(biggest["delta_sum_R"]),
        "layer_df": layer_df,
    }


strategy_rows = []
strategy_curves = {}
for name, strategy in STRATEGIES.items():
    print(f"Berechne: {name} ...")
    summary = summarize_strategy(name, strategy, sample, pred_label)
    strategy_curves[name] = summary.pop("layer_df")
    strategy_rows.append(summary)

strategy_df = pd.DataFrame(strategy_rows)
strategy_csv = target_dir / "strategy_conservation_comparison.csv"
strategy_df.to_csv(strategy_csv, index=False)
print("\n", strategy_df.to_string(index=False, float_format=lambda v: f"{v:10.6f}"))
print(f"\nGespeichert: {strategy_csv}")

fig, ax = plt.subplots(figsize=(12, 5))
x_pos = np.arange(len(strategy_df))
bars = ax.bar(x_pos, strategy_df["ratio"], color="C0", alpha=0.85)
ax.axhline(1.0, color="C3", ls="--", label="perfekte Erhaltung")
ax.set_xticks(x_pos)
ax.set_xticklabels(strategy_df["strategy"], rotation=25, ha="right")
ax.set_ylabel(r"$\sum R_{\mathrm{input}} \;/\; \sum R_{\mathrm{start}}$")
ax.set_title(f"Erhaltungsquote je Strategie (Beispiel {IMAGE_IDX}, Klasse {pred_label})")
ax.set_ylim(0, max(1.1, float(strategy_df["ratio"].max()) * 1.1))
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
for bar, val in zip(bars, strategy_df["ratio"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.2%}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
plt.tight_layout()
fig_path = target_dir / "03_strategy_conservation_ratios.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Abbildung: {fig_path}")

fig, ax = plt.subplots(figsize=(13, 6))
for name, curve in strategy_curves.items():
    ax.plot(
        curve["layer_idx"],
        curve["ratio_to_start"],
        "o-",
        lw=1.8,
        markersize=3.5,
        label=name,
    )
ax.axhline(1.0, color="k", ls="--", lw=1)
ax.set_xlabel("Schichtindex")
ax.set_ylabel(r"$\sum R \;/\; \sum R_{\mathrm{start}}$")
ax.set_title("Erhaltungsverlauf über die Schichten — Strategievergleich")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="best")
plt.tight_layout()
fig_path = target_dir / "04_strategy_conservation_curves.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Abbildung: {fig_path}")

# %% [markdown]
# <a id="sec-07"></a>
# ## 7. Zusatz: Vorzeichenmasse und räumliche Konzentration
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Conservation betrifft die **nette Summe**. Zusätzlich lohnt sich:
#
# * **positive vs. negative Masse** — αβ kann beide groß halten, während die Nettosumme stabil bleibt
# * **Konzentration** — wie viele Elemente tragen 90 % der $|R|$? Grobe Pooling-Raster zeigen sich hier

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(df["layer_idx"], df["sum_pos"], "o-", color="C3", label=r"$\sum R_+$")
ax.plot(df["layer_idx"], df["sum_neg"], "o-", color="C0", label=r"$\sum R_-$")
ax.plot(df["layer_idx"], df["sum_R"], "s--", color="k", label=r"$\sum R$ (netto)")
ax.set_xlabel("Schichtindex")
ax.set_ylabel("Relevanzsumme")
ax.set_title("Positive / negative / netto Relevanz")
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[1]
ax.plot(
    df["layer_idx"],
    df["frac_voxels_for_90pct_abs"],
    "o-",
    color="C4",
)
ax.set_xlabel("Schichtindex")
ax.set_ylabel("Anteil Elemente für 90 % von $|R|$")
ax.set_title("Räumliche Konzentration der Relevanz")
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = target_dir / "05_sign_mass_and_concentration.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Abbildung: {fig_path}")

# Heatmap der Input-Relevanz (Mittelschnitte) — Plausibilität
R_input = np.asarray(lrp.predict(sample, verbose=0))[0, ..., 0]
vol = sample[0, ..., 0]
mid = [s // 2 for s in vol.shape]

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
planes = [
    ("axial (z)", vol[:, :, mid[2]], R_input[:, :, mid[2]]),
    ("koronal (y)", vol[:, mid[1], :], R_input[:, mid[1], :]),
    ("sagittal (x)", vol[mid[0], :, :], R_input[mid[0], :, :]),
]
vmax = float(np.max(np.abs(R_input))) or 1.0
for col, (title, img, heat) in enumerate(planes):
    axes[0, col].imshow(img.T, origin="lower", cmap="Greys")
    axes[0, col].set_title(f"Input — {title}")
    axes[0, col].axis("off")
    im = axes[1, col].imshow(
        heat.T, origin="lower", cmap="seismic", vmin=-vmax, vmax=vmax
    )
    axes[1, col].set_title(f"LRP — {title}")
    axes[1, col].axis("off")
fig.colorbar(im, ax=axes[1, :].ravel().tolist(), fraction=0.02, pad=0.02)
fig.suptitle(
    f"Input vs. Relevanz (Klasse {pred_label}, ΣR={float(np.sum(R_input)):.3f})",
    y=1.02,
)
plt.tight_layout()
fig_path = target_dir / "06_input_vs_relevance_slices.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Abbildung: {fig_path}")

# %% [markdown]
# <a id="sec-08"></a>
# ## 8. Zusatz: Stichproben und alle Zielklassen
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### 8.1 Erhaltungsquote für alle 10 Klassen desselben Volumens
#
# LRP erklärt immer *eine* Klasse. Auch bei negativem Logit sollte die
# Input-Summe näherungsweise diesem Logit entsprechen.

# %%
class_rows = []
for cls in range(10):
    explainer = build_lrp(model, cls, COMPOSITE_STRATEGY)
    layer_df = collect_layer_relevance(explainer, sample)
    r0 = float(layer_df["sum_R"].iloc[0])
    r1 = float(layer_df["sum_R"].iloc[-1])
    class_rows.append(
        {
            "class": cls,
            "prob": float(probs[cls]),
            "sum_R_start": r0,
            "sum_R_input": r1,
            "ratio": (r1 / r0) if r0 != 0 else np.nan,
            "is_pred": cls == pred_label,
            "is_true": cls == true_label,
        }
    )

class_df = pd.DataFrame(class_rows)
class_csv = target_dir / "per_class_conservation.csv"
class_df.to_csv(class_csv, index=False)
print(class_df.to_string(index=False, float_format=lambda v: f"{v:10.6f}"))
print(f"\nGespeichert: {class_csv}")

fig, ax = plt.subplots(figsize=(10, 4))
colors = [
    "C2" if r.is_pred else ("C1" if r.is_true else "C0") for _, r in class_df.iterrows()
]
ax.bar(class_df["class"], class_df["ratio"], color=colors, alpha=0.85)
ax.axhline(1.0, color="k", ls="--")
ax.set_xlabel("Zielklasse")
ax.set_ylabel("Erhaltungsquote")
ax.set_title(
    "ΣR(input)/ΣR(start) je erklärter Klasse "
    "(grün=Vorhersage, orange=Label falls abweichend)"
)
ax.set_xticks(range(10))
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
fig_path = target_dir / "07_per_class_conservation.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Abbildung: {fig_path}")

# %% [markdown]
# ### 8.2 Kleine Stichprobe zufälliger Testbeispiele
#
# Robustheit: bleibt die Quote über mehrere Volumen ähnlich?

# %%
RNG = np.random.default_rng(0)
n_samples = 12
sample_ids = RNG.choice(len(test_X), size=n_samples, replace=False)

sample_rows = []
for sid in sample_ids:
    x = test_X[sid : sid + 1]
    p = model.predict(x, verbose=0)[0]
    cls = int(np.argmax(p))
    explainer = build_lrp(model, cls, COMPOSITE_STRATEGY)
    layer_df = collect_layer_relevance(explainer, x)
    r0 = float(layer_df["sum_R"].iloc[0])
    r1 = float(layer_df["sum_R"].iloc[-1])
    sample_rows.append(
        {
            "test_idx": int(sid),
            "true": int(test_y[sid]),
            "pred": cls,
            "prob": float(p[cls]),
            "sum_R_start": r0,
            "sum_R_input": r1,
            "ratio": (r1 / r0) if r0 != 0 else np.nan,
        }
    )

sample_df = pd.DataFrame(sample_rows)
sample_csv = target_dir / "sample_conservation_ratios.csv"
sample_df.to_csv(sample_csv, index=False)
print(sample_df.to_string(index=False, float_format=lambda v: f"{v:10.6f}"))
print(
    f"\nMittel ± Std der Erhaltungsquote: "
    f"{sample_df['ratio'].mean():.3f} ± {sample_df['ratio'].std():.3f}"
)
print(f"Gespeichert: {sample_csv}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(sample_df)), sample_df["ratio"], color="C0", alpha=0.85)
ax.axhline(1.0, color="C3", ls="--")
ax.axhline(sample_df["ratio"].mean(), color="C2", ls=":", label="Mittel")
ax.set_xticks(range(len(sample_df)))
ax.set_xticklabels(
    [f"{r.test_idx}\n({r.true}→{r.pred})" for r in sample_df.itertuples()],
    fontsize=8,
)
ax.set_ylabel("Erhaltungsquote")
ax.set_title("Composite-LRP: Erhaltung über zufällige Testbeispiele")
ax.grid(True, axis="y", alpha=0.3)
ax.legend()
plt.tight_layout()
fig_path = target_dir / "08_sample_conservation_ratios.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Abbildung: {fig_path}")

# %% [markdown]
# <a id="sec-09"></a>
# ## 9. Fazit
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was dieses Notebook prüft
#
# 1. **Schichtweise ΣR** entlang des LRP-Rückwärtspfads (ab Mask-Lambda)
# 2. **Composite vs. einheitliche αβ-Regel** — trennt Regel-Lecks von Implementierungslecks
# 3. **Wo Relevanz verloren geht** (typisch: Dense + ε / Bias)
# 4. **Einfluss der Regelwahl** (`adjust_epsilon`, `ignore_bias`, αβ vs. ε)
# 5. **Robustheit** über Klassen und Stichproben
#
# ### Erwartetes Bild
#
# * **Composite:** Nach den beiden Dense-ε-Schichten fällt ΣR spürbar (oft auf ~70–85 %
#   des Logits); Conv/Pooling danach flach.
# * **Einheitlich αβ:** derselbe Startwert; ΣR kann an `DenseLRP` spürbar **steigen**
#   (α-Gewichtung / Bias-Normalisierung). Conv-/Pooling-Abschnitte danach oft wieder flach.
#   Die End-Quote ist deshalb nicht automatisch „besser" als Composite — sie misst einen
#   anderen Regelmechanismus.
# * Gemeinsame Sprünge in späteren Conv-Schichten → Code der betroffenen LRP-Schicht prüfen.
#
# ### Wann man stutzig werden sollte
#
# | Beobachtung | Mögliche Ursache |
# |---|---|
# | ΣR explodiert oder wechselt das Vorzeichen mitten im Netz | falsche Rückverteilung / Shape-Mismatch |
# | Pooling ändert ΣR stark | Pooling-Strategie prüfen (`winner-takes-all` vs. `redistribute`) |
# | flat/b-Regel erzeugt Relevanz auf Null-Voxeln | Maskierung der Heatmap nötig (vgl. Brain-Age-Notebook) |
# | `adjust_epsilon=True` stellt Quote ≈ 1 her, ohne dass Heatmaps sinnvoll bleiben | Erhaltung erzwungen, aber Verteilung kann sich ändern |
#
# Alle Tabellen und Abbildungen liegen unter:
#
# ```text
# output/notebooks/check_relevance_conservation_in_LRP/
# ```
