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
# <a id="top"></a>
# # LRP-Regeln an einfachen Beispielen
#
# Dieses Notebook benutzt `explainability.LRP` aus diesem Repo. Untersucht werden vier Regeln, jede an Daten, bei denen man den Effekt ohne Training ablesen kann:
#
# * **LRP-ε** mit mehreren Faktoren
# * **LRP-α1β0**
# * **LRP-α2β1**
# * **flat**
#
# Die Netze sind klein, die Gewichte sind fest eingetragen, der Bias ist aus. LRP-0 (die z-Regel) steht daneben als Referenz: dort ist die Relevanz genau der Beitrag $x_i w_i$.
#
# Jede Regel beantwortet eine andere Frage. Die Beispiele sind auf diese Frage zugeschnitten.
#
# ## Inhaltsverzeichnis
#
# | # | Abschnitt | Frage |
# |---|---|---|
# | 1 | [Score mit vier Merkmalen](#sec-score) | Referenz: Relevanz = Beitrag $x w$ |
# | 2 | [LRP-α1β0](#sec-a1b0) | Was spricht *für* den Score? |
# | 3 | [LRP-α2β1](#sec-a2b1) | Was spricht dafür, was dagegen? |
# | 4 | [flat](#sec-flat) | Wie sieht die Erklärung ohne Inhalt aus? |
# | 5 | [LRP-ε, instabiler Nenner](#sec-eps-unstable) | Ist die große Relevanz belastbar? |
# | 6 | [LRP-ε, stark gegen schwach](#sec-eps-path) | Welcher Pfad trägt den Score wirklich? |
# | 7 | [Dieselben Daten, alle Regeln](#sec-vergleich) | Worin unterscheiden sich die Antworten? |
# | 8 | [Fazit](#sec-fazit) | Welche Regel für welche Frage |
#

# %% [markdown]
# <a id="sec-api"></a>
# ## Wie die Regeln im Code heißen
#
# `LRP` legt dieselbe Regel auf jede gewichtstragende Schicht, solange keine `LRPStrategy` übergeben wird. `idx=0` ist das erklärte Ausgabeneuron. `layer` ist der Index der Schicht, deren Ausgabe erklärt wird.
#
# ```python
# LRP(model, layer=..., idx=0)                       # LRP-0
# LRP(model, layer=..., idx=0, epsilon=0.1)          # LRP-ε
# LRP(model, layer=..., idx=0, alpha=1.0, beta=0.0)  # α1β0
# LRP(model, layer=..., idx=0, alpha=2.0, beta=1.0)  # α2β1
# LRP(model, layer=..., idx=0, strategy=LRPStrategy(
#     layers=[{"flat": True}]   # ein Dict je Dense/Conv, von der Eingabe zur Ausgabe
# ))
# ```
#
# Im Code muss $\alpha = \beta + 1$ gelten. Die αβ-Implementierung der Dense-Schicht summiert über die Batch-Achse, deshalb geht hier immer nur **eine** Zeile in `LRP`.
#
# Die Formeln, gegen die wir die Zahlen halten:
#
# $$
# R_i^{\,0} = \sum_j \frac{x_i w_{ij}}{z_j}\, R_j
# \qquad
# z_j = \sum_i x_i w_{ij}
# $$
#
# $$
# R_i^{\,\varepsilon}
# = \sum_j \frac{x_i w_{ij}}{z_j + \varepsilon\,\mathrm{sign}(z_j)}\, R_j
# $$
#
# $$
# R_i^{\,\alpha\beta}
# = \sum_j \left(
#   \alpha\,\frac{z_{ij}^{+}}{z_j^{+}}
#   - \beta\,\frac{z_{ij}^{-}}{z_j^{-}}
# \right) R_j
# \qquad
# \alpha - \beta = 1
# $$
#
# `flat` setzt vor der z-Regel $x \leftarrow 1$ und $w \leftarrow 1$. Danach bekommt jeder Eingang denselben Anteil.
#

# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from IPython.display import display
from tensorflow.keras import Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Input


def find_repo_root() -> Path:
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "explainability").is_dir() and (candidate / "pyproject.toml").exists():
            return candidate
    raise RuntimeError("Repo-Wurzel von keras-explainability nicht gefunden")


repo_root = find_repo_root()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from explainability import LRP, LRPStrategy
from explainability.layers.layer import StandardLRPLayer

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
})

RULE_COLORS = {
    "LRP-0": "#4d4d4d",
    "ε": "#2c7fb8",
    "α1β0": "#31a354",
    "α2β1": "#e6550d",
    "flat": "#756bb1",
}


def make_dense(weights: np.ndarray, name: str = "score") -> Model:
    """Eine Dense-Schicht ohne Bias. `weights` hat Form (Eingänge, Ausgänge)."""
    weights = np.asarray(weights, dtype=np.float32)
    inputs = Input(shape=(weights.shape[0],), name="x")
    outputs = Dense(
        units=int(weights.shape[1]),
        use_bias=False,
        kernel_initializer=Constant(weights),
        name=name,
    )(inputs)
    return Model(inputs, outputs, name=name + "_model")


def explain(model: Model, x: np.ndarray, **lrp_kwargs) -> tuple[float, np.ndarray]:
    """Relevanz der Eingabe für Ausgabeneuron 0. Immer Batchgröße 1."""
    batch = np.asarray(x, dtype=np.float32).reshape(1, -1)
    explainer = LRP(model, layer=len(model.layers) - 1, idx=0, **lrp_kwargs)
    relevance = np.asarray(explainer(batch), dtype=np.float64).ravel()
    prediction = float(np.asarray(model(batch)).ravel()[0])
    return prediction, relevance


def report(names: list[str], prediction: float, relevance: np.ndarray) -> pd.DataFrame:
    total = float(relevance.sum())
    ratio = total / prediction if prediction != 0 else np.nan
    print(f"y = {prediction:.6g}    Summe R = {total:.6g}    Summe / y = {ratio:.4f}")
    frame = pd.DataFrame({"Merkmal": names, "Relevanz": relevance})
    display(frame.style.format({"Relevanz": "{:+.4f}"}).hide(axis="index"))
    return frame


def plot_relevance(names: list[str], relevance: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    colors = ["#2c7fb8" if value >= 0 else "#e34a33" for value in relevance]
    positions = np.arange(len(names))
    ax.bar(positions, relevance, color=colors, width=0.72)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(positions, names, rotation=15, ha="right")
    ax.set_ylabel("Relevanz")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_grouped(names: list[str], series: list[tuple[str, np.ndarray, str]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    positions = np.arange(len(names))
    width = 0.8 / len(series)
    for index, (label, values, color) in enumerate(series):
        offset = (index - (len(series) - 1) / 2) * width
        ax.bar(positions + offset, values, width=width * 0.92, label=label, color=color)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(positions, names, rotation=15, ha="right")
    ax.set_ylabel("Relevanz")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=len(series))
    fig.tight_layout()
    plt.show()


def active_rules(model: Model, **lrp_kwargs) -> None:
    explainer = LRP(model, layer=len(model.layers) - 1, idx=0, **lrp_kwargs)
    print("Gesetzte Regeln, von der Ausgabe zurück zur Eingabe:")
    for layer in explainer.layers:
        if not isinstance(layer, StandardLRPLayer):
            continue
        parts = []
        if layer.flat:
            parts.append("flat")
        if layer.alpha is not None:
            parts.append(f"α={layer.alpha:g}, β={layer.beta:g}")
        if layer.epsilon:
            parts.append(f"ε={layer.epsilon:g}")
        if not parts:
            parts.append("LRP-0")
        print(f"  {layer.name}: {', '.join(parts)}")



# %% [markdown]
# <a id="sec-score"></a>
# ## 1. Gemeinsames Beispiel: ein linearer Score
#
# [↑ Inhalt](#top)
#
# Vier Merkmale, eine Ausgabe, kein Bias. Das vierte Merkmal hat die **größte Aktivierung** und das **Gewicht 0**. Daran sieht man später, ob eine Regel den Inhalt benutzt oder ihn ignoriert.
#
# | Merkmal | Aktivierung $x$ | Gewicht $w$ | Beitrag $x w$ |
# |---|---:|---:|---:|
# | Stütze | 1 | +4 | +4 |
# | Widerspruch | 1 | −3 | −3 |
# | schwache Stütze | 1 | +1 | +1 |
# | totes Gewicht | 5 | 0 | 0 |
#
# $$y = 4 - 3 + 1 + 0 = 2$$
#
# **Frage an LRP-0:** Welchen Anteil hat jedes Merkmal am Score, gemessen an seinem Beitrag $x w$?
#
# Die z-Regel ohne Bias gibt diese Beiträge unverändert zurück. Summe der Relevanz = $y$.
#

# %%
FEATURE_NAMES = ["Stütze", "Widerspruch", "schwache Stütze", "totes Gewicht"]
WEIGHTS = np.array([[4.0], [-3.0], [1.0], [0.0]], dtype=np.float32)
X_SCORE = np.array([1.0, 1.0, 1.0, 5.0], dtype=np.float32)

score_model = make_dense(WEIGHTS)
contributions = X_SCORE * WEIGHTS.ravel()
print("Beiträge x*w:", contributions, "   Summe =", contributions.sum())

active_rules(score_model)
y_score, r_lrp0 = explain(score_model, X_SCORE)
report(FEATURE_NAMES, y_score, r_lrp0)
plot_relevance(FEATURE_NAMES, r_lrp0, "LRP-0: Relevanz = Beitrag x·w")

np.testing.assert_allclose(r_lrp0, contributions, atol=1e-5)
np.testing.assert_allclose(r_lrp0.sum(), y_score, atol=1e-5)


# %% [markdown]
# <a id="sec-a1b0"></a>
# ## 2. LRP-α1β0 — was spricht für den Score?
#
# [↑ Inhalt](#top)
#
# **Aufgabenstellung.** Der Score ist positiv ($y = 2$). Nenne nur die Merkmale, die ihn nach oben ziehen. Was ihn drückt, soll in der Erklärung nicht vorkommen.
#
# **Warum diese Regel.** $\alpha=1$, $\beta=0$ behält die positiven Beiträge und verwirft die negativen. Die positiven Beiträge teilen sich $y$ im Verhältnis ihrer Größe. $\alpha - \beta = 1$ hält die Summe bei $y$, sobald es positive Beiträge gibt.
#
# **Was an diesem Beispiel sichtbar werden soll.**
#
# * Stütze und schwache Stütze bleiben, im Verhältnis $4 : 1$.
# * Der Widerspruch (Beitrag −3) bekommt Relevanz 0.
# * Das tote Gewicht bleibt 0, weil sein Beitrag weder positiv noch negativ ist.
#
# $$
# R = 2 \cdot \left[\tfrac{4}{5},\, 0,\, \tfrac{1}{5},\, 0\right]
# = [1.6,\, 0,\, 0.4,\, 0]
# $$
#

# %%
active_rules(score_model, alpha=1.0, beta=0.0)
_, r_a1b0 = explain(score_model, X_SCORE, alpha=1.0, beta=0.0)
report(FEATURE_NAMES, y_score, r_a1b0)
plot_relevance(FEATURE_NAMES, r_a1b0, "α1β0: nur die stützenden Beiträge")

np.testing.assert_allclose(r_a1b0, [1.6, 0.0, 0.4, 0.0], atol=1e-5)


# %% [markdown]
# <a id="sec-a2b1"></a>
# ## 3. LRP-α2β1 — was spricht dafür, was dagegen?
#
# [↑ Inhalt](#top)
#
# **Aufgabenstellung.** Dieselben vier Merkmale. Zeige beide Richtungen: positive Relevanz stützt den Score, negative Relevanz widerspricht ihm. Die stützende Seite soll doppelt so stark gewichtet sein wie die widersprechende, die Summe soll $y$ bleiben.
#
# **Warum diese Regel.** $\alpha=2$, $\beta=1$ erfüllt $\alpha - \beta = 1$. Jeder positive Beitrag wird mit 2 gewichtet, jeder negative mit 1, jeweils normiert auf die eigene Seite. Die negative Relevanz ist damit lesbar als „spricht dagegen“, ohne dass die Summe von $y$ abweicht — **solange beide Seiten vorkommen**.
#
# **Was an diesem Beispiel sichtbar werden soll.**
#
# $$
# R_{\text{Stütze}} = 2 \cdot \tfrac{4}{5} \cdot 2 = 3.2
# \qquad
# R_{\text{Widerspruch}} = -1 \cdot \tfrac{3}{3} \cdot 2 = -2
# \qquad
# R_{\text{schwach}} = 2 \cdot \tfrac{1}{5} \cdot 2 = 0.8
# $$
#
# Der Widerspruch trägt die gesamte negative Masse. Die beiden Stützen sind gegenüber α1β0 verstärkt, damit die Summe wieder $y = 2$ ist.
#
# Danach dasselbe Netz mit einer zweiten Eingabe, bei der der Widerspruch der **größte Einzelbeitrag** ist und der Score nur noch knapp positiv bleibt. α1β0 blendet genau diesen größten Beitrag aus. α2β1 zeigt ihn.
#

# %%
active_rules(score_model, alpha=2.0, beta=1.0)
_, r_a2b1 = explain(score_model, X_SCORE, alpha=2.0, beta=1.0)
report(FEATURE_NAMES, y_score, r_a2b1)

plot_grouped(
    FEATURE_NAMES,
    [
        ("α1β0", r_a1b0, RULE_COLORS["α1β0"]),
        ("α2β1", r_a2b1, RULE_COLORS["α2β1"]),
    ],
    "Dieselbe Eingabe: nur dafür (α1β0) oder dafür und dagegen (α2β1)",
)
np.testing.assert_allclose(r_a2b1, [3.2, -2.0, 0.8, 0.0], atol=1e-5)

# Widerspruch ist hier der größte Beitrag, y bleibt knapp positiv.
x_close = np.array([1.0, 1.5, 1.0, 5.0], dtype=np.float32)
close_contributions = x_close * WEIGHTS.ravel()
print("Zweite Eingabe, Beiträge x*w:", close_contributions, "  Summe =", close_contributions.sum())
y_close, r_close_a1 = explain(score_model, x_close, alpha=1.0, beta=0.0)
_, r_close_a2 = explain(score_model, x_close, alpha=2.0, beta=1.0)
print("α1β0")
report(FEATURE_NAMES, y_close, r_close_a1)
print("α2β1")
report(FEATURE_NAMES, y_close, r_close_a2)
plot_grouped(
    FEATURE_NAMES,
    [
        ("α1β0", r_close_a1, RULE_COLORS["α1β0"]),
        ("α2β1", r_close_a2, RULE_COLORS["α2β1"]),
    ],
    "Widerspruch ist der größte Beitrag — α1β0 zeigt ihn nicht",
)
np.testing.assert_allclose(r_close_a1, [0.4, 0.0, 0.1, 0.0], atol=1e-5)
np.testing.assert_allclose(r_close_a2, [0.8, -0.5, 0.2, 0.0], atol=1e-5)

# Ohne negative Beiträge entfällt der β-Term, α=2 verdoppelt die Summe.
weights_positive = np.array([[4.0], [0.0], [1.0], [0.0]], dtype=np.float32)
positive_model = make_dense(weights_positive, name="nur_positiv")
x_positive = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
y_pos, r_pos = explain(positive_model, x_positive, alpha=2.0, beta=1.0)
print("Nur positive Gewichte, α2β1: Summe R ist 2·y, weil es keine negative Seite gibt.")
report(FEATURE_NAMES, y_pos, r_pos)
np.testing.assert_allclose(r_pos.sum(), 2.0 * y_pos, atol=1e-4)


# %% [markdown]
# <a id="sec-flat"></a>
# ## 4. flat — Erklärung ohne Inhalt
#
# [↑ Inhalt](#top)
#
# **Aufgabenstellung.** Verteile die Relevanz des Scores so, als wäre jede Aktivierung 1 und jedes Gewicht 1. Die Erklärung soll ein Nullmodell sein: sie darf weder auf die große Aktivierung „totes Gewicht“ noch auf die echten Gewichte reagieren.
#
# **Warum diese Regel.** `flat` ersetzt vor der z-Regel Aktivierung und Gewicht durch Einsen. Bei vier Eingängen ist der flache Nenner 4, jedes Merkmal bekommt $y / 4 = 0.5$.
#
# **Was an diesem Beispiel sichtbar werden soll.** Stütze, Widerspruch, schwache Stütze und das tote Gewicht sind gleichauf, obwohl die Beiträge $+4, -3, +1, 0$ sind und das tote Gewicht die Aktivierung 5 hat.
#
# In den CNNs dieses Repos liegt `flat` auf den ersten Convolution-Schichten aus demselben Grund: die erste Umverteilung soll nicht der Pixelhelligkeit folgen. Das Nullmodell hier ist dieselbe Idee an einer Schicht, die man noch von Hand nachrechnen kann.
#
# `LRPStrategy.layers` hat einen Eintrag pro gewichtstragender Schicht, von der Eingabe zur Ausgabe. Dieses Netz hat eine Dense-Schicht, also einen Eintrag.
#

# %%
flat_strategy = LRPStrategy(layers=[{"flat": True}])
active_rules(score_model, strategy=flat_strategy)
_, r_flat = explain(score_model, X_SCORE, strategy=flat_strategy)
report(FEATURE_NAMES, y_score, r_flat)
plot_relevance(FEATURE_NAMES, r_flat, "flat: jedes Merkmal bekommt y/4, Inhalt wird ignoriert")

np.testing.assert_allclose(r_flat, np.full(4, y_score / 4.0), atol=1e-5)


# %% [markdown]
# <a id="sec-eps-unstable"></a>
# ## 5. LRP-ε — zwei Buchungen, die sich fast aufheben
#
# [↑ Inhalt](#top)
#
# **Aufgabenstellung.** Eine Einnahme von $+1$ und eine Ausgabe von $-0.999$ lassen den Saldo bei $y \approx 0.001$. LRP-0 erklärt diesen winzigen Saldo mit zwei Relevanzen vom Betrag $\approx 1$, die sich gegenseitig fast auslöschen. Ist das eine belastbare Zerlegung oder ein Artefakt der Division durch einen sehr kleinen Nenner?
#
# **Warum diese Regel.** LRP-ε vergrößert den Nenner um $\varepsilon\,\mathrm{sign}(z)$. Der Anteil $\varepsilon / (|z| + \varepsilon)$ wird vom Stabilisator geschluckt und keinem Merkmal zugeteilt. Die **Form** der Erklärung (Einnahme gegen Ausgabe) bleibt auf dieser einen Schicht erhalten, der **Betrag** fällt. Ein größeres $\varepsilon$ schluckt mehr.
#
# **Faktoren.** LRP-0 und $\varepsilon \in \{0.01,\, 0.1,\, 1,\, 10\}$.
#
# Auf dem Score aus Abschnitt 1 passiert dasselbe, nur ohne Explosion: dort ist $z = 2$ gutartig, und ε skaliert alle vier Relevanzen mit demselben Faktor $2 / (2 + \varepsilon)$. Die Frage „was ist positiv, was ist negativ?“ beantwortet ε deshalb nicht. Dafür sind αβ und flat da. Der Vergleich steht in Abschnitt 7.
#

# %%
ledger_names = ["Einnahme", "Ausgabe"]
ledger_weights = np.array([[1.0], [-0.999]], dtype=np.float32)
ledger_model = make_dense(ledger_weights, name="saldo")
x_ledger = np.array([1.0, 1.0], dtype=np.float32)

epsilon_factors = [
    ("LRP-0", None),
    ("ε = 0.01", 0.01),
    ("ε = 0.1", 0.1),
    ("ε = 1", 1.0),
    ("ε = 10", 10.0),
]

ledger_rows = []
for label, epsilon in epsilon_factors:
    kwargs = {} if epsilon is None else {"epsilon": float(epsilon)}
    prediction, relevance = explain(ledger_model, x_ledger, **kwargs)
    ledger_rows.append({
        "Regel": label,
        "R Einnahme": relevance[0],
        "R Ausgabe": relevance[1],
        "Summe R": relevance.sum(),
        "Summe / y": relevance.sum() / prediction,
        "|R| max": np.max(np.abs(relevance)),
    })

ledger_table = pd.DataFrame(ledger_rows)
display(ledger_table.style.format({
    "R Einnahme": "{:+.4f}",
    "R Ausgabe": "{:+.4f}",
    "Summe R": "{:.6f}",
    "Summe / y": "{:.4f}",
    "|R| max": "{:.4f}",
}).hide(axis="index"))

fig, ax = plt.subplots(figsize=(7.2, 3.8))
positions = np.arange(len(ledger_table))
ax.plot(positions, ledger_table["|R| max"], marker="o", color=RULE_COLORS["ε"])
ax.set_xticks(positions, ledger_table["Regel"], rotation=15, ha="right")
ax.set_ylabel("Betrag der größeren Relevanz")
ax.set_yscale("log")
ax.set_title("ε schluckt die instabile Masse, die Richtung bleibt")
fig.tight_layout()
plt.show()

lrp0_peak = float(ledger_table.loc[0, "|R| max"])
assert ledger_table["|R| max"].is_monotonic_decreasing
assert float(ledger_table["|R| max"].iloc[-1]) < lrp0_peak / 100


# %% [markdown]
# <a id="sec-eps-path"></a>
# ## 6. LRP-ε — starker Pfad gegen schwachen Pfad
#
# [↑ Inhalt](#top)
#
# **Aufgabenstellung.** Der Score entsteht aus zwei versteckten Neuronen. Beide gehen mit Gewicht 1 in die Ausgabe. Eines ist stark, eines ist nur ein leiser Nebenpfad. Welche Eingaben tragen den Score, und ab welchem $\varepsilon$ ist der Nebenpfad in der Erklärung praktisch verschwunden?
#
# ```text
# Signal A, Signal B  --·5-->   h_signal = 10  --·1-->  y = 10.4
# Rauschen C, Rauschen D --·0.2--> h_rauschen = 0.4 --·1--^
# ```
#
# Alle vier Eingaben sind 1. LRP-0 gibt $[5, 5, 0.2, 0.2]$. Das Verhältnis Signal zu Rauschen ist $25$.
#
# **Warum ε hier die Form ändert und nicht nur die Summe.** Auf dem Weg zurück durch ein verstecktes Neuron wird dessen Relevanz mit $h / (h + \varepsilon)$ weitergereicht. Für $h = 10$ ist dieser Faktor bei moderatem $\varepsilon$ noch nahe 1. Für $h = 0.4$ fällt er schnell. Der schwache Pfad wird also überproportional geschluckt. Dasselbe $\varepsilon$ liegt hier auf beiden Dense-Schichten.
#
# **Faktoren.** Wieder LRP-0 und $\varepsilon \in \{0.01,\, 0.1,\, 1,\, 10\}$.
#

# %%
path_names = ["Signal A", "Signal B", "Rauschen C", "Rauschen D"]
hidden = Dense(
    2,
    use_bias=False,
    kernel_initializer=Constant(np.array([
        [5.0, 0.0],
        [5.0, 0.0],
        [0.0, 0.2],
        [0.0, 0.2],
    ], dtype=np.float32)),
    name="hidden",
)
output = Dense(
    1,
    use_bias=False,
    kernel_initializer=Constant(np.array([[1.0], [1.0]], dtype=np.float32)),
    name="output",
)
path_input = Input(shape=(4,), name="x")
path_model = Model(path_input, output(hidden(path_input)), name="zwei_pfade")
x_path = np.ones(4, dtype=np.float32)

hidden_values = np.asarray(hidden(x_path.reshape(1, -1))).ravel()
print("Versteckte Aktivierungen [Signal, Rauschen]:", hidden_values)
print("y =", float(np.asarray(path_model(x_path.reshape(1, -1))).ravel()[0]))
active_rules(path_model, epsilon=1.0)

path_rows = []
path_relevances = []
for label, epsilon in epsilon_factors:
    kwargs = {} if epsilon is None else {"epsilon": float(epsilon)}
    prediction, relevance = explain(path_model, x_path, **kwargs)
    path_relevances.append(relevance)
    path_rows.append({
        "Regel": label,
        "Signal A": relevance[0],
        "Rauschen C": relevance[2],
        "Verhältnis Signal/Rauschen": relevance[0] / relevance[2],
        "Summe R": relevance.sum(),
        "Summe / y": relevance.sum() / prediction,
    })

path_table = pd.DataFrame(path_rows)
display(path_table.style.format({
    "Signal A": "{:.4f}",
    "Rauschen C": "{:.4f}",
    "Verhältnis Signal/Rauschen": "{:.1f}",
    "Summe R": "{:.4f}",
    "Summe / y": "{:.4f}",
}).hide(axis="index"))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
positions = np.arange(len(path_names))
width = 0.36
axes[0].bar(positions - width / 2, path_relevances[0], width=width, label="LRP-0", color=RULE_COLORS["LRP-0"])
axes[0].bar(positions + width / 2, path_relevances[-1], width=width, label="ε = 10", color=RULE_COLORS["ε"])
axes[0].set_xticks(positions, path_names, rotation=15, ha="right")
axes[0].set_ylabel("Relevanz")
axes[0].set_title("LRP-0 gegen ε = 10")
axes[0].legend(frameon=False)

axes[1].plot(
    np.arange(len(path_table)),
    path_table["Verhältnis Signal/Rauschen"],
    marker="o",
    color=RULE_COLORS["ε"],
)
axes[1].set_xticks(np.arange(len(path_table)), path_table["Regel"], rotation=15, ha="right")
axes[1].set_ylabel("Relevanz Signal A  /  Relevanz Rauschen C")
axes[1].set_title("Der schwache Pfad fällt überproportional")
fig.tight_layout()
plt.show()

ratios = path_table["Verhältnis Signal/Rauschen"].to_numpy()
assert np.all(np.diff(ratios) > 0)
np.testing.assert_allclose(path_relevances[0], [5.0, 5.0, 0.2, 0.2], atol=1e-4)


# %% [markdown]
# <a id="sec-vergleich"></a>
# ## 7. Dieselben vier Merkmale, alle Regeln
#
# [↑ Inhalt](#top)
#
# Der Score aus Abschnitt 1 noch einmal unter jeder Regel. $\varepsilon = 1$ steht mit dabei, damit man sieht: auf dieser gut konditionierten Schicht ändert ε nur den Maßstab ($2/3$ von LRP-0), nicht die Rollen der Merkmale.
#
# | Regel | Stütze | Widerspruch | schwache Stütze | totes Gewicht | Summe |
# |---|---:|---:|---:|---:|---:|
# | LRP-0 | +4 | −3 | +1 | 0 | 2 |
# | ε = 1 | +2.667 | −2 | +0.667 | 0 | 1.333 |
# | α1β0 | +1.6 | 0 | +0.4 | 0 | 2 |
# | α2β1 | +3.2 | −2 | +0.8 | 0 | 2 |
# | flat | +0.5 | +0.5 | +0.5 | +0.5 | 2 |
#

# %%
_, r_eps1 = explain(score_model, X_SCORE, epsilon=1.0)
comparison = [
    ("LRP-0", r_lrp0, RULE_COLORS["LRP-0"]),
    ("ε = 1", r_eps1, RULE_COLORS["ε"]),
    ("α1β0", r_a1b0, RULE_COLORS["α1β0"]),
    ("α2β1", r_a2b1, RULE_COLORS["α2β1"]),
    ("flat", r_flat, RULE_COLORS["flat"]),
]
plot_grouped(FEATURE_NAMES, comparison, "Ein Score, fünf Antworten")

overview = pd.DataFrame(
    {label: values for label, values, _ in comparison},
    index=FEATURE_NAMES,
)
overview.loc["Summe"] = overview.sum(axis=0)
display(overview.style.format("{:+.3f}"))
np.testing.assert_allclose(r_eps1, r_lrp0 * (y_score / (y_score + 1.0)), atol=1e-4)


# %% [markdown]
# <a id="sec-fazit"></a>
# ## 8. Fazit
#
# [↑ Inhalt](#top)
#
# | Frage | Regel | Was das Beispiel gezeigt hat |
# |---|---|---|
# | Was spricht für den positiven Score? | **α1β0** | Nur Stütze und schwache Stütze. Der Widerspruch bleibt unsichtbar, auch wenn er der größte Einzelbeitrag ist. |
# | Was spricht dafür, und was dagegen? | **α2β1** | Positive und negative Balken. Die Summe bleibt $y$, solange beide Seiten vorkommen. Ohne negative Beiträge verdoppelt $\alpha=2$ die Summe. |
# | Wie sähe die Erklärung aus, wenn Inhalt und Gewichte keine Rolle spielen? | **flat** | Gleichverteilung, auch auf ein Gewicht von 0 und eine große Aktivierung. Nullmodell, und dieselbe Idee wie `flat` auf den ersten Conv-Schichten. |
# | Ist eine große, sich aufhebende Relevanz belastbar? | **LRP-ε**, Faktor variieren | Einnahme gegen Ausgabe bei $y \approx 0.001$. Die Richtung bleibt, der Betrag fällt mit $\varepsilon$. Die Summe fällt mit, das ist der Stabilisator. |
# | Welcher Pfad trägt, welcher ist nur schwach? | **LRP-ε** über mehr als eine Schicht | Signal bleibt, Rauschen fällt überproportional, weil $h / (h + \varepsilon)$ für das kleine Neuron schneller sinkt. |
#
# ε und αβ beantworten verschiedene Fragen. ε entscheidet, **wie viel** von einem Beitrag stabil genug ist, um in der Erklärung zu bleiben. αβ entscheidet, **welche Richtung** eines Beitrags gezeigt wird. `flat` entscheidet gar nicht nach Inhalt.
#
