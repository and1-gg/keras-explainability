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
# # LRP-Regeln: von der Dense-Schicht zum CNN
#
# Dieses Notebook benutzt `explainability.LRP` aus diesem Repo und ist zweiteilig.
#
# **Teil I — die Regeln an einer einzelnen Dense-Schicht.** Vier Regeln an Daten, bei denen man den
# Effekt ohne Training ablesen kann: **LRP-ε** mit mehreren Faktoren, **LRP-α1β0**, **LRP-α2β1**,
# **flat**. Die Netze sind klein, die Gewichte sind fest eingetragen, der Bias ist aus. LRP-0 (die
# z-Regel) steht daneben als Referenz: dort ist die Relevanz genau der Beitrag $x_i w_i$.
#
# **Teil II — dieselben Regeln in einem CNN.** Grundlage ist das Buchkapitel *Samek, Arras, Osman,
# Montavon, Müller (2021), „Explaining the Decisions of Convolutional and Recurrent Neural
# Networks“*, Kapitel **1.4.1**, **1.4.2** und **Tabelle 1.1**. Teil II erklärt den
# LRP-Rückwärtsdurchlauf durch ein Convolutional Network, behandelt **jede Regel aus Tabelle 1.1**
# an einem Beispiel mit Abbildung — einschließlich **LRP-γ**, der **w²-Regel** und der
# **z$^\mathcal{B}$-Regel**, die in Teil I nicht vorkommen — und beantwortet am Ende die praktische
# Frage: **welche Regel gehört auf welchen Schichttyp?**
#
# Jede Regel beantwortet eine andere Frage. Die Beispiele sind auf diese Frage zugeschnitten.
#
# ## Inhaltsverzeichnis
#
# ### Teil I — eine Dense-Schicht
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
# | 8 | [Fazit Teil I](#sec-fazit) | Welche Regel für welche Frage |
#
# ### [Teil II — LRP im CNN](#teil-2)
#
# | # | Abschnitt | Frage |
# |---|---|---|
# | 9 | [Der Rückwärtsdurchlauf im CNN](#sec-cnn-ablauf) | Wie fließt Relevanz durch Convolution, ReLU und Pooling? |
# | 10 | [Tabelle 1.1](#sec-tabelle) | Welche Regeln gibt es, und was kann dieses Repo davon? |
# | 11 | [Das Demo-CNN](#sec-demo-cnn) | Welche Bildregion wirkt wirklich auf die Vorhersage? |
# | 12 | [LRP-0 im CNN](#sec-cnn-lrp0) | Referenz: Erhaltung, und wer leer ausgeht |
# | 13 | [LRP-ε](#sec-cnn-eps) | Was bleibt übrig, wenn man stabilisiert? |
# | 14 | [LRP-γ](#sec-cnn-gamma) | Wie bevorzugt man die stützende Seite, ohne Masse zu verlieren? |
# | 15 | [LRP-αβ](#sec-cnn-ab) | Was kostet eine rein positive Karte? |
# | 16 | [flat](#sec-cnn-flat) | Wie sieht „nur das rezeptive Feld“ aus? |
# | 17 | [w²-Regel und z$^\mathcal{B}$-Regel](#sec-cnn-first) | Womit erklärt man die *erste* Schicht? |
# | 18 | [Pooling und BatchNorm](#sec-cnn-pool) | Was passiert an den Schichten ohne Gewichte? |
# | 19 | [Welcher Input steuert welche Regel?](#sec-cnn-einfluss) | Wie reagiert jede Regel auf eine Änderung im Bild? |
# | 20 | [Komposit-Strategie](#sec-cnn-composite) | Wie setzt man die Empfehlung des Papers als `LRPStrategy` um? |
# | 21 | [Fazit Teil II](#sec-cnn-fazit) | Welche Regel auf welche Schicht |
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
    "LRP-0": "#595959",
    "ε": "#2c7fb8",
    "α1β0": "#31a354",
    "α2β1": "#e6550d",
    "flat": "#dd3497",
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
# # Teil I — LRP-Regeln an einer Dense-Schicht
#
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
# ## 8. Fazit Teil I
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

# %% [markdown]
# <a id="teil-2"></a>
# # Teil II — LRP in Convolutional Neural Networks
#
# [↑ Inhalt](#top)
#
# Quelle für diesen Teil: **Samek, Arras, Osman, Montavon, Müller (2021), „Explaining the Decisions of
# Convolutional and Recurrent Neural Networks“**, Kapitel **1.4.1 (LRP in Convolutional Neural
# Networks)**, **1.4.2 (Theoretical Interpretation / Choosing LRP Rules in Practice)** und
# **Tabelle 1.1**.
#
# Teil I hat jede Regel an einer einzelnen Dense-Schicht gezeigt, wo man die Zahlen von Hand
# nachrechnen kann. Teil II stellt die Frage, die in der Praxis zählt: **ein Netz hat viele
# Schichten verschiedenen Typs — welche Regel gehört auf welche?**
#
# <a id="sec-cnn-ablauf"></a>
# ## 9. Der LRP-Rückwärtsdurchlauf im CNN
#
# [↑ Inhalt](#top)
#
# **Vorwärts.** In jeder Schicht $i$ rechnet das Netz (Gl. 9 im Paper)
#
# $$
# x_k^{[i]} = \sigma\Big( \sum_{j=1}^{n_{i-1}} x_j^{[i-1]} w_{jk}^{[i]} + b_k^{[i]} \Big),
# \qquad \sigma(x) = \max(0, x).
# $$
#
# Bei einer Convolution ist das dieselbe Formel — nur laufen $j$ und $k$ über *Ort × Kanal*, und
# dieselben Gewichte $w^{[i]}$ werden an jedem Ort wiederverwendet. Für LRP ist eine Convolution
# deshalb **kein neuer Fall**, sondern eine lineare Schicht mit geteilten Gewichten.
#
# **Initialisierung.** Erklärt wird die Vorhersage *vor* Softmax. Die Relevanz der Ausgabeschicht
# wird auf diesen Wert gesetzt, alle anderen Ausgabeneuronen auf 0. Genau das macht `LRP(..., idx=k)`
# in diesem Repo: `remove_activation` entfernt vorher Sigmoid/Softmax, eine Maske lässt nur Neuron
# `idx` stehen.
#
# **Rückwärts.** Die allgemeine Umverteilungsregel (Gl. 10) lautet
#
# $$
# R_j^{[i-1]} \;=\; \sum_{k=1}^{n_i}
# \frac{z_{jk}^{[i]}}{\sum_{l=1}^{n_{i-1}} z_{lk}^{[i]} + b_k^{[i]}}\; R_k^{[i]} ,
# $$
#
# wobei $z_{jk}^{[i]}$ misst, *wie stark* Neuron $j$ der unteren Schicht Neuron $k$ der oberen
# Schicht relevant gemacht hat. Alle Regeln aus Tabelle 1.1 sind verschiedene Antworten auf die
# Frage, was genau man für $z_{jk}$ einsetzt.
#
# **Erhaltung.** Jedes Neuron gibt so viel Relevanz nach unten weiter, wie es von oben bekommen hat.
# Über alle Schichten hinweg (Gl. 8):
#
# $$
# f(X) \;\approx\; \sum_{k} R_k^{[L]} \;=\; \dots \;=\; \sum_{i} R_i^{[0]} .
# $$
#
# Das gilt exakt nur ohne Bias; Bias-Terme ziehen einen Teil der Relevanz ab. Und **LRP-ε
# verletzt die Erhaltung absichtlich**: der Stabilisator schluckt Relevanz, um das
# Signal-Rausch-Verhältnis zu verbessern (Fußnote 3 im Paper). Das haben wir in Teil I,
# Abschnitt 5, an zwei Buchungen gesehen — gleich sehen wir es an einem Bild wieder.
#
# **Die Abbildung 1.1 des Papers in Worten.** Ein Bild (ein Hahn) läuft durch Convolution-Schichten
# und dann durch Fully-connected-Schichten. Die Vorhersage „rooster“ wird als Relevanz gesetzt und
# rückwärts geschoben: durch die **fc-Schichten mit LRP-ε**, durch die **Convolution-Schichten mit
# LRP-α₁β₀**. Heraus kommt eine Heatmap, auf der Kamm und Kopf leuchten. Dass dort **zwei
# verschiedene Regeln** benutzt werden, ist keine Nachlässigkeit, sondern der Kern von Kapitel 1.4.2:
# verschiedene Schichttypen brauchen verschiedene Regeln.
#

# %% [markdown]
# <a id="sec-tabelle"></a>
# ## 10. Tabelle 1.1 — alle Regeln auf einen Blick
#
# [↑ Inhalt](#top)
#
# Notation wie im Paper: $(x)^+ = \max(0,x)$, $(x)^- = \min(0,x)$, und der Index $l=0$ trägt den Bias
# mit $x_0^{[i-1]} := 1$, $w_{0k}^{[i]} := b_k^{[i]}$.
#
# | Regel | Umverteilung $R_j^{[i-1]} = \sum_k (\dots)\, R_k^{[i]}$ | Empfohlen für | DTD |
# |---|---|---|:--:|
# | **LRP-0** | $\dfrac{x_j^{[i-1]} w_{jk}^{[i]}}{\sum_l x_l^{[i-1]} w_{lk}^{[i]}}$ | fc-Schichten (nur die obersten) | ✓ |
# | **LRP-ε** | $\dfrac{x_j^{[i-1]} w_{jk}^{[i]}}{\varepsilon + \sum_l x_l^{[i-1]} w_{lk}^{[i]}}$ | fc-Schichten, oberste conv-Schichten | ✓ |
# | **LRP-γ** | $\dfrac{x_j^{[i-1]}\big(w_{jk}^{[i]} + \gamma (w_{jk}^{[i]})^+\big)}{\sum_l x_l^{[i-1]}\big(w_{lk}^{[i]} + \gamma (w_{lk}^{[i]})^+\big)}$ | conv-Schichten | ✓ |
# | **LRP-αβ**<br>($\alpha-\beta=1$) | $\alpha\,\dfrac{(x_j^{[i-1]} w_{jk}^{[i]})^+}{\sum_l (x_l^{[i-1]} w_{lk}^{[i]})^+} - \beta\,\dfrac{(x_j^{[i-1]} w_{jk}^{[i]})^-}{\sum_l (x_l^{[i-1]} w_{lk}^{[i]})^-}$ | conv-Schichten | ✗ (außer $\alpha{=}1,\beta{=}0$) |
# | **flat** | $\dfrac{1}{n_{i-1}}$ | Auflösung senken | ✗ |
# | **w²-Regel** | $\dfrac{(w_{ij}^{[1]})^2}{\sum_l (w_{lj}^{[1]})^2}$ | **erste Schicht**, Eingabe in $\mathbb{R}^d$ | ✓ |
# | **z$^\mathcal{B}$-Regel** | $\dfrac{x_i^{[0]} w_{ij}^{[1]} - l_i (w_{ij}^{[1]})^+ - h_i (w_{ij}^{[1]})^-}{\sum_l \big(x_l^{[0]} w_{lj}^{[1]} - l_l (w_{lj}^{[1]})^+ - h_l (w_{lj}^{[1]})^-\big)}$ | **erste Schicht**, Pixel in $[l_i, h_i]$ | ✓ |
#
# Dazu die beiden Schichttypen, die im Paper separat abgehandelt werden:
#
# | Schichttyp | Behandlung laut Paper |
# |---|---|
# | **Max-Pooling** | winner-take-all: die gesamte Relevanz geht an das Maximum des Fensters |
# | **Average-Pooling** | lineare Schicht mit positiven konstanten Gewichten → die Regeln oben gelten unverändert |
# | **BatchNorm** | *vor* LRP mit der benachbarten conv/fc-Schicht verschmelzen; LRP sieht dann nur conv/fc, ReLU, Pooling |
#
# ### Was davon in diesem Repo direkt verfügbar ist
#
# | Regel | Aufruf | Status |
# |---|---|---|
# | LRP-0 | `LRP(model, layer=…, idx=0)` | implementiert |
# | LRP-ε | `epsilon=…` bzw. `{"epsilon": …}` | implementiert |
# | LRP-γ | `gamma=…` bzw. `{"gamma": …}` | implementiert |
# | LRP-αβ | `alpha=…, beta=…` bzw. `{"alpha": …, "beta": …}` | implementiert ($\alpha = \beta + 1$ erzwungen) |
# | flat | `{"flat": True}` | implementiert |
# | Max-Pooling | `LRPStrategy(pooling=[{"strategy": "winner-takes-all"}])` | implementiert (Default) |
# | Average-Pooling | `{"strategy": "redistribute"}` | implementiert (Default) |
# | BatchNorm | `fuse_batchnorm` läuft automatisch in `LRP.__init__` | implementiert |
# | w²-Regel | — | **nicht implementiert** → Abschnitt 17 rechnet sie hier nach |
# | z$^\mathcal{B}$-Regel | — | **nicht implementiert** → Abschnitt 17 rechnet sie hier nach |
#
# Zusätzlich kennt `StandardLRPLayer` das Repo-eigene Flag `b=True` (Aktivierungen durch Einsen
# ersetzen, Gewichte behalten). Das steht nicht in Tabelle 1.1; es liegt zwischen `flat`
# (Aktivierungen *und* Gewichte auf 1) und den datenabhängigen Regeln.
#
# `LRPStrategy(layers=[...])` erwartet **einen Eintrag pro gewichtstragender Schicht, von der
# Eingabe zur Ausgabe** — bei unserem Netz also `[conv1, conv2, score]`. `pooling=[...]` analog
# für die Pooling-Schichten.
#

# %% [markdown]
# <a id="sec-demo-cnn"></a>
# ## 11. Das Demo-CNN: drei Regionen mit bekannter Wirkung
#
# [↑ Inhalt](#top)
#
# Wie in Teil I wird nichts trainiert. Alle Gewichte sind von Hand eingetragen, damit jede Heatmap
# gegen eine bekannte Wahrheit geprüft werden kann.
#
# **Das Bild** ist 12×12, ein Kanal, und enthält drei Regionen:
#
# | Region | Muster | Helligkeit | Rolle |
# |---|---|---:|---|
# | **Plus** | $3\times3$-Kreuz | 1 | das Muster, das das Netz belohnt |
# | **Balken** | waagerechter $1\times3$-Strich | 1 | das Muster, das das Netz bestraft |
# | **Fleck** | volles $4\times4$-Quadrat | **2** | **das hellste im Bild — und ohne jede Wirkung** |
#
# **Das Netz:**
#
# ```text
# x (12,12,1)
#   └ conv1  3 Filter 3×3, kein Bias   → Musterdetektoren
#     └ relu1
#       └ pool1  MaxPooling 2×2        → (6,6,3)
#         └ conv2  2 Filter 3×3        → mischt Kanäle: Kanal 0 → „dafür“, Kanal 1 → „dagegen“
#           └ relu2
#             └ pool2  MaxPooling 2×2  → (3,3,2)
#               └ flatten (18)
#                 └ score  Dense(1)    → +1 auf jede „dafür“-Zelle, −1 auf jede „dagegen“-Zelle
# ```
#
# Die drei Filter von `conv1`:
#
# $$
# f_{\text{Plus}} = \begin{pmatrix}-3&2&-3\\0&1&0\\-3&2&-3\end{pmatrix}
# \quad
# f_{\text{Balken}} = \begin{pmatrix}-1&-1&-1\\1&1&1\\-1&-1&-1\end{pmatrix}
# \quad
# f_{\text{Fleck}} = \tfrac{1}{9}\begin{pmatrix}1&1&1\\1&1&1\\1&1&1\end{pmatrix}
# $$
#
# Die beiden ersten sind so gewählt, dass sie auf einer **gleichmäßig hellen Fläche und an deren
# Kanten $\le 0$ liefern** — nach ReLU also exakt 0. Der Fleck erreicht damit nur Kanal 2, und Kanal 2
# bekommt in `conv2` das Gewicht 0. Ergebnis: **der Fleck ist die hellste Region des Bildes und
# beeinflusst die Vorhersage überhaupt nicht.** Das ist die Sonde für die Frage „folgt diese Regel
# dem Inhalt oder der Helligkeit?“.
#
# Der Hintergrund ist die zweite Sonde: er ist exakt 0. Jede Regel, die mit $x_j$ multipliziert,
# muss ihm zwangsläufig Relevanz 0 geben. Regeln, die das nicht tun (`flat`, w², z$^\mathcal{B}$),
# erkennt man sofort daran, dass dort etwas leuchtet.
#

# %%
from tensorflow.keras.layers import Activation, Conv2D, Flatten, MaxPooling2D

GRID = 12

PATTERNS = {
    "Plus":   np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.float32),
    "Balken": np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.float32),
    "Fleck":  np.ones((4, 4), dtype=np.float32),
}
POSITIONS = {"Plus": (1, 1), "Balken": (1, 8), "Fleck": (7, 7)}
AMPLITUDES = {"Plus": 1.0, "Balken": 1.0, "Fleck": 2.0}
REGION_NAMES = ["Plus", "Balken", "Fleck", "Hintergrund"]

FILTER_PLUS = np.array([[-3.0, 2.0, -3.0], [0.0, 1.0, 0.0], [-3.0, 2.0, -3.0]], dtype=np.float32)
FILTER_BALKEN = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], dtype=np.float32)
FILTER_FLECK = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)

CONV1_KERNEL = np.stack([FILTER_PLUS, FILTER_BALKEN, FILTER_FLECK], axis=-1)[:, :, None, :]
# conv2: raeumlicher 3x3-Mittelwert, dabei Kanal 0 -> Ausgang 0, Kanal 1 -> Ausgang 1, Kanal 2 -> nichts
CHANNEL_MIX = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
CONV2_KERNEL = np.tile(CHANNEL_MIX[None, None], (3, 3, 1, 1)) / 9.0
SCORE_KERNEL = np.zeros((18, 1), dtype=np.float32)
SCORE_KERNEL[0::2, 0] = 1.0    # Kanal 0 der 3x3x2-Karte: spricht dafuer
SCORE_KERNEL[1::2, 0] = -1.0   # Kanal 1: spricht dagegen


def make_image(**amplitudes) -> np.ndarray:
    """12x12-Bild mit den drei Mustern. Schluesselwoerter skalieren einzelne Regionen."""
    scales = dict(AMPLITUDES)
    scales.update(amplitudes)
    image = np.zeros((GRID, GRID), dtype=np.float32)
    for name, (row, col) in POSITIONS.items():
        pattern = PATTERNS[name]
        image[row:row + pattern.shape[0], col:col + pattern.shape[1]] += pattern * scales[name]
    return image


def region_masks(border: int = 1) -> dict[str, np.ndarray]:
    """Eine Maske je Muster, um den Rand des 3x3-Rezeptiven Felds erweitert."""
    masks = {}
    for name, (row, col) in POSITIONS.items():
        height, width = PATTERNS[name].shape
        mask = np.zeros((GRID, GRID), dtype=bool)
        mask[max(row - border, 0):row + height + border,
             max(col - border, 0):col + width + border] = True
        masks[name] = mask
    covered = np.logical_or.reduce(list(masks.values()))
    masks["Hintergrund"] = ~covered
    return masks


MASKS = region_masks()


def _cnn_head(tensor):
    """Alles oberhalb von conv1 — als Funktion, damit conv1 spaeter separat behandelt werden kann."""
    tensor = Activation("relu", name="relu1")(tensor)
    tensor = MaxPooling2D(2, name="pool1")(tensor)
    tensor = Conv2D(2, 3, padding="same", use_bias=False,
                    kernel_initializer=Constant(CONV2_KERNEL), name="conv2")(tensor)
    tensor = Activation("relu", name="relu2")(tensor)
    tensor = MaxPooling2D(2, name="pool2")(tensor)
    tensor = Flatten(name="flatten")(tensor)
    return Dense(1, use_bias=False, kernel_initializer=Constant(SCORE_KERNEL),
                 name="score")(tensor)


def build_cnn() -> Model:
    inputs = Input((GRID, GRID, 1), name="x")
    conv1 = Conv2D(3, 3, padding="same", use_bias=False,
                   kernel_initializer=Constant(CONV1_KERNEL), name="conv1")(inputs)
    return Model(inputs, _cnn_head(conv1), name="cnn_demo")


def build_cnn_tail() -> Model:
    """Dasselbe Netz ohne conv1: Eingang ist die Ausgabe von conv1."""
    inputs = Input((GRID, GRID, 3), name="conv1_out")
    return Model(inputs, _cnn_head(inputs), name="cnn_demo_tail")


cnn = build_cnn()
cnn.summary()

IMAGE = make_image()
X_IMAGE = IMAGE[None, ..., None]
Y_IMAGE = float(np.asarray(cnn(X_IMAGE)).ravel()[0])
print(f"\ny = {Y_IMAGE:.6g}")


# %% [markdown]
# ### 11.1 Werkzeuge für Teil II
#
# Vier kleine Helfer, die ab hier jede Abbildung tragen:
#
# * `lrp_map(...)` — eine Heatmap über dem Eingabebild, für beliebige LRP-Argumente.
# * `region_series(R)` — die Relevanzmasse je Region (Plus, Balken, Fleck, Hintergrund).
# * `show_map(...)` / `map_row(...)` — Heatmaps, immer symmetrisch um 0 (`seismic`: blau = negativ,
#   weiß = 0, rot = positiv), damit Vorzeichen und Nullpunkt über alle Bilder hinweg vergleichbar sind.
# * `plot_region_bars(...)` — dieselbe Information als Balken, weil man aus einer Heatmap keine
#   Summen ablesen kann.
#

# %%
from matplotlib.patches import Rectangle

# Regel-Farben (Teil I) um LRP-gamma erweitert und auf Farbsehschwaeche geprueft:
# alle Paare erreichen OKLab-DeltaE >= 8 unter simulierter Protanopie/Deuteranopie, ausser
# a1b0/a2b1 (6.9) -- dort trennen zusaetzlich Balkenposition und Legende.
RULE_COLORS.update({
    "γ": "#7b3294",
    "flat": "#dd3497",
    "LRP-0": "#595959",
})
RULE_ORDER = ["LRP-0", "ε", "γ", "α1β0", "α2β1", "flat"]


def lrp_map(model: Model, image: np.ndarray, **lrp_kwargs) -> np.ndarray:
    """Relevanz je Eingabepixel für Ausgabeneuron 0, als (GRID, GRID)-Array."""
    batch = np.asarray(image, dtype=np.float32).reshape((1, GRID, GRID, 1))
    explainer = LRP(model, layer=len(model.layers) - 1, idx=0, **lrp_kwargs)
    return np.asarray(explainer(batch), dtype=np.float64).reshape(GRID, GRID)


def region_series(relevance: np.ndarray) -> pd.Series:
    """Relevanzmasse je Region plus Gesamtsumme."""
    values = {name: float(relevance[MASKS[name]].sum()) for name in REGION_NAMES}
    values["Summe"] = float(relevance.sum())
    return pd.Series(values)


def _annotate_regions(ax, color: str = "#222222") -> None:
    """Gepunktete Kaesten um die drei Regionen, Beschriftung innerhalb des Kastens."""
    for name, (row, col) in POSITIONS.items():
        height, width = PATTERNS[name].shape
        ax.add_patch(Rectangle((col - 1.5, row - 1.5), width + 1, height + 1,
                               fill=False, edgecolor=color, linewidth=1.0, linestyle=":"))
        ax.text(col - 1.3, row - 1.35, name, color=color, fontsize=7,
                ha="left", va="top")


def show_map(ax, values: np.ndarray, *, title: str = "", cmap: str = "seismic",
             vmax: float = None, regions: bool = True):
    values = np.asarray(values, dtype=np.float64)
    if cmap == "seismic":
        limit = float(np.max(np.abs(values))) if vmax is None else float(vmax)
        limit = limit if limit > 0 else 1.0
        image = ax.imshow(values, cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest")
    else:
        image = ax.imshow(values, cmap=cmap, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9)
    if regions:
        _annotate_regions(ax, "#222222" if cmap == "seismic" else "#d8b365")
    return image


def map_row(maps: list[tuple[str, np.ndarray]], suptitle: str, *,
            shared_scale: bool = True, width: float = 2.4) -> None:
    """Eine Reihe Heatmaps.

    `shared_scale=True` legt eine gemeinsame Farbskala über alle Karten — dann sind die
    Beträge direkt vergleichbar und es gibt eine Farbleiste. `shared_scale=False` skaliert
    jede Karte einzeln; dann bekommt jede Karte ihre eigene Farbleiste und ihr Maximum in
    den Titel, weil eine gemeinsame Leiste dort falsch wäre.
    """
    limit = max(float(np.max(np.abs(values))) for _, values in maps) if shared_scale else None
    extra = 0.9 if shared_scale else 1.5
    fig, axes = plt.subplots(1, len(maps),
                             figsize=(width * len(maps) + extra, width + 1.0))
    axes = np.atleast_1d(axes)
    for ax, (label, values) in zip(axes, maps):
        title = label if shared_scale else f"{label}\n|R| max = {np.max(np.abs(values)):.3g}"
        image = show_map(ax, values, title=title, vmax=limit)
        if not shared_scale:
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    if shared_scale:
        fig.colorbar(image, ax=axes, fraction=0.028, pad=0.02, label="Relevanz")
    else:
        fig.tight_layout()
    fig.suptitle(suptitle, fontsize=10)
    plt.show()


def plot_region_bars(series: dict[str, pd.Series], title: str,
                     names: list[str] = None) -> pd.DataFrame:
    """Relevanzmasse je Region, eine Balkengruppe je Regel."""
    names = names or REGION_NAMES
    frame = pd.DataFrame(series).loc[names]
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    positions = np.arange(len(names))
    width = 0.8 / frame.shape[1]
    for index, label in enumerate(frame.columns):
        offset = (index - (frame.shape[1] - 1) / 2) * width
        ax.bar(positions + offset, frame[label].to_numpy(), width=width * 0.9,
               label=label, color=RULE_COLORS.get(label, "#777777"))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(positions, names)
    ax.set_ylabel("Relevanzmasse")
    ax.set_title(title)
    ax.legend(frameon=False, ncol=min(frame.shape[1], 6), fontsize=8)
    fig.tight_layout()
    plt.show()
    return frame


# %% [markdown]
# ### 11.2 Was die drei Regionen wirklich bewirken
#
# Bevor irgendeine Heatmap gezeichnet wird, wird die **Wahrheit** bestimmt, gegen die alle Regeln
# geprüft werden: Region ausradieren, Vorhersage neu rechnen, Differenz ablesen. Das ist keine
# Erklärungsmethode, sondern eine direkte Messung — hier möglich, weil das Netz winzig und das Bild
# konstruiert ist.
#

# %%
conv1_layer = cnn.get_layer("conv1")
relu1_model = Model(cnn.input, cnn.get_layer("relu1").output)
activations = np.asarray(relu1_model(X_IMAGE))[0]

print("relu1, Maximum je Kanal im gesamten Bild:",
      np.round(activations.max(axis=(0, 1)), 4))
for name in ["Plus", "Balken", "Fleck"]:
    print(f"  in Region {name:7s}:", np.round(activations[MASKS[name]].max(axis=0), 4))

ablation = {}
for name in ["Plus", "Balken", "Fleck"]:
    blanked = np.where(MASKS[name], 0.0, IMAGE)
    y_blanked = float(np.asarray(cnn(blanked[None, ..., None])).ravel()[0])
    ablation[name] = Y_IMAGE - y_blanked
ablation_series = pd.Series(ablation)
print(f"\ny = {Y_IMAGE:.6g}")
print("Kausaler Beitrag (y minus y ohne Region):")
display(ablation_series.to_frame("Δy").style.format("{:+.4f}"))

fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
show_map(axes[0], IMAGE, title="Eingabebild", cmap="Greys_r")
axes[1].bar(list(ablation), [ablation[k] for k in ablation],
            color=["#2c7fb8" if ablation[k] >= 0 else "#e34a33" for k in ablation], width=0.6)
axes[1].axhline(0.0, color="black", linewidth=0.8)
axes[1].set_title("Kausaler Beitrag Δy", fontsize=9)
axes[1].set_ylabel("Δy")
axes[2].bar(["Plus", "Balken", "Fleck"],
            [float(IMAGE[MASKS[n]].max()) for n in ["Plus", "Balken", "Fleck"]],
            color="#595959", width=0.6)
axes[2].set_title("Maximale Helligkeit", fontsize=9)
axes[2].set_ylabel("Pixelwert")
fig.tight_layout()
plt.show()

np.testing.assert_allclose(ablation["Fleck"], 0.0, atol=1e-6)
assert ablation["Plus"] > 0 > ablation["Balken"]

# %% [markdown]
# <a id="sec-cnn-lrp0"></a>
# ## 12. LRP-0 im CNN — die Referenz
#
# [↑ Inhalt](#top)
#
# **Frage.** Wie verteilt sich der Score auf die Pixel, wenn man nichts stabilisiert und nichts
# bevorzugt?
#
# **Regel.** $z_{jk} = x_j w_{jk}$, also genau die Beitragszerlegung aus Teil I, Abschnitt 1 — nur
# jetzt über mehrere Schichten hinweg und mit geteilten Gewichten.
#
# **Was am Bild sichtbar werden soll.**
#
# * Das **Plus** leuchtet rot, der **Balken** blau. LRP-0 behandelt beide Richtungen gleich.
# * Der **Fleck** bleibt weiß — obwohl er der hellste Bereich des Bildes ist. Aktivierung allein
#   erzeugt keine Relevanz; sie muss auch ein Gewicht finden, über das sie die Ausgabe erreicht.
# * Der **Hintergrund** bleibt exakt 0, weil dort $x_j = 0$ steht und jeder Term $x_j w_{jk}$
#   verschwindet.
# * Die Summe ist $y$ — das Netz hat keinen Bias, also ist die Erhaltung exakt.
#
# Das Paper hält LRP-0 gleichzeitig für die *schwächste* Wahl in tiefen Netzen: sie folgt der
# Funktion und ihrem Gradienten und erbt damit das *gradient shattering*. In der DTD-Sprache liegt
# der Wurzelpunkt im Ursprung, also weit weg von den Daten. Unser Netz ist zu flach, um das zu
# zeigen — Abschnitt 19 zeigt stattdessen den anderen Schwachpunkt: was passiert, wenn $y \to 0$.
#

# %%
R_lrp0 = lrp_map(cnn, IMAGE)
summary_lrp0 = region_series(R_lrp0)
display(summary_lrp0.to_frame("LRP-0").style.format("{:+.4f}"))

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
show_map(axes[0], IMAGE, title="Eingabe", cmap="Greys_r")
image = show_map(axes[1], R_lrp0, title="LRP-0")
fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.03, label="Relevanz")
fig.suptitle("LRP-0: der hellste Bereich (Fleck) bekommt nichts", fontsize=10)
fig.tight_layout()
plt.show()

np.testing.assert_allclose(summary_lrp0["Summe"], Y_IMAGE, atol=1e-5)
np.testing.assert_allclose(summary_lrp0["Fleck"], 0.0, atol=1e-6)
np.testing.assert_allclose(summary_lrp0["Hintergrund"], 0.0, atol=1e-6)


# %% [markdown]
# <a id="sec-cnn-eps"></a>
# ## 13. LRP-ε — stabilisieren und ausdünnen
#
# [↑ Inhalt](#top)
#
# **Frage.** Welche Teile der Erklärung sind belastbar genug, um stehen zu bleiben?
#
# **Regel.** $\varepsilon$ wächst den Nenner:
# $R_j = \sum_k \frac{x_j w_{jk}}{\varepsilon + z_k} R_k$. Neuronen, deren Nettobeitrag $z_k$ klein
# gegen $\varepsilon$ ist, geben fast nichts weiter. Das Paper nennt das den **Sparsifizierungs-Effekt**:
# „the relevance of neurons with weak net contributions is driven to zero by the stabilization term“.
#
# **Empfehlung des Papers.** fc-Schichten im oberen Teil des Netzes und die obersten
# Convolution-Schichten.
#
# **Was am Bild sichtbar werden soll.** Die *Form* der Erklärung bleibt (Plus rot, Balken blau,
# Fleck und Hintergrund leer), aber der *Betrag* fällt — und zwar nicht gleichmäßig: schwache
# Beiträge verschwinden zuerst. Die rechte Abbildung zählt beides mit:
#
# * **erhaltene Masse** $\sum_j R_j / y$ — geht mit wachsendem $\varepsilon$ gegen 0;
# * **Konzentration** — der Anteil der Relevanzmasse, der in den 3 stärksten Pixeln steckt. Er
#   *steigt*: was bleibt, ist die Spitze. (Nur 8 der 144 Pixel tragen hier überhaupt Relevanz;
#   deshalb misst man die Konzentration an wenigen Pixeln, nicht an zehn.)
#

# %%
EPSILONS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

epsilon_rows = []
epsilon_maps = {}
for epsilon in EPSILONS:
    kwargs = {} if epsilon == 0 else {"epsilon": float(epsilon)}
    relevance = lrp_map(cnn, IMAGE, **kwargs)
    epsilon_maps[epsilon] = relevance
    absolute = np.abs(relevance)
    top3 = np.sort(absolute.ravel())[-3:].sum()
    epsilon_rows.append({
        "ε": epsilon,
        "Plus": float(relevance[MASKS["Plus"]].sum()),
        "Balken": float(relevance[MASKS["Balken"]].sum()),
        "Summe R": float(relevance.sum()),
        "Summe / y": float(relevance.sum()) / Y_IMAGE,
        "nicht-leere Pixel": int((absolute > 1e-9).sum()),
        "Konzentration Top-3": float(top3 / absolute.sum()),
    })

epsilon_table = pd.DataFrame(epsilon_rows)
display(epsilon_table.style.format({
    "ε": "{:g}", "Plus": "{:+.4f}", "Balken": "{:+.4f}",
    "Summe R": "{:+.4f}", "Summe / y": "{:.4f}", "Konzentration Top-3": "{:.3f}",
}).hide(axis="index"))

map_row([(f"ε = {e:g}" if e else "LRP-0", epsilon_maps[e]) for e in [0.0, 0.1, 1.0, 10.0]],
        "LRP-ε: gleiche Form, immer weniger Masse", shared_scale=False)

fig, ax = plt.subplots(figsize=(7.4, 3.6))
positions = np.arange(len(epsilon_table))
ax.plot(positions, epsilon_table["Summe / y"], marker="o", color=RULE_COLORS["ε"],
        label="erhaltene Masse  Σ R / y")
ax.plot(positions, epsilon_table["Konzentration Top-3"], marker="s", linestyle="--",
        color=RULE_COLORS["LRP-0"], label="Konzentration in den 3 stärksten Pixeln")
ax.set_xticks(positions, [f"{e:g}" for e in epsilon_table["ε"]])
ax.set_xlabel("ε")
ax.set_ylabel("Anteil")
ax.set_title("ε schluckt Masse und verdichtet, was übrig bleibt")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
plt.show()

assert epsilon_table["Summe / y"].is_monotonic_decreasing
assert epsilon_table["Konzentration Top-3"].is_monotonic_increasing


# %% [markdown]
# <a id="sec-cnn-gamma"></a>
# ## 14. LRP-γ — positive Beiträge bevorzugen
#
# [↑ Inhalt](#top)
#
# **Frage.** Convolution-Schichten sind stark nichtlinear; einzelne Pixel sauber in „dafür“ und
# „dagegen“ zu trennen, gelingt dort kaum. Kann man stattdessen *Gruppen* von Pixeln gemeinsam
# Relevanz geben und dabei die stützende Seite bevorzugen?
#
# **Regel.** $\gamma$ verstärkt nur die positiven Gewichte:
# $w_{jk} \to w_{jk} + \gamma (w_{jk})^+$. Im Repo steht dafür genau eine Zeile
# (`explainability/layers/layer.py`):
#
# ```python
# if self.gamma:
#     w = tf.where(w >= 0, tf.multiply(w, 1 + self.gamma), w)
# ```
#
# Für $w \ge 0$ ist $w + \gamma w = w(1+\gamma)$, für $w < 0$ ist $(w)^+ = 0$ — die Zeile ist die
# Formel aus Tabelle 1.1.
#
# **Empfehlung des Papers.** Convolution-Schichten. LRP-γ hat eine DTD-Deutung (Wurzelpunkt am
# ReLU-Knick, also nahe am Datenpunkt) — anders als LRP-αβ mit $\alpha \ne 1$.
#
# **Was sichtbar werden soll.** Zwei Dinge auf einmal:
#
# 1. Die **Summe bleibt exakt $y$**, für jedes $\gamma$. $\gamma$ verschiebt nur, es schluckt nicht.
#    Das ist der klare Unterschied zu $\varepsilon$.
# 2. Die negative Masse schrumpft monoton, und **für $\gamma \to \infty$ läuft LRP-γ gegen
#    LRP-α₁β₀** — genau die Aussage aus Kapitel 1.4.1. Die rechte Kurve misst den Abstand
#    $\lVert R_\gamma - R_{\alpha_1\beta_0}\rVert_1$ und zeigt, dass er gegen 0 geht.
#

# %%
GAMMAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 25.0, 100.0]

R_a1b0 = lrp_map(cnn, IMAGE, alpha=1.0, beta=0.0)

gamma_rows = []
gamma_maps = {}
for gamma in GAMMAS:
    kwargs = {} if gamma == 0 else {"gamma": float(gamma)}
    relevance = lrp_map(cnn, IMAGE, **kwargs)
    gamma_maps[gamma] = relevance
    negative = float(np.abs(np.minimum(relevance, 0.0)).sum())
    gamma_rows.append({
        "γ": gamma,
        "Plus": float(relevance[MASKS["Plus"]].sum()),
        "Balken": float(relevance[MASKS["Balken"]].sum()),
        "Summe R": float(relevance.sum()),
        "negativer Anteil": negative / float(np.abs(relevance).sum()),
        "Abstand zu α1β0": float(np.abs(relevance - R_a1b0).sum()),
    })

gamma_table = pd.DataFrame(gamma_rows)
display(gamma_table.style.format({
    "γ": "{:g}", "Plus": "{:+.4f}", "Balken": "{:+.4f}", "Summe R": "{:+.4f}",
    "negativer Anteil": "{:.3f}", "Abstand zu α1β0": "{:.4f}",
}).hide(axis="index"))

map_row([("LRP-0", gamma_maps[0.0]), ("γ = 0.25", gamma_maps[0.25]),
         ("γ = 1", gamma_maps[1.0]), ("γ = 100", gamma_maps[100.0]),
         ("α1β0", R_a1b0)],
        "γ dreht LRP-0 stetig in Richtung α1β0")

fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
positions = np.arange(len(gamma_table))
axes[0].plot(positions, gamma_table["negativer Anteil"], marker="o", color=RULE_COLORS["γ"])
axes[0].set_xticks(positions, [f"{g:g}" for g in gamma_table["γ"]], rotation=45, ha="right")
axes[0].set_xlabel("γ")
axes[0].set_ylabel("Anteil negativer Relevanz")
axes[0].set_title("Die widersprechende Seite verschwindet", fontsize=9)
axes[1].plot(positions, gamma_table["Abstand zu α1β0"], marker="o", color=RULE_COLORS["γ"])
axes[1].set_xticks(positions, [f"{g:g}" for g in gamma_table["γ"]], rotation=45, ha="right")
axes[1].set_xlabel("γ")
axes[1].set_ylabel("Σ |R(γ) − R(α1β0)|")
axes[1].set_yscale("log")
axes[1].set_title("γ → ∞ läuft gegen α1β0", fontsize=9)
fig.tight_layout()
plt.show()

np.testing.assert_allclose(gamma_table["Summe R"], Y_IMAGE, atol=1e-4)
assert gamma_table["negativer Anteil"].is_monotonic_decreasing
assert gamma_table["Abstand zu α1β0"].iloc[-1] < gamma_table["Abstand zu α1β0"].iloc[0] / 20


# %% [markdown]
# <a id="sec-cnn-ab"></a>
# ## 15. LRP-αβ — die beiden Seiten getrennt normieren
#
# [↑ Inhalt](#top)
#
# **Frage.** Dieselbe wie bei γ — positive Beiträge bevorzugen — aber mit einem festen Verhältnis
# statt einer Gewichtsverzerrung.
#
# **Regel.** Positive und negative Beiträge werden **getrennt** normiert und mit $\alpha$ bzw.
# $\beta$ gewichtet, $\alpha - \beta = 1$. Im Repo ist $\alpha = \beta + 1$ als `assert` erzwungen.
#
# **Empfehlung des Papers.** Convolution-Schichten. Und als Default, wenn man ein neues
# ReLU-Netz vor sich hat: *„a default configuration that can be tried is to apply the LRP-αβ rule
# with α = 1, β = 0 in every hidden layer“* — der Vorteil ist, dass es **keinen freien
# Hyperparameter** gibt. Das Paper schränkt aber auch ein: auf typischen Bild-CNNs ist α₁β₀ oft
# *zu wenig selektiv*, während dieselbe Regel beim Pruning und beim Relation Network sehr gut
# funktioniert hat.
#
# **Was am Bild sichtbar werden soll.**
#
# * **α1β0** liefert eine rein positive Karte. Der Balken, der den Score nachweislich *senkt*
#   ($\Delta y < 0$, Abschnitt 11.2), bekommt trotzdem eine **positive** Relevanz. Genau das meint
#   „zu wenig selektiv“: die Regel kann nicht sagen, was dagegen spricht.
# * **α2β1** zeigt beide Richtungen, aber die Summe ist nicht mehr $y$. Der Grund ist derselbe wie
#   in Teil I, Abschnitt 3 — nur verstärkt: wo eine Schicht keine negativen Beiträge hat, entfällt
#   der β-Term und $\alpha = 2$ verdoppelt die Masse, und über drei Schichten multipliziert sich das.
# * Fleck und Hintergrund bleiben auch hier bei 0: αβ multipliziert weiterhin mit $x_j$.
#

# %%
R_a2b1 = lrp_map(cnn, IMAGE, alpha=2.0, beta=1.0)

ab_frame = pd.DataFrame({
    "LRP-0": region_series(R_lrp0),
    "γ = 0.25": region_series(gamma_maps[0.25]),
    "α1β0": region_series(R_a1b0),
    "α2β1": region_series(R_a2b1),
})
display(ab_frame.style.format("{:+.4f}"))
print(f"Summe α2β1 / y = {region_series(R_a2b1)['Summe'] / Y_IMAGE:.3f}")

map_row([("LRP-0", R_lrp0), ("α1β0", R_a1b0), ("α2β1", R_a2b1)],
        "α1β0 kennt nur „dafür“, α2β1 zeigt beide Seiten", shared_scale=False)

plot_region_bars(
    {"LRP-0": region_series(R_lrp0), "γ": region_series(gamma_maps[0.25]),
     "α1β0": region_series(R_a1b0), "α2β1": region_series(R_a2b1)},
    "Der Balken senkt den Score — nur LRP-0, γ und α2β1 sagen das auch",
)

assert float(R_a1b0.min()) >= -1e-6, "α1β0 sollte keine negative Relevanz erzeugen"
assert R_a1b0[MASKS["Balken"]].sum() > 0 > ablation["Balken"]
np.testing.assert_allclose(region_series(R_a1b0)["Summe"], Y_IMAGE, atol=1e-4)
assert R_a2b1[MASKS["Balken"]].sum() < 0


# %% [markdown]
# <a id="sec-cnn-flat"></a>
# ## 16. flat — das rezeptive Feld statt des Merkmals
#
# [↑ Inhalt](#top)
#
# **Frage.** Manchmal will man gar nicht wissen, *welches* Pixel im Fenster zählt, sondern nur,
# *welcher Bildbereich* überhaupt beteiligt war. Wie sieht eine Erklärung aus, die genau das tut?
#
# **Regel.** $R_j = \sum_k \frac{1}{n_{i-1}} R_k$ — jeder Eingang des Fensters bekommt denselben
# Anteil. Im Repo: `{"flat": True}`, intern `a ← 1`, `w ← 1` vor der z-Regel.
#
# **Empfehlung des Papers.** „to reduce the spatial resolution of heatmaps (by simply uniformly
# redistributing relevance from some intermediate layer onto the input)“ und „if we simply want to
# highlight the receptive fields rather than the contributing features within the receptive field“.
#
# **Was sichtbar werden soll.** Die entscheidende Beobachtung betrifft die **Nullpixel** — alle
# Pixel mit Wert exakt 0, also fast das ganze Bild. Keine datenabhängige Regel kann ihnen Relevanz
# geben, weil jeder Term $x_j w_{jk}$ dort verschwindet. `flat` gibt ihnen trotzdem welche, denn
# `flat` fragt nicht nach dem Pixelwert. Und je weiter oben man `flat` ansetzt, desto größer ist das
# rezeptive Feld, über das verschmiert wird:
#
# | `flat` auf | verschmiert über | Effekt |
# |---|---|---|
# | nur `conv1` | 3×3 Pixel | leichte Verbreiterung, Muster noch erkennbar |
# | `conv1` + `conv2` | 3×3 nach Pooling ⇒ ~6×6 Pixel | deutlich grobkörniger |
# | alle drei Schichten | das ganze Bild | nur noch Blöcke, kein Muster mehr |
#
# Die Zeile *Hintergrund* der Tabelle misst etwas anderes als die Abbildung: die Regionsmasken
# enthalten bereits einen Rand von einem Pixel — also genau das rezeptive Feld von `conv1`. Deshalb
# steht dort bei `flat auf conv1` noch eine 0, während die Abbildung, die alle Nullpixel zählt,
# schon einen Ausschlag zeigt.
#
# **Nebeneffekt, den man kennen sollte.** `flat` ganz oben (nur auf `score`) verteilt Relevanz auch
# auf Zellen, deren ReLU-Ausgabe 0 ist. Darunter teilt die z-Regel durch $z = 0$ und setzt den
# Anteil auf 0 — diese Relevanz geht verloren. Die Summe ist dann kleiner als $y$.
#

# %%
flat_variants = {
    "LRP-0": {},
    "flat auf conv1": {"strategy": LRPStrategy(layers=[{"flat": True}, {}, {}])},
    "flat auf conv1+conv2": {"strategy": LRPStrategy(layers=[{"flat": True}, {"flat": True}, {}])},
    "flat überall": {"strategy": LRPStrategy(layers=[{"flat": True}] * 3)},
    "flat nur auf score": {"strategy": LRPStrategy(layers=[{}, {}, {"flat": True}])},
}

flat_maps = {label: lrp_map(cnn, IMAGE, **kwargs) for label, kwargs in flat_variants.items()}
flat_frame = pd.DataFrame({label: region_series(values) for label, values in flat_maps.items()})
display(flat_frame.style.format("{:+.4f}"))

map_row([(label, flat_maps[label]) for label in
         ["LRP-0", "flat auf conv1", "flat auf conv1+conv2", "flat überall"]],
        "Je weiter oben flat ansetzt, desto gröber die Karte", shared_scale=False)

ZERO_PIXELS = IMAGE == 0.0
print(f"{int(ZERO_PIXELS.sum())} von {GRID * GRID} Pixeln sind exakt 0.")

fig, ax = plt.subplots(figsize=(7.8, 3.6))
labels = ["LRP-0", "flat auf conv1", "flat auf conv1+conv2", "flat überall"]
share = [float(np.abs(flat_maps[label][ZERO_PIXELS]).sum() / np.abs(flat_maps[label]).sum())
         for label in labels]
ax.bar(np.arange(len(labels)), share, color=RULE_COLORS["flat"], width=0.6)
ax.set_xticks(np.arange(len(labels)), labels, rotation=12, ha="right", fontsize=8)
ax.set_ylabel("Anteil der |Relevanz| auf Nullpixeln")
ax.set_title("Pixel mit Wert 0 — nur flat gibt ihnen Relevanz")
fig.tight_layout()
plt.show()

np.testing.assert_allclose(flat_frame.loc["Hintergrund", "LRP-0"], 0.0, atol=1e-6)
assert flat_frame.loc["Hintergrund", "flat überall"] > 0.1 * Y_IMAGE
np.testing.assert_allclose(share[0], 0.0, atol=1e-9)
assert share[0] < share[1] < share[-1]
assert flat_frame.loc["Summe", "flat nur auf score"] < Y_IMAGE - 1e-3


# %% [markdown]
# <a id="sec-cnn-first"></a>
# ## 17. Die erste Schicht ist ein Sonderfall: w²-Regel und z$^\mathcal{B}$-Regel
#
# [↑ Inhalt](#top)
#
# **Warum überhaupt ein Sonderfall.** LRP-0, LRP-ε, LRP-γ und LRP-αβ haben in der DTD-Deutung alle
# einen Wurzelpunkt mit **positiven** Komponenten. Das passt zu Schichten, deren Eingang eine
# ReLU-Ausgabe ist — dort ist alles $\ge 0$. Die **erste** Schicht bekommt aber Pixel, und Pixel
# haben einen ganz anderen Wertebereich: entweder einen beschränkten Kasten $[l_i, h_i]$ (Bilder)
# oder den ganzen $\mathbb{R}^d$ (z-standardisierte Daten, wie die MRT-Volumen in diesem Repo).
# Genau dafür sind die beiden letzten Zeilen von Tabelle 1.1 gemacht.
#
# **w²-Regel** ($\mathbb{R}^d$, keine Bereichsannahme):
# $R_i^{[0]} = \sum_j \frac{(w_{ij}^{[1]})^2}{\sum_l (w_{lj}^{[1]})^2} R_j^{[1]}$.
# Der Pixelwert kommt darin **überhaupt nicht vor**. Innerhalb eines rezeptiven Felds verteilt sich
# die Relevanz allein nach der Gewichtsstärke.
#
# **z$^\mathcal{B}$-Regel** (Pixel in $[l_i, h_i]$):
# $R_i^{[0]} = \sum_j \frac{x_i w_{ij} - l_i (w_{ij})^+ - h_i (w_{ij})^-}
# {\sum_l \big(x_l w_{lj} - l_l (w_{lj})^+ - h_l (w_{lj})^-\big)} R_j^{[1]}$.
# Sie misst das Pixel nicht gegen 0, sondern **gegen die Ränder seines Kastens**. Ein Pixel mit Wert
# 0 kann darin Relevanz bekommen, wenn 0 nicht der Rand ist.
#
# **Beide sind in `explainability` nicht implementiert.** Das ist auch nicht schlimm: man braucht
# sie nur für *eine* Schicht. Vorgehen hier:
#
# 1. Ein **Tail-Modell** bauen, dessen Eingang die Ausgabe von `conv1` ist (`build_cnn_tail()`).
# 2. Mit der Bibliothek ganz normal bis dorthin zurückpropagieren → $R^{[1]}$.
# 3. Den letzten Schritt (`conv1` → Pixel) von Hand rechnen, in beiden Varianten.
#
# Dass dieses Vorgehen korrekt angeschlossen ist, wird zweifach geprüft: die z-Regel von Hand muss
# exakt das volle `LRP-0` reproduzieren, und die z$^\mathcal{B}$-Regel mit dem entarteten Kasten
# $l = h = 0$ ebenfalls — setzt man $l = h = 0$ ein, fällt sie auf die z-Regel zurück.
#

# %%
cnn_tail = build_cnn_tail()
CONV1_OUT = np.asarray(conv1_layer(X_IMAGE))
print("Tail-Modell liefert dieselbe Vorhersage:",
      float(np.asarray(cnn_tail(CONV1_OUT)).ravel()[0]), "≈", Y_IMAGE)

W_CONV1 = np.asarray(conv1_layer.kernel, dtype=np.float64)


def relevance_at_conv1_output(**lrp_kwargs) -> np.ndarray:
    explainer = LRP(cnn_tail, layer=len(cnn_tail.layers) - 1, idx=0, **lrp_kwargs)
    return np.asarray(explainer(CONV1_OUT), dtype=np.float64)


def _conv(activation, kernel):
    return np.asarray(tf.nn.conv2d(tf.constant(activation, tf.float64),
                                   tf.constant(kernel, tf.float64),
                                   strides=[1, 1, 1, 1], padding="SAME"))


def _conv_transpose(share, kernel, shape):
    return np.asarray(tf.nn.conv2d_transpose(tf.constant(share, tf.float64),
                                             tf.constant(kernel, tf.float64),
                                             output_shape=shape,
                                             strides=[1, 1, 1, 1], padding="SAME"))


def _share(relevance, z):
    out = np.zeros_like(z)
    nonzero = z != 0
    out[nonzero] = relevance[nonzero] / z[nonzero]
    return out


def first_layer_z(a, R):
    """z-Regel (LRP-0) auf conv1 — nur zur Kontrolle."""
    z = _conv(a, W_CONV1)
    return a * _conv_transpose(_share(R, z), W_CONV1, a.shape)


def first_layer_w2(a, R):
    """w²-Regel: der Pixelwert kommt nicht vor."""
    squared = W_CONV1 ** 2
    z = _conv(np.ones_like(a), squared)
    return _conv_transpose(_share(R, z), squared, a.shape)


def first_layer_zB(a, R, low, high):
    """z^B-Regel für den Kasten [low, high]."""
    w_pos, w_neg = np.maximum(W_CONV1, 0.0), np.minimum(W_CONV1, 0.0)
    lows, highs = np.full_like(a, low), np.full_like(a, high)
    z = _conv(a, W_CONV1) - _conv(lows, w_pos) - _conv(highs, w_neg)
    share = _share(R, z)
    return (a * _conv_transpose(share, W_CONV1, a.shape)
            - lows * _conv_transpose(share, w_pos, a.shape)
            - highs * _conv_transpose(share, w_neg, a.shape))


A_PIXELS = np.asarray(X_IMAGE, dtype=np.float64)
R_CONV1_OUT = relevance_at_conv1_output()
BOX = (float(IMAGE.min()), float(IMAGE.max()))
print(f"Pixel-Kasten [l, h] = [{BOX[0]:g}, {BOX[1]:g}]")

R_first = {
    "LRP-0 (z-Regel)": first_layer_z(A_PIXELS, R_CONV1_OUT).reshape(GRID, GRID),
    "w²-Regel": first_layer_w2(A_PIXELS, R_CONV1_OUT).reshape(GRID, GRID),
    f"z^B-Regel [{BOX[0]:g}, {BOX[1]:g}]": first_layer_zB(A_PIXELS, R_CONV1_OUT, *BOX).reshape(GRID, GRID),
}

# Anschluss-Kontrollen
np.testing.assert_allclose(R_first["LRP-0 (z-Regel)"], R_lrp0, atol=1e-6)
np.testing.assert_allclose(first_layer_zB(A_PIXELS, R_CONV1_OUT, 0.0, 0.0).reshape(GRID, GRID),
                           R_lrp0, atol=1e-6)
print("Kontrolle bestanden: z-Regel von Hand == volles LRP-0, und z^B mit [0,0] ebenfalls.")

first_frame = pd.DataFrame({label: region_series(values) for label, values in R_first.items()})
display(first_frame.style.format("{:+.4f}"))
map_row(list(R_first.items()), "Dieselbe Relevanz aus conv1, drei Regeln für den letzten Schritt")

for label, values in R_first.items():
    np.testing.assert_allclose(float(values.sum()), Y_IMAGE, atol=1e-4,
                               err_msg=f"{label} ist nicht konservativ")


# %% [markdown]
# ### 17.1 Was die beiden Regeln jeweils steuert
#
# **w²: die Aufteilung im Fenster hängt nur am Filter.** Die drei linken Abbildungen zeigen für
# jeden `conv1`-Filter das normierte Muster $w^2 / \sum w^2$ — das ist der Schlüssel, nach dem die
# Relevanz eines Ausgabeneurons auf seine 3×3 Pixel verteilt wird, **unabhängig vom Bild**. Beim
# Plus-Filter fließen $4 \times 0.20$ in die vier *Ecken*, weil dort die betragsgrößten Gewichte
# stehen ($-3$) — obwohl diese Ecken im Plus-Muster gar nicht gesetzt sind. Das Zentrum, das das
# Muster ausmacht, bekommt 0.02. Balken- und Fleck-Filter haben dagegen überall denselben Betrag,
# dort verteilt w² gleichmäßig ($1/9 = 0.11$).
#
# **z$^\mathcal{B}$: der Kasten steuert, nicht der Rohwert.** Die rechte Abbildung fährt die obere
# Kastengrenze $h$ von 0 nach oben. Bei $h = l = 0$ ist die Regel identisch mit LRP-0 (alle
# Nullpixel leer). Sobald der Kasten aufgeht, wandert Relevanz auf Nullpixel — sie sind jetzt nicht
# mehr „nichts“, sondern „am unteren Rand des Bereichs“. Genau deshalb muss man für z$^\mathcal{B}$
# den **tatsächlichen Wertebereich der Eingabe** kennen; ein falsch gesetzter Kasten ist eine
# falsche Annahme, kein Hyperparameter zum Drehen.
#

# %%
filter_names = ["Plus-Filter", "Balken-Filter", "Fleck-Filter"]
w2_shares = [W_CONV1[:, :, 0, i] ** 2 / (W_CONV1[:, :, 0, i] ** 2).sum()
             for i in range(len(filter_names))]
w2_max = max(float(share.max()) for share in w2_shares)

fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.2))
for index, (name, share) in enumerate(zip(filter_names, w2_shares)):
    ax = axes[index]
    ax.imshow(share, cmap="Blues", vmin=0.0, vmax=w2_max, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.grid(False, which="major")
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"w² normiert\n{name}", fontsize=9)
    for (row, col), value in np.ndenumerate(share):
        ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if value > 0.6 * w2_max else "#1a1a1a")

box_heights = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
zero_share = []
for high in box_heights:
    values = first_layer_zB(A_PIXELS, R_CONV1_OUT, 0.0, high).reshape(GRID, GRID)
    zero_share.append(float(np.abs(values[ZERO_PIXELS]).sum() / np.abs(values).sum()))
axes[3].plot(box_heights, zero_share, marker="o", color=RULE_COLORS["ε"])
axes[3].set_xlabel("obere Kastengrenze h  (l = 0)")
axes[3].set_ylabel("Anteil |R| auf Nullpixeln")
axes[3].set_title("z^B: der Kasten entscheidet", fontsize=9)
fig.tight_layout()
plt.show()

np.testing.assert_allclose(zero_share[0], 0.0, atol=1e-9)
assert zero_share[-1] > zero_share[0]


# %% [markdown]
# <a id="sec-cnn-pool"></a>
# ## 18. Pooling- und BatchNorm-Schichten
#
# [↑ Inhalt](#top)
#
# Das Paper behandelt beide knapp, und beide sind in diesem Repo fertig vorhanden.
#
# **Max-Pooling — winner-take-all.** Die gesamte Relevanz des Fensters geht an das Maximum. Im Repo
# ist das der Default (`MaxPoolingLRP(strategy='winner-takes-all')`), technisch über den Gradienten
# von `tf.nn.max_pool`.
#
# **Average-Pooling — lineare Schicht.** Average-Pooling ist eine lineare Schicht mit positiven
# konstanten Gewichten, also gelten die Regeln von oben unverändert. Im Repo heißt diese Variante
# `'redistribute'` (Default für `AveragePoolingLRP`) und entspricht der z-Regel.
#
# **`'flat'` für Pooling.** Zusätzlich kennt das Repo `strategy='flat'`: die Relevanz wird
# gleichmäßig über das Fenster verteilt, unabhängig von den Aktivierungen. Das ist die
# Pooling-Variante des `flat`-Gedankens — aber Vorsicht, in dieser Implementierung ist sie
# **nicht konservativ**; die Tabelle unten zeigt das.
#
# **BatchNorm.** Das Paper empfiehlt, BatchNorm *vor* LRP mit der benachbarten conv/fc-Schicht zu
# verschmelzen, damit LRP nur noch conv/fc, ReLU und Pooling sieht. Genau das macht
# `fuse_batchnorm`, und `LRP.__init__` ruft es automatisch auf — man muss nichts konfigurieren.
#
# **Was am Beispiel sichtbar wird.** Unser Netz hat zwei `MaxPooling2D`-Schichten. `winner-takes-all`
# hält die Karte scharf: nur die Pixel, die den Maximalwert eines Fensters erzeugt haben, bekommen
# etwas. `redistribute` zieht die Relevanz über das ganze Fenster und weicht die Ränder auf.
#

# %%
pool_variants = {
    "winner-takes-all (Default)": "winner-takes-all",
    "redistribute": "redistribute",
    "flat": "flat",
}
pool_maps = {label: lrp_map(cnn, IMAGE,
                            strategy=LRPStrategy(pooling=[{"strategy": strategy}] * 2))
             for label, strategy in pool_variants.items()}
pool_frame = pd.DataFrame({label: region_series(values) for label, values in pool_maps.items()})
pool_frame.loc["Summe / y"] = pool_frame.loc["Summe"] / Y_IMAGE
display(pool_frame.style.format("{:+.4f}"))

map_row(list(pool_maps.items()), "Dieselbe Regel auf den Gewichtsschichten, drei Pooling-Strategien")

np.testing.assert_allclose(pool_frame.loc["Summe", "winner-takes-all (Default)"], Y_IMAGE, atol=1e-4)
np.testing.assert_allclose(pool_frame.loc["Summe", "redistribute"], Y_IMAGE, atol=1e-4)
assert abs(pool_frame.loc["Summe / y", "flat"] - 1.0) > 1e-3, \
    "die flat-Pooling-Strategie ist erwartungsgemäß nicht konservativ"


# %% [markdown]
# <a id="sec-cnn-einfluss"></a>
# ## 19. Welcher Input steuert welche Regel?
#
# [↑ Inhalt](#top)
#
# Bis hierher stand jedes Mal *ein* Bild vor *mehreren* Regeln. Jetzt umgekehrt: eine
# Bildeigenschaft wird stetig verändert, und jede Regel muss sagen, wie sie darauf reagiert. Drei
# Durchläufe, drei Fragen.
#
# ### 19.1 Helligkeit ohne Wirkung
#
# Die Helligkeit des Flecks läuft von 0 bis 4. Der Fleck erreicht nur Kanal 2, und Kanal 2 hat in
# `conv2` das Gewicht 0 — **$y$ bleibt konstant**, egal wie hell er wird. Eine Regel, die ihm
# Relevanz gibt, reagiert auf Helligkeit statt auf Wirkung.
#
# ### 19.2 Widerspruch, der stärker wird
#
# Die Stärke des Balkens läuft von 0 bis 1.4. $y$ fällt dabei von $+2.44$ auf fast 0 — der Balken
# frisst den Score auf. LRP-0, ε, γ und α2β1 zeigen das als wachsende negative Relevanz. **α1β0
# kann es nicht zeigen**, weil es keine negative Relevanz kennt; dort bleibt der Balken positiv.
# Das ist konkret die Einschränkung, die das Paper mit „somewhat too low selectivity“ meint.
#
# ### 19.3 Wenn der Score gegen 0 geht
#
# Bei Balkenstärke $1.5$ heben sich Plus und Balken exakt auf, $y = 0$. Kurz davor ist das
# dieselbe Lage wie in Teil I, Abschnitt 5 — nur im CNN: LRP-0 erklärt einen winzigen Score mit
# riesigen, sich fast aufhebenden Relevanzen. Gemessen wird das als **Rauschverhältnis**
# $\sum_j |R_j| \,/\, |\sum_j R_j|$. Es explodiert für LRP-0, während ε und γ es beschränkt halten.
#

# %%
SWEEP_RULES = {
    "LRP-0": {},
    "ε": {"epsilon": 0.1},
    "γ": {"gamma": 0.25},
    "α1β0": {"alpha": 1.0, "beta": 0.0},
    "flat": {"strategy": LRPStrategy(layers=[{"flat": True}] * 3)},
}
SWEEP_ORDER = [name for name in RULE_ORDER if name in SWEEP_RULES]

fleck_levels = np.linspace(0.0, 4.0, 9)
fleck_rows, fleck_scores = [], []
for level in fleck_levels:
    image = make_image(Fleck=float(level))
    fleck_scores.append(float(np.asarray(cnn(image[None, ..., None])).ravel()[0]))
    fleck_rows.append({name: float(lrp_map(cnn, image, **kwargs)[MASKS["Fleck"]].sum())
                       for name, kwargs in SWEEP_RULES.items()})
fleck_table = pd.DataFrame(fleck_rows, index=np.round(fleck_levels, 2))

balken_levels = np.linspace(0.0, 1.4, 8)
balken_rows, balken_scores = [], []
for level in balken_levels:
    image = make_image(Balken=float(level))
    balken_scores.append(float(np.asarray(cnn(image[None, ..., None])).ravel()[0]))
    balken_rows.append({name: float(lrp_map(cnn, image, **kwargs)[MASKS["Balken"]].sum())
                        for name, kwargs in SWEEP_RULES.items()})
balken_table = pd.DataFrame(balken_rows, index=np.round(balken_levels, 2))

fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9))
for name in SWEEP_ORDER:
    axes[0].plot(fleck_levels, fleck_table[name], marker="o", markersize=4,
                 color=RULE_COLORS[name], label=name)
    axes[1].plot(balken_levels, balken_table[name], marker="o", markersize=4,
                 color=RULE_COLORS[name], label=name)
axes[0].axhline(0.0, color="black", linewidth=0.8)
axes[0].set_xlabel("Helligkeit des Flecks")
axes[0].set_ylabel("Relevanz im Fleck")
axes[0].set_title(f"y bleibt konstant bei {fleck_scores[0]:.3f} — nur flat reagiert", fontsize=9)
axes[0].legend(frameon=False, fontsize=8, ncol=2)
axes[1].axhline(0.0, color="black", linewidth=0.8)
axes[1].set_xlabel("Stärke des Balkens")
axes[1].set_ylabel("Relevanz im Balken")
axes[1].set_title("y fällt von +2.44 auf ~0 — α1β0 bleibt trotzdem positiv", fontsize=9)
axes[1].legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout()
plt.show()

display(pd.DataFrame({"Helligkeit Fleck": np.round(fleck_levels, 2), "y": fleck_scores})
        .style.format({"Helligkeit Fleck": "{:.2f}", "y": "{:.6f}"}).hide(axis="index"))

assert np.allclose(fleck_scores, fleck_scores[0], atol=1e-5)
for name in ["LRP-0", "ε", "γ", "α1β0"]:
    np.testing.assert_allclose(fleck_table[name].to_numpy(), 0.0, atol=1e-6,
                               err_msg=f"{name} reagiert auf blosse Helligkeit")
assert fleck_table["flat"].abs().max() > 0.05
assert balken_table["LRP-0"].iloc[-1] < -0.5
assert balken_table["α1β0"].iloc[-1] >= -1e-6


# %%
NOISE_RULES = {"LRP-0": {}, "ε": {"epsilon": 0.1}, "γ": {"gamma": 0.25}}
near_zero_levels = [1.0, 1.2, 1.35, 1.44, 1.47, 1.49]

noise_rows = []
for level in near_zero_levels:
    image = make_image(Balken=float(level))
    score = float(np.asarray(cnn(image[None, ..., None])).ravel()[0])
    row = {"Balkenstärke": level, "y": score}
    for name, kwargs in NOISE_RULES.items():
        relevance = lrp_map(cnn, image, **kwargs)
        row[f"|R| max — {name}"] = float(np.abs(relevance).max())
        row[f"Rauschen — {name}"] = float(np.abs(relevance).sum() / abs(relevance.sum()))
    noise_rows.append(row)
noise_table = pd.DataFrame(noise_rows)
display(noise_table.style.format({"Balkenstärke": "{:.2f}", "y": "{:+.4f}",
                                  **{c: "{:.2f}" for c in noise_table.columns[2:]}})
        .hide(axis="index"))

fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
for name in NOISE_RULES:
    axes[0].plot(noise_table["y"], noise_table[f"|R| max — {name}"], marker="o",
                 color=RULE_COLORS[name], label=name)
    axes[1].plot(noise_table["y"], noise_table[f"Rauschen — {name}"], marker="o",
                 color=RULE_COLORS[name], label=name)
for ax in axes:
    ax.set_xlabel("y (Score, läuft gegen 0)")
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("größte Einzelrelevanz |R| max")
axes[0].set_title("LRP-0 bleibt groß, obwohl y verschwindet", fontsize=9)
axes[1].set_ylabel("Σ|R| / |Σ R|")
axes[1].set_yscale("log")
axes[1].set_title("Rauschverhältnis", fontsize=9)
fig.tight_layout()
plt.show()

assert noise_table["Rauschen — LRP-0"].iloc[-1] > 10 * noise_table["Rauschen — LRP-0"].iloc[0]
assert noise_table["|R| max — LRP-0"].iloc[-1] > 5 * noise_table["|R| max — ε"].iloc[-1]


# %% [markdown]
# <a id="sec-cnn-composite"></a>
# ## 20. Die Komposit-Strategie aus dem Paper
#
# [↑ Inhalt](#top)
#
# Kapitel 1.4.2 („Choosing LRP Rules in Practice“) endet mit einer konkreten Empfehlung, die in der
# Literatur als **LRP-Composite** bzw. **LRP-CMP** läuft:
#
# | Schicht | Regel | Begründung im Paper |
# |---|---|---|
# | oberste fc-Schichten | **LRP-ε** | stabilisiert, dünnt schwache Beiträge aus; Wurzelpunkt nahe am Datenpunkt |
# | oberste conv-Schichten | **LRP-ε** | dito |
# | untere conv-Schichten | **LRP-γ** oder **LRP-αβ** | dort ist die Nichtlinearität stark; besser Gruppen von Pixeln zuweisen und die positive Seite bevorzugen |
# | erste Schicht | **z$^\mathcal{B}$** (Pixel im Kasten) oder **w²** ($\mathbb{R}^d$) | die einzigen Regeln, die keine positiven Eingänge voraussetzen |
# | Max-Pooling | winner-take-all | — |
# | BatchNorm | vorher verschmelzen | — |
# | (nur Auflösung senken) | **flat** | zeigt rezeptive Felder statt Merkmale |
#
# In diesem Repo ist das eine `LRPStrategy`. Die Liste hat **einen Eintrag pro gewichtstragender
# Schicht, von der Eingabe zur Ausgabe** — hier `[conv1, conv2, score]`:
#
# ```python
# LRP(cnn, layer=..., idx=0, strategy=LRPStrategy(
#     layers=[{"alpha": 1, "beta": 0},   # conv1  — untere conv-Schicht
#             {"gamma": 0.25},           # conv2  — mittlere conv-Schicht
#             {"epsilon": 0.1}],         # score  — fc-Kopf
#     pooling=[{"strategy": "winner-takes-all"}] * 2,
# ))
# ```
#
# Für die erste Schicht ergänzt Abschnitt 17 den Rest: Tail-Modell bis `conv1`, dann z$^\mathcal{B}$
# von Hand. Unten stehen beide Varianten nebeneinander — einmal „eine Regel für alles“, einmal das
# Komposit.
#

# %%
composite_strategy = LRPStrategy(
    layers=[{"alpha": 1, "beta": 0}, {"gamma": 0.25}, {"epsilon": 0.1}],
    pooling=[{"strategy": "winner-takes-all"}] * 2,
)
R_composite = lrp_map(cnn, IMAGE, strategy=composite_strategy)

R_composite_tail = relevance_at_conv1_output(
    strategy=LRPStrategy(layers=[{"gamma": 0.25}, {"epsilon": 0.1}],
                         pooling=[{"strategy": "winner-takes-all"}] * 2))
R_composite_zB = first_layer_zB(A_PIXELS, R_composite_tail, *BOX).reshape(GRID, GRID)

comparison_maps = {
    "LRP-0 überall": R_lrp0,
    "ε = 0.1 überall": epsilon_maps[0.1],
    "α1β0 überall": R_a1b0,
    "Komposit": R_composite,
    "Komposit + z^B": R_composite_zB,
}
comparison_frame = pd.DataFrame({label: region_series(values)
                                 for label, values in comparison_maps.items()})
comparison_frame.loc["Summe / y"] = comparison_frame.loc["Summe"] / Y_IMAGE
display(comparison_frame.style.format("{:+.4f}"))
map_row(list(comparison_maps.items()), "Eine Regel für alles gegen die Komposit-Strategie",
        shared_scale=False)

print("Aktive Regeln der Komposit-Strategie:")
explainer = LRP(cnn, layer=len(cnn.layers) - 1, idx=0, strategy=composite_strategy)
for layer in explainer.layers:
    if isinstance(layer, StandardLRPLayer):
        parts = []
        if layer.flat:
            parts.append("flat")
        if layer.alpha is not None:
            parts.append(f"α={layer.alpha:g}, β={layer.beta:g}")
        if layer.gamma:
            parts.append(f"γ={layer.gamma:g}")
        if layer.epsilon:
            parts.append(f"ε={layer.epsilon:g}")
        print(f"  {layer.layer.name}: {', '.join(parts) or 'LRP-0'}")

assert R_composite[MASKS["Plus"]].sum() > 0 > R_composite[MASKS["Balken"]].sum()
np.testing.assert_allclose(float(R_composite_zB.sum()), float(R_composite_tail.sum()), atol=1e-4)


# %% [markdown]
# <a id="sec-cnn-fazit"></a>
# ## 21. Fazit Teil II — welche Regel auf welche Schicht
#
# [↑ Inhalt](#top)
#
# ### Die Entscheidungstabelle
#
# | Schichttyp | Regel | Warum | Im Repo |
# |---|---|---|---|
# | oberste fc-Schichten | **LRP-ε** | stabilisiert, dünnt schwache Beiträge aus | `{"epsilon": …}` |
# | oberste conv-Schichten | **LRP-ε** | dito | `{"epsilon": …}` |
# | untere conv-Schichten | **LRP-γ** | bevorzugt positive Beiträge, DTD-fundiert, Summe bleibt $y$ | `{"gamma": …}` |
# | untere conv-Schichten (ohne Hyperparameter) | **LRP-α₁β₀** | kein freier Parameter, rein positive Karte — dafür kein „dagegen“ | `{"alpha": 1, "beta": 0}` |
# | erste Schicht, Pixel in $[l,h]$ | **z$^\mathcal{B}$** | einzige Regel, die den Wertebereich der Pixel kennt | selbst rechnen (Abschnitt 17) |
# | erste Schicht, Eingabe in $\mathbb{R}^d$ | **w²** | braucht keine Bereichsannahme, ignoriert den Eingabewert | selbst rechnen (Abschnitt 17) |
# | Max-Pooling | winner-take-all | Relevanz folgt dem Maximum | `pooling=[{"strategy": "winner-takes-all"}]` (Default) |
# | Average-Pooling | z-Regel | lineare Schicht mit positiven Gewichten | `{"strategy": "redistribute"}` (Default) |
# | BatchNorm | vorher verschmelzen | LRP soll nur conv/fc, ReLU, Pooling sehen | `fuse_batchnorm`, läuft automatisch |
# | nur Auflösung senken | **flat** | zeigt rezeptive Felder statt Merkmale | `{"flat": True}` |
# | nur einfache Funktionen / oberste Schichten | **LRP-0** | gleiche Behandlung beider Richtungen, aber anfällig für *gradient shattering* | Default |
#
# ### Was die Beispiele in diesem Teil gezeigt haben
#
# | Beobachtung | Abschnitt |
# |---|---|
# | Der hellste Bereich des Bildes bekommt von LRP-0, ε, γ und αβ **exakt 0** — Aktivierung ohne Weg zur Ausgabe erzeugt keine Relevanz. | 12, 19.1 |
# | ε erhält die Form, nimmt aber Masse weg und verdichtet, was bleibt. Erhaltung ist absichtlich verletzt. | 13 |
# | γ verschiebt nur, es schluckt nicht: die Summe bleibt für jedes γ gleich $y$. | 14 |
# | γ → ∞ läuft messbar gegen α₁β₀ — die Aussage aus Kapitel 1.4.1, nachgerechnet. | 14 |
# | α₁β₀ gibt dem Balken **positive** Relevanz, obwohl er den Score nachweislich senkt. | 15, 19.2 |
# | α₂β₁ zeigt beide Richtungen, verliert aber die Erhaltung, sobald eine Schicht keine negative Seite hat. | 15 |
# | `flat` ist die einzige Regel, die Pixeln mit Wert 0 Relevanz gibt — und je höher sie ansetzt, desto gröber die Karte. | 16 |
# | w² verteilt im Fenster allein nach $w^2$; der Pixelwert kommt in der Formel nicht vor. | 17.1 |
# | z$^\mathcal{B}$ mit dem entarteten Kasten $l = h = 0$ **ist** LRP-0; der Kasten ist eine Annahme, kein Regler. | 17, 17.1 |
# | Wenn $y \to 0$ geht, bleibt LRP-0s größte Einzelrelevanz gleich groß — das Rauschverhältnis explodiert. ε und γ halten es beschränkt. | 19.3 |
#
# ### Die Verbindung zu Teil I
#
# Die Fragen sind dieselben geblieben, nur die Schicht ist größer geworden:
#
# * **ε** entscheidet, *wie viel* eines Beitrags stabil genug ist, um in der Erklärung zu bleiben —
#   an zwei Buchungen (Teil I, Abschnitt 5) genauso wie an einem Bild, dessen Score gegen 0 geht
#   (Abschnitt 19.3).
# * **αβ und γ** entscheiden, *welche Richtung* gezeigt wird — am vierstelligen Score (Teil I,
#   Abschnitt 2 und 3) genauso wie am Balken, der das Plus auffrisst (Abschnitt 19.2).
# * **flat** entscheidet gar nicht nach Inhalt — beim toten Gewicht (Teil I, Abschnitt 4) genauso
#   wie beim leeren Hintergrund (Abschnitt 16).
# * **w² und z$^\mathcal{B}$** sind das, was in Teil I noch nicht vorkommen konnte: Regeln, die nicht
#   die Ausgabe einer ReLU erklären, sondern rohe Messwerte.
#
# ### Weiterlesen im Repo
#
# * `notebooks/ipynb_files/check_relevance_conservation_in_LRP.ipynb` — Erhaltung schichtweise nachmessen.
# * `notebooks/ipynb_files/Train_and_explain_3D_mnist_model.ipynb` — dieselben Strategien auf einem
#   echten 3D-CNN mit zehn gewichtstragenden Schichten.
# * `notebooks/ipynb_files/analysis_LRP_for_right_thalamus_volume_based_on_CNN_prediction.ipynb` —
#   dieselbe Frage an einem MRT-Modell, wo die Eingabe z-standardisiert ist und deshalb genau der
#   Fall vorliegt, für den w² und z$^\mathcal{B}$ gemacht sind.
#
# ### Quelle
#
# W. Samek, L. Arras, A. Osman, G. Montavon, K.-R. Müller (2021):
# *Explaining the Decisions of Convolutional and Recurrent Neural Networks*, Buchkapitel —
# Kapitel 1.4.1, 1.4.2 und Tabelle 1.1.
#
