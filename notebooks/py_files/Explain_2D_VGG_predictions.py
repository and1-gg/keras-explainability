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
# # VGG19 erklären: die LRP-Regeln im direkten Vergleich
#
# ## Worum geht es in diesem Notebook?
#
# Wir nehmen ein **fertig trainiertes Bildklassifikationsnetz** (VGG19, trainiert auf ImageNet),
# zeigen ihm ein Katzenfoto und stellen die Frage: **„Welche Pixel haben dich zu dieser
# Entscheidung gebracht?"**
#
# Die Antwort kommt als **Heatmap** — eine Karte, die jedem Eingabepixel eine Zahl zuweist:
# **wie stark hat dieser Pixel für (rot) oder gegen (blau) die erklärte Klasse gesprochen?**
# Das Verfahren dazu heißt **Layer-wise Relevance Propagation (LRP)**.
#
# Das Besondere an diesem Notebook: Es geht **nicht** darum, eine schöne Heatmap zu produzieren.
# Es geht darum zu zeigen, dass es **die eine richtige Heatmap nicht gibt**. LRP ist keine
# einzelne Formel, sondern eine **Familie von Regeln**. Dasselbe Modell, dasselbe Bild, dieselbe
# Klasse — aber fünf verschiedene Regelsätze liefern fünf völlig verschieden aussehende Bilder,
# von unbrauchbarem Rauschen bis zu einer messerscharfen Konturkarte.
#
# Zusätzlich ist das Notebook ein **Validierungs-Test**: Jede eigene Heatmap wird gegen eine
# vorberechnete Referenz aus [**iNNvestigate**](https://github.com/albermax/innvestigate)
# gestellt, der etablierten LRP-Bibliothek für Keras/TensorFlow. Die Referenzen liegen als
# `.npy`-Dateien in `tests/data/`. Sehen beide Bilder gleich aus, ist die Implementierung in
# diesem Repository korrekt.
#
# ## Der größere Kontext: warum überhaupt XAI?
#
# Ein neuronales Netz lernt seine Regeln aus Daten und versteckt sie in Millionen von Gewichten.
# Der Kernsatz der XAI-Community lautet:
#
# > *„Just because a model is right doesn't mean it got there for the right reason."*
#
# Ein Netz könnte eine Katze auch am typischen Sofa im Hintergrund erkennen, an einem
# Wasserzeichen oder an der Bildkomposition. Solche Fälle heißen **Clever-Hans-Prädiktoren**
# (nach dem „rechnenden Pferd", das in Wahrheit die Körpersprache seines Besitzers las).
# Erklärungsverfahren wie LRP machen genau das sichtbar — und sind damit die Voraussetzung
# dafür, ein Modell in einem Bereich einzusetzen, in dem Fehler Konsequenzen haben (Medizin,
# Kreditvergabe, autonomes Fahren).
#
# ## Warum VGG19 und ein Katzenbild?
#
# Das ist das „Hello World" der XAI. Der Aufbau hat drei Vorteile:
#
# 1. **Wir kennen die Antwort.** Auf dem Bild ist eine Katze. Zeigt die Heatmap auf den
#    Hintergrund, ist etwas faul — entweder am Modell oder an unserer Erklärung.
# 2. **1000 Klassen zum Vergleich.** Man kann dasselbe Bild aus Sicht der Klasse „Katze" *und*
#    aus Sicht der Klasse „Hund" erklären lassen (Abschnitt 10).
# 3. **VGG19 ist maximal einfach gebaut.** Nur Faltung, ReLU und Max-Pooling — keine
#    Skip-Connections, keine Batch-Normalisierung. Genau die Architektur, für die die
#    LRP-Regeln ursprünglich formuliert wurden.
#
# ## LRP in drei Sätzen
#
# 1. **Observe** — ein normaler Vorwärtsdurchlauf liefert die Vorhersage und merkt sich, wie
#    stark jedes Neuron aktiviert war.
# 2. **Redistribute** — der Ausgabewert der Zielklasse wird Schicht für Schicht **rückwärts**
#    verteilt, jeweils proportional dazu, wie stark ein Neuron zur Aktivierung des
#    nachfolgenden beigetragen hat.
# 3. **Reveal** — am Ende liegt die gesamte „Relevanz" auf den Eingabepixeln und ergibt die
#    Heatmap.
#
# Die zentrale Eigenschaft ist die **Erhaltung (Conservation)**: LRP erzeugt und vernichtet
# keine Relevanz, es verteilt sie nur um.
#
# $$\sum_j R_j \;=\; \sum_k R_k \qquad \text{für jede Schicht}$$
#
# Wichtig zu verinnerlichen: **Relevanz ist nicht dasselbe wie Aktivierung.** Ein schwach
# aktiviertes Neuron kann sehr relevant sein, wenn genau es den Ausschlag gab.
#
# ## Ablauf dieses Notebooks
#
# ```text
#                          ┌─ Teil A: Regelvergleich am Referenzbild ─────────────┐
#   Katzenfoto ──► [VGG19] ──► Logit "tabby" ──► [LRP rückwärts] ──► Heatmap
#                          │        │                                            │
#                          │        └── 5× mit unterschiedlichen Regeln:          │
#                          │            LRP-0 → LRP-ε → α1β0 → α2β1 → Composite   │
#                          │                    ↕ jeweils Vergleich mit iNNvestigate
#                          └─────────────────────────────────────────────────────┘
#
#                          ┌─ Teil B: Klassenspezifität ──────────────────────────┐
#   4 Fotos (Fisch, Hund, Katze, Vogel) × 4 Zielklassen  ──►  4×4-Heatmap-Matrix
#                          └─────────────────────────────────────────────────────┘
# ```
#
# ---
#
# <a id="toc"></a>
# ## Inhaltsverzeichnis
#
# | # | Abschnitt | Inhalt |
# |---|---|---|
# | 1 | [Setup, Modell und Testbild](#sec-01) | VGG19 laden, Preprocessing verstehen |
# | 2 | [Was sieht VGG19?](#sec-02) | Vorhersage und Klassenindex 281 |
# | 3 | [LRP-0: die nackte z-Regel](#sec-03) | Theorie, erste Heatmap, warum sie rauscht |
# | 4 | [LRP-ε: Rauschen dämpfen](#sec-04) | ε = 0.25 als Stabilisator |
# | 5 | [LRP-αβ mit α=1, β=0](#sec-05) | nur positive Evidenz — und ein Plot-Bug |
# | 6 | [LRP-αβ mit α=2, β=1](#sec-06) | negative Evidenz und das Outlier-Problem |
# | 7 | [Composite-Strategie: das beste Ergebnis](#sec-07) | pro Schicht eine eigene Regel |
# | 8 | [Eigene Bilder laden](#sec-08) | vier Fotos aus dem Netz |
# | 9 | [Vorhersagen für die vier Bilder](#sec-09) | wie sicher ist das Netz? |
# | 10 | [Klassenspezifität: die 4×4-Matrix](#sec-10) | dasselbe Bild, vier Zielklassen |
# | 11 | [Fazit und Fallstricke](#sec-11) | was man mitnehmen sollte |
#
# **Hintergrunddokument im Repo:**
# [`doc/LRP_Fraunhofer_HHI_Video_Zusammenfassung.md`](../../doc/LRP_Fraunhofer_HHI_Video_Zusammenfassung.md)
# — die LRP-Theorie in Worten, inklusive der Rot/Blau-Semantik.

# %% [markdown]
# <a id="sec-01"></a>
# ## 1. Setup, Modell und Testbild
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Vier Dinge in einer Zelle:
#
# 1. **Repository finden.** `find_repo_root()` läuft vom aktuellen Arbeitsverzeichnis nach oben,
#    bis es `pyproject.toml` oder den Ordner `explainability/` sieht, und hängt das Ergebnis an
#    `sys.path`. Erst dadurch ist `from explainability import LRP` importierbar, egal aus welchem
#    Ordner der Kernel gestartet wurde.
# 2. **VGG19 laden.** `VGG19(weights='imagenet')` lädt die Architektur *und* die trainierten
#    Gewichte (beim ersten Aufruf ~550 MB Download nach `~/.keras/models/`). Das Netz hat
#    143,7 Mio. Parameter und 26 Keras-Schichten.
# 3. **Zwei Varianten desselben Katzenbildes laden.** `original_cat.npy` ist das Foto zum
#    Anschauen (`uint8`, Werte 1–233), `preprocessed_cat.npy` ist dasselbe Bild in der Form, die
#    das Netz erwartet (`float64`, Werte −118,16 bis +110,22).
# 4. **Nichts weiter berechnen** — die Zelle erzeugt keine Ausgabe außer den TensorFlow-Meldungen
#    über gefundene GPUs.
#
# ### Warum zwei Bildvarianten? Das Preprocessing
#
# `preprocess_input` von VGG19 arbeitet im **Caffe-Modus** (so wurde das Netz 2014 trainiert)
# und macht zwei Dinge:
#
# 1. Kanalreihenfolge von RGB nach **BGR** drehen,
# 2. den **Trainingsmittelwert** pro Kanal abziehen:
#
# $$X_{\text{BGR}} \;\leftarrow\; X_{\text{BGR}} - \begin{pmatrix} 103{,}939 \\ 116{,}779 \\ 123{,}680 \end{pmatrix}$$
#
# Es wird **nicht** auf $[0,1]$ skaliert. Deshalb liegen die Werte grob zwischen −124 und +152.
# Das ist für LRP wichtig: die Eingabe hat **negative Werte**, und einige Regeln (insbesondere
# αβ) behandeln positive und negative Aktivierungen unterschiedlich.
#
# ### Der Aufbau von VGG19
#
# | Index | Schicht | Ausgabeform | Rolle |
# |---|---|---|---|
# | 0 | `input_layer` | (224, 224, 3) | Eingabe |
# | 1–2 | `block1_conv1/2` | (224, 224, 64) | Kanten, Farbübergänge |
# | 3 | `block1_pool` | (112, 112, 64) | Auflösung halbieren |
# | 4–5 | `block2_conv1/2` | (112, 112, 128) | Texturen |
# | 6 | `block2_pool` | (56, 56, 128) | |
# | 7–10 | `block3_conv1…4` | (56, 56, 256) | Muster, Teilformen |
# | 11 | `block3_pool` | (28, 28, 256) | |
# | 12–15 | `block4_conv1…4` | (28, 28, 512) | Objektteile (Augen, Ohren) |
# | 16 | `block4_pool` | (14, 14, 512) | |
# | 17–20 | `block5_conv1…4` | (14, 14, 512) | ganze Objektkonzepte |
# | 21 | `block5_pool` | (7, 7, 512) | |
# | 22 | `flatten` | (25088,) | 3D-Gitter → Vektor |
# | 23–24 | `fc1`, `fc2` | (4096,) | Klassifikator |
# | 25 | `predictions` | (1000,) | ein Wert je ImageNet-Klasse |
#
# Insgesamt **16 Faltungs- und 3 Dense-Schichten** — daher der Name „VGG-19". Diese 19 Schichten
# sind genau die, für die LRP eine Regel braucht (Abschnitt 7). Die 5 Pooling-Schichten und die
# ReLUs bekommen eigene, einfachere Regeln.
#
# ### Einordnung
#
# `len(model.layers) - 1 = 25` ist also die Schicht `predictions`. Alle LRP-Aufrufe in diesem
# Notebook erklären folglich den **Ausgabe-Score vor der Softmax** (das sogenannte *Logit*) —
# `LRP` entfernt die Softmax-Aktivierung intern selbst. Das ist Absicht: die Softmax ist eine
# Normalisierung über alle 1000 Klassen, ihr Gradient vermischt Klassen und macht Erklärungen
# unschärfer.

# %%
import sys
from pathlib import Path


def find_repo_root() -> Path:
    p = Path.cwd().resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / "explainability").is_dir():
            return candidate
    return p


repo_root = find_repo_root()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from skimage.io import imread
from skimage.transform import resize
from explainability import LRP, LRPStrategy

data = repo_root / "tests" / "data"
model = VGG19(weights='imagenet')

image = np.load(data / "preprocessed_cat.npy")
original_image = np.load(data / "original_cat.npy")

# %% [markdown]
# <a id="sec-02"></a>
# ## 2. Was sieht VGG19 auf dem Bild?
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# `np.expand_dims(image, 0)` macht aus dem Bild der Form `(224, 224, 3)` einen **Batch** der Form
# `(1, 224, 224, 3)` — Keras erwartet immer eine Batch-Dimension. `model.predict` liefert dann
# 1000 Wahrscheinlichkeiten (nach Softmax), und `decode_predictions(..., 5)` übersetzt die fünf
# höchsten in lesbare Namen.
#
# ### Ausgabe
#
# ```text
# [('n02123045', 'tabby',        0.399),
#  ('n02123159', 'tiger_cat',    0.335),
#  ('n02124075', 'Egyptian_cat', 0.154),
#  ('n02127052', 'lynx',         0.006),
#  ('n02971356', 'carton',       0.004)]
# ```
#
# Die Kürzel wie `n02123045` sind **WordNet-IDs**, die eindeutigen Bezeichner der ImageNet-Klassen.
#
# ### Interpretation
#
# **Die Top-3 sind alle Katzen.** Das Netz ist sich absolut sicher, *dass* es eine Katze sieht
# (0,399 + 0,335 + 0,154 = 88,8 %), aber unsicher, *welche Rasse*. Das ist typisch für
# **feingranulare Klassen**: ImageNet enthält allein rund ein Dutzend Katzenklassen, deren
# Unterschiede (Fellmuster, Kopfform) selbst für Menschen schwer sind. Die 0,399 sind also
# **kein Zeichen von Schwäche**, sondern ein korrekt kalibrierter Ausdruck der Zweideutigkeit.
#
# **`carton` (Pappkarton) auf Platz 5** ist der interessanteste Eintrag: Das Netz hat im
# Bildhintergrund etwas gefunden, das es an einen Karton erinnert. Genau solchen
# Hintergrund-Einfluss wollen wir mit LRP sichtbar machen.
#
# ### Warum im Folgenden `idx=281`?
#
# 281 ist der ImageNet-Index von `tabby` — die Top-1-Klasse. LRP braucht immer **eine**
# Zielklasse: die Frage lautet nicht „warum diese Vorhersage", sondern immer „was sprach für
# und gegen **genau diese Klasse**". Man kann jede der 1000 Klassen erklären, auch eine, die das
# Netz gar nicht gewählt hat (das nutzen wir in Abschnitt 10 aus).
#
# ### Einordnung
#
# Dieser Schritt ist der **Observe**-Teil von LRP. Er ist auch die Referenz für alles Weitere:
# eine Heatmap ist nur so viel wert wie die Vorhersage, die sie erklärt. Eine Erklärung für eine
# falsche oder unsichere Vorhersage zu interpretieren, ist einer der häufigsten Anfängerfehler.

# %%
predictions = model.predict(np.expand_dims(image, 0))
print(f'Predictions: {decode_predictions(predictions, 5)}')

# %% [markdown]
# <a id="sec-03"></a>
# ## 3. LRP-0: die nackte z-Regel
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### 3.1 Die Mathematik dahinter
#
# Jedes Neuron $k$ einer Schicht berechnet im Vorwärtspass
#
# $$z_k \;=\; \sum_j a_j \, w_{jk} \;+\; b_k$$
#
# wobei $a_j$ die Aktivierungen der Vorgängerschicht sind. LRP dreht das um: Wenn Neuron $k$
# die Relevanz $R_k$ zugewiesen bekommen hat, gibt es sie an seine Vorgänger $j$ weiter — und
# zwar **proportional zu deren Beitrag** $a_j w_{jk}$:
#
# $$R_j \;=\; \sum_k \frac{a_j\,w_{jk}}{\sum_{j'} a_{j'} w_{j'k}} \; R_k$$
#
# Das ist die **z-Regel**, auch **LRP-0** genannt. Der Bruch summiert sich über $j$ zu 1, deshalb
# gilt die Erhaltung $\sum_j R_j = \sum_k R_k$. Das Vorzeichen entsteht ganz natürlich: Ist
# $a_j w_{jk} < 0$, hat der Pixel der Aktivierung **entgegengewirkt** und bekommt negative
# Relevanz → **blau**.
#
# ### 3.2 Wie die Regel im Code aussieht
#
# Naiv würde man diese Doppelsumme explizit ausrechnen — bei 25088 × 4096 Verbindungen ist das
# unmöglich. Der Trick (Montavon et al.) ist eine Umformung in **vier Schritte**, die alle mit
# den vorhandenen Faltungs-Operationen erledigt werden können. So steht es wörtlich in
# `explainability/layers/layer.py`:
#
# | Schritt | Formel | Code |
# |---|---|---|
# | 1. forward | $z_k = \sum_j a_j w_{jk}$ | `z = self.forward(a, w)` |
# | 2. divide | $s_k = R_k / z_k$ | `s = R / z` |
# | 3. backward | $c_j = \sum_k w_{jk} s_k$ | `c = self.backward(w, s)` |
# | 4. multiply | $R_j = a_j \, c_j$ | `R = tf.multiply(a, c)` |
#
# Schritt 3 ist eine **transponierte Faltung** — genau die Operation, die auch beim normalen
# Backpropagation-Gradienten benutzt wird. LRP ist deshalb rechnerisch etwa so teuer wie ein
# Trainingsschritt.
#
# Zwei Details der Implementierung, die man kennen sollte:
#
# * **Bias.** `z` enthält den Bias *nicht*. Vorher korrigiert der Code mit
#   `R = (R * z) / (z + bias)` die eingehende Relevanz um den Anteil, den der Bias am
#   Pre-Aktivierungswert hat. Dieser Anteil **verschwindet** — Bias-Terme „schlucken" Relevanz,
#   die Erhaltung gilt also nur bis auf die Bias-Beiträge.
# * **Max-Pooling.** Ohne weitere Angabe nutzt `MaxPoolingLRP` die Strategie
#   *winner-takes-all*: die gesamte Relevanz eines 2×2-Fensters geht an das Pixel, das im
#   Vorwärtspass das Maximum geliefert hat. Bei fünf Pooling-Stufen wird die Relevanz also
#   wiederholt auf einzelne Pixel zusammengezogen.
#
# ### 3.3 Was diese Zelle konkret tut
#
# ```python
# lrp = LRP(model, layer=len(model.layers) - 1, idx=281, epsilon=1e-15)
# ```
#
# | Argument | Bedeutung |
# |---|---|
# | `model` | das zu erklärende (trainierte) Netz |
# | `layer=25` | ab welcher Schicht rückwärts erklärt wird → `predictions` |
# | `idx=281` | welches Ausgabeneuron erklärt wird → `tabby` |
# | `epsilon=1e-15` | Regel-Parameter, hier nur als Schutz gegen Division durch 0 |
#
# `LRP` ist selbst ein **Keras-Modell**. Es wird gebaut, indem der Graph des Originalnetzes
# topologisch sortiert, umgedreht und jede Schicht durch ihre LRP-Gegenstück-Schicht ersetzt
# wird. Der Aufruf `lrp(...)` ist dann ein normaler Forward-Pass durch dieses Erklärer-Netz.
#
# Die Initialisierung der Relevanz geschieht per Maske:
#
# $$R_k^{(\text{letzte Schicht})} \;=\; \begin{cases} z_{281} & k = 281 \\ 0 & \text{sonst}\end{cases}$$
#
# Danach drei Nachbearbeitungsschritte:
#
# 1. `explanations` hat die Form `(1, 224, 224, 3)` — ein Relevanzwert **pro Farbkanal**. Die
#    Summe über die Kanäle macht daraus eine 2D-Karte:
#    $$H_{xy} = \sum_{c \in \{B,G,R\}} R_{xyc}$$
# 2. Normalisierung auf den Bereich $[-1, 1]$: $\tilde H = H / \max_{xy} |H_{xy}|$
# 3. Anzeige mit `cmap='seismic'` und `clim=(-1, 1)`. Die Farbskala *seismic* läuft
#    **blau → weiß → rot**, `clim` fixiert sie so, dass **weiß immer genau 0** bedeutet. Ohne
#    dieses `clim` würde matplotlib automatisch skalieren und die Null wäre nicht mehr weiß —
#    die Heatmap wäre unlesbar.
#
# Parallel wird `cat_explanations_none.npy` geladen und identisch behandelt: die
# iNNvestigate-Referenz für dieselbe Regel.
#
# ### 3.4 Was man auf der Abbildung sieht
#
# **Beide Bilder sind praktisch identisch.** Das ist die eigentliche Botschaft dieser Zelle: die
# Implementierung in diesem Repository reproduziert iNNvestigate. Ein solcher Vergleich gegen
# eine Referenzimplementierung ist unverzichtbar, denn ein XAI-Verfahren liefert **immer** ein
# buntes Bild — auch wenn ein Tensor vertauscht oder eine Regel falsch angewandt ist. Das sieht
# man einem Bild nicht an.
#
# **Das Bild ist unbrauchbar rauschig.** Ein dichtes Salz-und-Pfeffer-Muster aus roten und blauen
# Einzelpixeln, ohne erkennbare Struktur. Die Katze ist nur als sehr blasser Umriss zu erahnen.
#
# **Die stärkste Aktivität liegt im linken Bilddrittel** — also im dunklen, texturierten
# Hintergrund (Türrahmen), **nicht** auf der Katze. Wer dieses Bild ohne Vorwissen
# interpretieren müsste, käme zum Schluss „das Netz schaut auf den Hintergrund". Das wäre
# **falsch** — es ist ein Artefakt der Regel, nicht eine Eigenschaft des Modells.
#
# ### 3.5 Warum rauscht LRP-0 so?
#
# Der Nenner $\sum_{j'} a_{j'} w_{j'k}$ kann beliebig **klein** werden, wenn sich positive und
# negative Beiträge fast aufheben. Dann explodiert der Bruch. Bei 19 hintereinandergeschalteten
# Schichten multiplizieren sich diese Instabilitäten auf.
#
# Theoretisch ist LRP-0 zudem nichts Neues: für ReLU-Netze ohne Bias ist es **identisch zu
# $\text{Gradient} \times \text{Input}$**. Und dass reine Gradienten-Saliency verrauscht ist, ist
# seit Jahren bekannt — ein Gradient beschreibt nur eine *infinitesimale* Umgebung des
# Eingabepunkts, nicht die eigentliche Entscheidung.
#
# ### Einordnung
#
# LRP-0 ist der Ausgangspunkt, nicht das Ziel. Die folgenden vier Abschnitte zeigen, wie man von
# hier zu einer brauchbaren Erklärung kommt — durch immer geschicktere Regeln. Wichtig für den
# Kopf: **Alle fünf Bilder sind „korrekt"** in dem Sinne, dass sie ihre jeweilige Regel exakt
# umsetzen. Es gibt keine objektiv wahre Heatmap; es gibt nur Regeln mit unterschiedlichen
# Eigenschaften.

# %%
lrp = LRP(model, layer=len(model.layers) - 1, idx=281, epsilon=1e-15)
explanations = lrp(np.expand_dims(image, 0))
explanations = np.sum(explanations, axis=-1)
explanations = explanations / np.amax(np.abs(explanations))

innvestigate = np.load(data / 'cat_explanations_none.npy')
innvestigate = np.sum(innvestigate, axis=-1)
innvestigate = innvestigate / np.amax(np.abs(innvestigate))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Epsilon=1e-15')

ax[0].imshow(explanations[0], cmap='seismic', clim=(-1, 1))
ax[0].axis('off')
ax[0].set_title('Ours')
ax[1].imshow(innvestigate, cmap='seismic', clim=(-1, 1))
ax[1].axis('off')
ax[1].set_title('Innvestigate')
plt.show()

# %% [markdown]
# <a id="sec-04"></a>
# ## 4. LRP-ε: das Rauschen dämpfen
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Genau derselbe Code wie in Abschnitt 3 — nur `epsilon=0.25` statt `1e-15`. Die Regel lautet
# jetzt:
#
# $$R_j \;=\; \sum_k \frac{a_j\,w_{jk}}{z_k + \epsilon \cdot \mathrm{sign}(z_k)} \; R_k$$
#
# Im Code ist das eine einzige Zeile:
#
# ```python
# epsilon = tf.multiply(self.epsilon, tf.math.sign(z))
# z = tf.add(z, epsilon)
# ```
#
# Das `sign(z)` ist wesentlich: $\epsilon$ wird **immer betragsvergrößernd** addiert, damit der
# Nenner nie kleiner wird und nie das Vorzeichen wechselt.
#
# ### Wie wirkt das?
#
# $\epsilon$ ist ein **Schwellwert für Bedeutsamkeit**. Für Neuronen mit großem $|z_k|$ (starke,
# eindeutige Aktivierung) ist $\epsilon = 0{,}25$ irrelevant. Für Neuronen mit $|z_k| \approx 0$
# — genau die Instabilitätsquelle aus Abschnitt 3.5 — dominiert $\epsilon$ den Nenner und drückt
# den Bruch nach 0.
#
# $$\frac{a_j w_{jk}}{z_k \pm \epsilon} \quad\longrightarrow\quad
# \begin{cases}
# \text{unverändert} & |z_k| \gg \epsilon \\
# \approx 0 & |z_k| \ll \epsilon
# \end{cases}$$
#
# Der Preis: **die Erhaltung wird verletzt.** Ein Teil der Relevanz „versickert" bei jedem
# Schritt. Konkret nachrechenbar an den Referenzdateien: die Gesamtrelevanz sinkt von
# **3,724** (LRP-0) auf **3,689** (ε = 0,25), also knapp 1 % Verlust. Der `LRPLayer` hat für
# solche Fälle das optionale Flag `adjust_epsilon`, das die Summe hinterher wieder
# hochskaliert — hier wird es nicht benutzt.
#
# ### Was man auf der Abbildung sieht
#
# **Etwas ruhiger, aber dieselbe Grundstruktur.** Vergleicht man Seite an Seite mit Abschnitt 3,
# sind die Extrempixel weniger stark, die weißen Flächen etwas größer, insgesamt weniger
# „hartes" Rot und Blau. Die Aussage des Bildes ändert sich jedoch **nicht**: das linke
# Bilddrittel dominiert weiterhin, die Katze bleibt kaum erkennbar.
#
# **Wieder stimmen beide Panels überein.** Zweiter erfolgreicher Validierungstest.
#
# ### Interpretation
#
# ε = 0,25 ist zu wenig, um das Problem zu lösen. Man könnte ε drastisch erhöhen — dann
# verschwindet zwar das Rauschen, aber es verschwindet auch fast alle Relevanz. ε steuert also
# einen **Trade-off zwischen Rauschen und Signalerhalt** und kann ihn nicht auflösen. Die Lösung
# liegt in einer anderen Regelfamilie (Abschnitt 5) bzw. in der Kombination beider
# (Abschnitt 7).
#
# ### Einordnung
#
# In der Praxis ist LRP-ε trotzdem wichtig — allerdings **nicht für das ganze Netz**, sondern
# gezielt für die **oberen, dichten Schichten**. Dort ist der Bedarf an Rauschunterdrückung am
# größten (viele Verbindungen, viel Kompensation), und der Relevanzverlust fällt weniger auf.
# Genau so wird ε in Abschnitt 7 eingesetzt.

# %%
lrp = LRP(model, layer=len(model.layers) - 1, idx=281, epsilon=0.25)
explanations = lrp(np.expand_dims(image, 0))
explanations = np.sum(explanations, axis=-1)
explanations = explanations / np.amax(np.abs(explanations))

innvestigate = np.load(data / 'cat_explanations_eps.npy')
innvestigate = np.sum(innvestigate, axis=-1)
innvestigate = innvestigate / np.amax(np.abs(innvestigate))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('epsilon=0.25')

ax[0].imshow(explanations[0], cmap='seismic', clim=(-1, 1))
ax[0].axis('off')
ax[0].set_title('Ours')
ax[1].imshow(innvestigate, cmap='seismic', clim=(-1, 1))
ax[1].axis('off')
ax[1].set_title('Innvestigate')
plt.show()

# %% [markdown]
# <a id="sec-05"></a>
# ## 5. LRP-αβ mit α=1, β=0: nur positive Evidenz
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Die Idee: positive und negative Beiträge getrennt behandeln
#
# Das Grundproblem von LRP-0/ε ist, dass sich positive und negative Beiträge im **gemeinsamen
# Nenner** gegenseitig auslöschen. Die **αβ-Regel** trennt sie und normiert jede Gruppe für sich:
#
# $$R_j \;=\; \sum_k \left( \alpha \, \frac{(a_j w_{jk})^{+}}{\sum_{j'} (a_{j'} w_{j'k})^{+}}
#                       \;-\; \beta \, \frac{(a_j w_{jk})^{-}}{\sum_{j'} (a_{j'} w_{j'k})^{-}} \right) R_k$$
#
# mit $(\cdot)^+ = \max(\cdot, 0)$ und $(\cdot)^- = \min(\cdot, 0)$. Weil jeder der beiden Nenner
# nur gleichsinnige Terme enthält, kann keiner mehr durch Auslöschung nahe Null geraten — die
# **Instabilität ist strukturell beseitigt**, ohne Regularisierungsparameter.
#
# Damit die Erhaltung erhalten bleibt, muss gelten:
#
# $$\alpha - \beta = 1$$
#
# Das erzwingt der Code als Assertion (`explainability/layers/layer.py`):
#
# ```python
# if alpha is not None:
#     assert alpha == beta + 1, 'beta must be equal to alpha + 1'
# ```
#
# Es gibt also nur **einen** freien Parameter. Gängige Wahlen: (α=1, β=0), (α=2, β=1),
# (α=1,5, β=0,5).
#
# ### Was passiert bei α=1, β=0?
#
# Der zweite Term fällt komplett weg. Es wird **ausschließlich positive Evidenz** verteilt:
#
# $$R_j \;=\; \sum_k \frac{(a_j w_{jk})^{+}}{\sum_{j'} (a_{j'} w_{j'k})^{+}} \; R_k$$
#
# Die Heatmap kann damit **keine negativen Werte** enthalten. Das ist an der Referenzdatei
# nachprüfbar: `cat_explanations_a1b0.npy` hat als Minimum $+3{,}5 \cdot 10^{-6}$, also
# ausschließlich positive Werte. Die Frage, die das Bild beantwortet, ist folglich nicht mehr
# „was sprach für und gegen Tabby", sondern nur noch: **„woraus baut sich die Tabby-Evidenz
# auf?"**
#
# ### Was man auf der Abbildung sieht — linkes Panel („Ours")
#
# **Ein völlig anderes Bild als in Abschnitt 3/4.** Aus dem Rauschteppich ist eine saubere,
# lesbare Karte geworden:
#
# * **Die Augen sind die klarsten Hotspots** — konzentrische rote Ringe um beide Pupillen.
# * **Die Ohrkonturen** treten als scharfe rote Linien hervor, ebenso die **Nase** und die
#   **Schnurrhaar-Wurzeln**.
# * **Das gepunktete Fellmuster** auf Brust und Flanke ist als feine rote Textur erkennbar —
#   plausibel, denn „tabby" ist definitionsgemäß eine **Fellzeichnung**.
# * **Der Hintergrund ist fast vollständig weiß**, das linke Bilddrittel, das LRP-0 dominierte,
#   spielt praktisch keine Rolle mehr.
# * **Alles ist rot**, kein einziger blauer Bereich — wie oben hergeleitet.
#
# Das ist die klassische Lehrbuch-Heatmap aus dem Original-Paper von
# [Bach et al. (2015)](https://doi.org/10.1371/journal.pone.0130140): das Modell schaut auf die
# Katze, und dort auf genau die Merkmale, die ein Mensch auch nennen würde. Das Netz liegt also
# aus den richtigen Gründen richtig.
#
# ### Das rechte Panel ist leer — warum?
#
# Das ist **kein Fehler des Verfahrens, sondern ein Plot-Bug in dieser Zelle**. In allen anderen
# vier Blöcken folgt auf `np.sum(...)` eine Normalisierungszeile:
#
# ```python
# innvestigate = innvestigate / np.amax(np.abs(innvestigate))
# ```
#
# In diesem Block **fehlt genau diese Zeile**. Die Referenzwerte haben ein Maximum von
# $3{,}72 \cdot 10^{-3}$, werden aber mit `clim=(-1, 1)` dargestellt. Alle Werte landen damit im
# Intervall $[0;\,0{,}004]$ der Farbskala — also praktisch exakt auf Weiß. Die Daten sind da, nur
# unsichtbar.
#
# **Lehre daraus:** Absolute Relevanzwerte sind zwischen Regeln nicht vergleichbar. Ohne
# Normalisierung ist eine LRP-Heatmap nicht interpretierbar, und ein leeres Panel bedeutet nicht
# „keine Relevanz", sondern meistens „falsche Skala".
#
# ### Einordnung
#
# αβ mit β=0 ist die freundlichste Regel für Präsentationen: stabil, aufgeräumt, gut lesbar. Sie
# hat aber einen konzeptionellen Preis. Indem sie negative Evidenz komplett verwirft, kann sie
# **nicht mehr zeigen, was gegen eine Klasse spricht** — und damit auch nicht erklären, warum
# eine konkurrierende Klasse verloren hat. Für die Fehleranalyse eines Modells („warum hat es
# *nicht* Klasse X gewählt?") ist sie deshalb ungeeignet. Genau darum geht es im nächsten
# Abschnitt.

# %%
lrp = LRP(model, layer=len(model.layers) - 1, idx=281, alpha=1, beta=0)
explanations = lrp(np.expand_dims(image, 0))
explanations = np.sum(explanations, axis=-1)
explanations = explanations / np.amax(np.abs(explanations))

innvestigate = np.load(data / 'cat_explanations_a1b0.npy')
innvestigate = np.sum(innvestigate, axis=-1)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('alpha=1, beta=0')

ax[0].imshow(explanations[0], cmap='seismic', clim=(-1, 1))
ax[0].axis('off')
ax[0].set_title('Ours')
ax[1].imshow(innvestigate, cmap='seismic', clim=(-1, 1))
ax[1].axis('off')
ax[1].set_title('Innvestigate')
plt.show()

# %% [markdown]
# <a id="sec-06"></a>
# ## 6. LRP-αβ mit α=2, β=1: negative Evidenz zulassen
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Dieselbe αβ-Regel, aber jetzt mit β = 1. Die Nebenbedingung α − β = 1 ist mit α = 2 erfüllt.
# Beide Terme sind aktiv:
#
# $$R_j \;=\; \sum_k \left( 2 \cdot \frac{(a_j w_{jk})^{+}}{\sum (a w)^{+}}
#                       \;-\; 1 \cdot \frac{(a_j w_{jk})^{-}}{\sum (a w)^{-}} \right) R_k$$
#
# Interpretation der Parameter: **α ist die Verstärkung der „Pro"-Evidenz, β die der
# „Contra"-Evidenz.** Bei (2, 1) wird doppelt so viel positive Relevanz verteilt wie bei (1, 0),
# davon aber eine Einheit negativer Relevanz wieder abgezogen. Netto bleibt dieselbe Summe
# (α − β = 1), aber die **Kontraste werden größer**: das Bild wird „kontrastreicher" bzw.
# „aggressiver".
#
# So sieht die Zerlegung im Code für Faltungsschichten aus
# (`explainability/layers/conv.py`) — sie zerlegt sowohl die Aktivierung als auch das Gewicht in
# Positiv- und Negativteil und kombiniert vier Faltungen:
#
# ```python
# zpos = forward(a⁺, w⁺) + forward(a⁻, w⁻)   # alle positiven Produkte
# zneg = forward(a⁺, w⁻) + forward(a⁻, w⁺)   # alle negativen Produkte
# ```
#
# Dass hier **beide** Vorzeichen von $a$ berücksichtigt werden, ist bei VGG19 relevant: nach dem
# Caffe-Preprocessing ist die Eingabe teils negativ (Abschnitt 1).
#
# ### Was man auf der Abbildung sieht — linkes Panel („Ours")
#
# **Ähnlich strukturiert wie α1β0, aber schwächer und mit Blau.** Die Konturen von Ohren, Kopf
# und Nase sind weiterhin die dominanten roten Linien. Neu sind zwei Dinge:
#
# * **Blaue Kerne in den Augen und an den Ohrspitzen.** Genau die Stellen, die bei α1β0 die
#   stärksten roten Hotspots waren, haben jetzt einen negativen Anteil. Das ist kein
#   Widerspruch: dieselbe Bildstelle liefert gleichzeitig starke Pro- und starke
#   Contra-Beiträge, und β = 1 macht letztere sichtbar. Inhaltlich plausibel — die Augen sind
#   *das* Merkmal, an dem sich die konkurrierenden Katzenklassen (`tiger_cat`, `Egyptian_cat`)
#   unterscheiden, sie drücken also gleichzeitig von `tabby` weg.
# * **Insgesamt blasser.** Die Fellzeichnung ist nur noch angedeutet.
#
# ### Warum ist das rechte Panel wieder fast leer?
#
# Diesmal **ist** die Normalisierungszeile vorhanden — die Erklärung ist eine andere und
# lehrreichere: **drei Ausreißer-Pixel**.
#
# Nachgerechnet an `cat_explanations_a2b1.npy` (nach Kanalsumme):
#
# | Größe | Wert |
# |---|---|
# | Maximum | +10,64 |
# | Minimum | −0,84 |
# | 99,9-Perzentil nach Normalisierung | 0,0094 |
# | 99-Perzentil nach Normalisierung | 0,0031 |
# | Median nach Normalisierung | 0,00004 |
# | Pixel mit \|Wert\| > 0,1 nach Normalisierung | **3 von 50 176** |
#
# Die Normalisierung $\tilde H = H / \max|H|$ teilt also durch einen Wert, der von drei
# Einzelpixeln bestimmt wird und **12-mal größer** ist als der stärkste negative Wert. Alle
# echten Bildstrukturen werden dadurch auf unter 1 % der Farbskala gestaucht.
#
# ### Einordnung — ein sehr praktischer Fallstrick
#
# Dieses Panel ist das beste Beispiel für einen Fehler, den fast jeder einmal macht:
# **Max-Normalisierung ist gegenüber Ausreißern nicht robust.** In der Praxis normalisiert man
# deshalb auf ein **Perzentil** statt auf das Maximum, z. B.
#
# $$\tilde H \;=\; \mathrm{clip}\!\left(\frac{H}{q_{99{,}5}(|H|)},\, -1,\, 1\right)$$
#
# Warum das linke Panel dieses Problem nicht hat: die Implementierung in diesem Repository
# erzeugt an dieser Stelle keine so extremen Einzelwerte. Die naheliegende Ursache ist eine
# unterschiedliche Behandlung der **Bias-Terme in den αβ-Nennern** — nachgewiesen ist das hier
# nicht, aber ein Blick auf die Gesamtsummen der Referenzdateien zeigt, dass die Regeln
# insgesamt sehr unterschiedliche Relevanzmengen durchlassen:
#
# | Regel | Summe aller Relevanzen |
# |---|---|
# | LRP-0 | 3,72 |
# | LRP-ε (0,25) | 3,69 |
# | α=1, β=0 | 8,62 |
# | α=2, β=1 | 107,90 |
# | Composite (Abschnitt 7) | 4,93 |
#
# Exakte Erhaltung gilt nur für die reine z-Regel ohne Bias. Sobald Bias-Terme im Spiel sind —
# und VGG19 hat in jeder Schicht welche —, wird Relevanz geschluckt oder verstärkt. Für die
# **visuelle** Interpretation ist das folgenlos, weil ohnehin normalisiert wird. Wer aber
# Relevanzen **summiert oder vergleicht** (etwa pro Bildregion oder pro Hirnregion, wie im
# Notebook `Explain_brain_age_predictions`), muss diese Verletzung kennen.

# %%
lrp = LRP(model, layer=len(model.layers) - 1, idx=281, alpha=2, beta=1)
explanations = lrp(np.expand_dims(image, 0))
explanations = np.sum(explanations, axis=-1)
explanations = explanations / np.amax(np.abs(explanations))

innvestigate = np.load(data / 'cat_explanations_a2b1.npy')
innvestigate = np.sum(innvestigate, axis=-1)
innvestigate = innvestigate / np.amax(np.abs(innvestigate))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('alpha=2, beta=1')

ax[0].imshow(explanations[0], cmap='seismic', clim=(-1, 1))
ax[0].axis('off')
ax[0].set_title('Ours')
ax[1].imshow(innvestigate, cmap='seismic', clim=(-1, 1))
ax[1].axis('off')
ax[1].set_title('Innvestigate')
plt.show()

# %% [markdown]
# <a id="sec-07"></a>
# ## 7. Composite-Strategie: für jede Schicht die passende Regel
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Die zentrale Einsicht
#
# Bisher haben wir **eine** Regel auf **alle** 19 Schichten angewandt. Das ist aber gar nicht
# sinnvoll, denn die Schichten haben völlig unterschiedliche Aufgaben:
#
# | Schichtbereich | Was dort passiert | Was die Erklärung braucht |
# |---|---|---|
# | **oben** (`fc1`, `fc2`, `predictions`) | abstrakte Klassenentscheidung, sehr viele Verbindungen, viel gegenseitige Kompensation | **Rauschunterdrückung** → LRP-ε |
# | **Mitte** (Faltungsblöcke) | Objektteile, räumliche Muster | **Stabilität und Struktur** → LRP-αβ |
# | **unten** (erste Faltung) | einzelne Pixel und Farbkanäle; hier entscheidet sich die räumliche Schärfe der Heatmap | **räumliche Glättung**, unabhängig von Pixelwerten → flat-Regel |
#
# Diese Kombination heißt **Composite-LRP** (auch *LRP-CMP*) und ist heute die empfohlene
# Standardanwendung — siehe [Montavon et al.
# (2019)](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10) und Kohlbrenner et al.
# (2020). Sie ist der Grund, warum die Abbildung dieses Abschnitts „Best" heißt.
#
# ### Die flat-Regel
#
# Neu ist hier die **flat-Regel** für die erste Faltungsschicht:
#
# $$R_j \;=\; \sum_k \frac{1}{\#\{j'\}} \; R_k$$
#
# Die Relevanz wird **völlig gleichmäßig** auf alle Eingänge des Filterfensters verteilt, ohne
# jeden Bezug auf Aktivierungen oder Gewichte. Im Code ist das denkbar einfach
# (`explainability/layers/layer.py`):
#
# ```python
# if self.flat:
#     a = tf.ones_like(a)
#     w = tf.ones_like(w)
# ```
#
# Wozu das gut ist: In der ersten Schicht wären die Relevanzen sonst direkt proportional zu den
# Pixelintensitäten — helle Pixel bekämen automatisch mehr Relevanz als dunkle, was inhaltlich
# nichts bedeutet. Die flat-Regel wirkt wie ein **lokaler Tiefpassfilter** und liefert genau die
# räumlich zusammenhängenden Flächen, die eine Heatmap lesbar machen (statt Einzelpixel-Rauschen).
#
# ### Wie die Strategie gelesen wird
#
# `LRPStrategy(layers=[...])` erwartet **genau einen Eintrag pro gewichtstragender Schicht**, hier
# also 19 (16 Faltungen + 3 Dense). Der Code prüft das hart:
#
# ```python
# assert len(strategy.layers) == len(standard_lrp_layers)
# ```
#
# Die Liste ist **von der Eingabe zur Ausgabe** geordnet (intern wird sie mit `[::-1]` gedreht,
# weil das Erklärer-Netz rückwärts aufgebaut wird). Damit ergibt sich diese Zuordnung:
#
# | Listeneinträge | Regel | Schichten |
# |---|---|---|
# | 1 | `flat` | `block1_conv1` |
# | 2 – 15 | α = 1,5 / β = 0,5 | `block1_conv2` … `block5_conv3` (14 Faltungen) |
# | 16 – 19 | ε = 0,25 | `block5_conv4`, `fc1`, `fc2`, `predictions` |
#
# Die 5 Max-Pooling-Schichten sind nicht Teil dieser Liste; sie behalten die Voreinstellung
# *winner-takes-all*. Über `LRPStrategy(pooling=[...])` könnte man auch das umstellen
# (`redistribute` oder `flat`).
#
# α = 1,5 / β = 0,5 ist ein Kompromiss zwischen den Abschnitten 5 und 6: negative Evidenz bleibt
# sichtbar, ist aber nur halb gewichtet.
#
# ### ⚠️ Eine Eigenheit dieser Zelle
#
# ```python
# explanations = lrp(np.expand_dims(image, 0) + np.amin(image))
# ```
#
# Hier wird zum Bild **das eigene Minimum addiert**, also $-118{,}16$ — jeder Pixel wird um diesen
# Betrag verschoben, der Wertebereich wandert von $[-118;\,110]$ nach $[-236;\,-8]$. Das ist eine
# **andere Eingabe** als in allen vorigen Abschnitten und weit außerhalb dessen, was das Netz je
# gesehen hat. Die Zeile diente offenbar dazu, exakt die Vorverarbeitung der
# iNNvestigate-Referenzdatei nachzubauen.
#
# Für die Interpretation heißt das: die gezeigte Heatmap erklärt streng genommen ein **stark
# abgedunkeltes** Bild, nicht das links daneben angezeigte Original. Dass das Ergebnis trotzdem
# überzeugend aussieht, liegt unter anderem an der flat-Regel in der ersten Schicht, die von den
# absoluten Pixelwerten unabhängig ist. Wer diese Strategie übernimmt, sollte den Offset
# weglassen — in Abschnitt 10 fehlt er auch.
#
# ### Was man auf der Abbildung sieht
#
# Drei Panels: **Original**, **Ours**, **Innvestigate**.
#
# **Links das Original.** Eine getigerte Katze frontal, vor einem dunklen Türrahmen mit
# rötlicher Fläche oben.
#
# **Mitte und rechts sind nahezu identisch** — der überzeugendste der fünf Validierungstests,
# weil hier die komplizierteste Regelkombination geprüft wird.
#
# **Die Heatmap ist scharf, zusammenhängend und anatomisch sinnvoll:**
#
# * Die **Silhouette der Katze** ist als durchgehende rote Linie nachgezeichnet — Ohren, Kopf,
#   Wangen, Brustkontur.
# * Die **Augen** sind wieder klare rote Ringe.
# * Die **Fellzeichnung** erscheint als feines rotes Punktmuster über Brust und Flanke — genau
#   das Merkmal, das „tabby" definiert.
# * Der **Hintergrund ist überwiegend weiß**, also irrelevant.
# * Am **linken und rechten Bildrand** liegt ein diffuser **blauer Saum**. Das ist ein
#   **Padding-Artefakt**: an den Bildrändern faltet das Netz über künstlich mit Nullen aufgefüllte
#   Bereiche, und die flat-Regel verteilt dort Relevanz auf Positionen, die kein echtes Signal
#   tragen. Solche Randstreifen sind ein bekanntes LRP-Artefakt und **kein** Hinweis darauf, dass
#   das Modell auf den Bildrand schaut.
#
# Der Vergleich mit Abschnitt 3 ist die Kernaussage des ganzen Notebooks: **gleiches Modell,
# gleiches Bild, gleiche Klasse — unbrauchbares Rauschen gegen eine publikationsreife
# Konturkarte.** Der Unterschied liegt ausschließlich in der Regelwahl.
#
# ### Einordnung
#
# Damit wird auch klar, wo die Grenze der Methode liegt: Man kann eine Heatmap durch Regelwahl
# in erheblichem Maße gestalten. Das macht LRP nicht wertlos, verlangt aber Disziplin:
#
# 1. **Regeln vor dem Ansehen der Ergebnisse festlegen**, nicht danach die schönste auswählen.
# 2. **Die verwendete Strategie immer mitpublizieren** — eine Heatmap ohne Regelangabe ist nicht
#    reproduzierbar.
# 3. Nicht auf das Auge verlassen, sondern **quantitativ prüfen** (Pixel-Flipping,
#    Erhaltungssummen, Klassenspezifität wie in Abschnitt 10).

# %%
alpha=1.5
beta=0.5

strategy = LRPStrategy(
    layers=[
        {'flat': True},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'epsilon': 0.25},
        {'epsilon': 0.25},
        {'epsilon': 0.25},
        {'epsilon': 0.25},
    ]
)

lrp = LRP(model, layer=len(model.layers) - 1, idx=281, strategy=strategy)
explanations = lrp(np.expand_dims(image, 0) + np.amin(image))
explanations = np.sum(explanations, axis=-1)
explanations = explanations / np.amax(np.abs(explanations))

innvestigate = np.load(data / 'cat_explanations_best.npy')
innvestigate = np.sum(innvestigate, axis=-1)
innvestigate = innvestigate / np.amax(np.abs(innvestigate))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Best')

ax[0].imshow(original_image)
ax[0].axis('off')
ax[0].set_title('Original image')
ax[1].imshow(explanations[0], cmap='seismic', clim=(-1, 1))
ax[1].axis('off')
ax[1].set_title('Ours')
ax[2].imshow(innvestigate, cmap='seismic', clim=(-1, 1))
ax[2].axis('off')
ax[2].set_title('Innvestigate')
plt.show()

# %% [markdown]
# <a id="sec-08"></a>
# ## 8. Teil B — vier eigene Bilder laden
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Teil A war ein Methodenvergleich an einem eingefrorenen Referenzbild. Teil B stellt eine ganz
# andere Frage: **Ist die Erklärung überhaupt klassenspezifisch?** Dafür brauchen wir mehrere
# Bilder mit klar unterschiedlichen Klassen.
#
# Die Zelle lädt vier Fotos per HTTP (Goldfisch, Border Collie, Katze, Kolibri), speichert sie
# nach `/tmp/` und bringt sie in Modellform:
#
# | Schritt | Code | Zweck |
# |---|---|---|
# | Herunterladen | `requests.get(urls[key])` | JPEG nach `/tmp/{key}.jpg` |
# | Einlesen | `imread(...)` | → NumPy-Array, beliebige Größe |
# | Skalieren | `resize(img, (224, 224), preserve_range=True)` | auf die Eingabegröße von VGG19 |
# | Zurückcasten | `.astype(np.uint8)` | `resize` liefert float; `preprocess_input` erwartet 0–255 |
# | Anzeigen | `plt.imshow(img)` | Kontrollblick, in `original_images` gemerkt |
# | Vorverarbeiten | `preprocess_input(img)` | BGR + Mittelwertabzug, nach `images` |
#
# Wichtig ist die **Doppelbuchhaltung**: `original_images[key]` ist die Version zum Anschauen,
# `images[key]` die Version fürs Netz. Die Reihenfolge ist entscheidend — `preprocess_input`
# **erst nach** dem Merken des Originals.
#
# Zwei Hinweise zu `preserve_range=True`: ohne dieses Flag würde `skimage.resize` die Werte
# automatisch nach $[0,1]$ skalieren, und der anschließende Cast nach `uint8` würde alles auf 0
# runden — ein komplett schwarzes Bild. Und `resize` auf ein Quadrat **verzerrt das
# Seitenverhältnis**; beim Fischbild (ursprünglich breiter als hoch) ist das sichtbar.
#
# ### Was man auf den vier Abbildungen sieht
#
# | Bild | Inhalt | Besonderheit für die Erklärung |
# |---|---|---|
# | `cat` | cremefarbenes Kätzchen, blaue Augen, vor dunkelrotem Hintergrund, auf Leopardenfell-Decke | **nicht** getigert — anders als die Referenzkatze aus Teil A |
# | `dog` | Border Collie im Profil, schwarz-weiß, Wiese/Gras-Hintergrund | ganzkörperlich, viel Hintergrund |
# | `bird` | Kolibri im Flug an einer roten Blütenrispe | zwei Objekte im Bild: Vogel **und** Blüten |
# | `fish` | zwei Goldfische auf hellem, fast weißem Grund | quasi kein Hintergrund |
#
# ### Einordnung
#
# Die Bildauswahl ist bewusst so getroffen, dass sie **vier weit auseinanderliegende
# ImageNet-Klassen** trifft. Bei zwei Hunderassen wären die Erklärungen kaum unterscheidbar.
#
# Praktischer Hinweis: Diese Zelle braucht eine **Internetverbindung**, und die URLs zeigen auf
# fremde Server. Läuft das Notebook offline oder ist ein Link tot, scheitert der Rest von Teil B.
# Für reproduzierbare Arbeit sollte man Testbilder ins Repository legen — so wie es Teil A mit
# `tests/data/*.npy` vormacht.

# %%
import requests
import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread
from skimage.transform import resize


urls = {
    'cat': 'https://cdn.pixabay.com/photo/2014/11/30/14/11/cat-551554__340.jpg',
    'dog': ('https://static.wikia.nocookie.net/naturerules1/images/f/f9/Border-collie-1.jpg/'
            'revision/latest?cb=20210403210149'),
    'bird': 'https://static.independent.co.uk/2021/04/29/22/newFile-3.jpg?quality=75&width=1200&auto=webp',
    'fish': 'https://m.media-amazon.com/images/I/61QN8NWuNlL._AC_SX679_.jpg'
}

images = {}
original_images = {}

for key in urls:
    req = requests.get(urls[key])

    with open(f'/tmp/{key}.jpg', 'wb') as f:
        f.write(req.content)

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(key)
    img = imread(f'/tmp/{key}.jpg')
    img = resize(img, (224, 224), preserve_range=True)
    img = img.astype(np.uint8)
    original_images[key] = img
    plt.imshow(img)
    plt.gca().axis('off')
    plt.show()
    
    img = preprocess_input(img)
    images[key] = img

# %% [markdown]
# <a id="sec-09"></a>
# ## 9. Vorhersagen für die vier Bilder
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Für jedes der vier Bilder ein Vorwärtsdurchlauf und die Top-10-Klassen. Das ist der
# Pflicht-Sanity-Check vor jeder Erklärung: **erklärt man eine Vorhersage, die das Modell
# tatsächlich getroffen hat?**
#
# ### Ausgabe (Top-1 je Bild)
#
# | Bild | Top-1-Klasse | Wahrscheinlichkeit | Top-2 |
# |---|---|---|---|
# | `cat` | `Egyptian_cat` | **0,570** | `tabby` (0,227) |
# | `dog` | `Border_collie` | **0,863** | `collie` (0,120) |
# | `bird` | `hummingbird` | **0,9999994** | `bulbul` ($3{,}8 \cdot 10^{-7}$) |
# | `fish` | `goldfish` | **1,0** | `rock_beauty` ($3{,}6 \cdot 10^{-9}$) |
#
# ### Interpretation
#
# **Alle vier sind korrekt klassifiziert** — die Vorbedingung für Teil B ist erfüllt.
#
# **Die Sicherheit unterscheidet sich um neun Größenordnungen.** Fisch und Vogel sind praktisch
# gesättigt (Softmax = 1,0), Katze und Hund liegen deutlich darunter. Der Grund ist wieder die
# **Feingranularität** der Klassen: `Border_collie` konkurriert mit `collie` und
# `Shetland_sheepdog`, `Egyptian_cat` mit `tabby` und `tiger_cat`. Beim Goldfisch gibt es
# schlicht keine ähnliche ImageNet-Klasse.
#
# Für die Heatmaps hat das eine wichtige Konsequenz: Die von LRP verteilte Gesamtrelevanz ist der
# **Logit** der Zielklasse. Bei Softmax 1,0 ist dieser Logit riesig, bei einer nicht gewählten
# Klasse kann er sogar negativ sein. **Absolute Relevanzwerte sind zwischen Bildern und Klassen
# nicht vergleichbar** — merken für Abschnitt 10.
#
# ### ⚠️ Ein Detail, das man leicht übersieht
#
# Das heruntergeladene Kätzchen wird als `Egyptian_cat` (ImageNet-Index **285**) erkannt. Im
# nächsten Abschnitt wird es aber mit `idx=281` (`tabby`) erklärt — also mit der **zweitplatzierten**
# Klasse (0,227). Das ist nicht falsch, aber man muss die Heatmap entsprechend lesen: sie zeigt
# nicht „warum das Netz sich entschieden hat", sondern **„was für die Klasse Tabby gesprochen
# hätte"**. Für den Zweck dieses Abschnitts — Klassenspezifität zeigen — ist das ausreichend.
#
# ### Einordnung
#
# Die Softmax-Werte sind hier auch ein Lehrstück über **Kalibrierung**. Eine Wahrscheinlichkeit
# von 1,0 heißt nicht „das Modell hat recht", sondern nur „das Modell ist sich sicher". Moderne
# Netze sind notorisch **überkonfident**, gerade außerhalb der Trainingsverteilung. Das ist ein
# weiteres Argument für XAI: Man braucht eine zweite, unabhängige Informationsquelle über die
# Entscheidung — und genau die liefert die Heatmap.

# %%
for key in images:
    prediction = model.predict(np.expand_dims(images[key], axis=0))
    print(f'Actual class: {key}')
    print('Predictions:')
    print(decode_predictions(prediction, 10))

# %% [markdown]
# <a id="sec-10"></a>
# ## 10. Klassenspezifität: die 4×4-Matrix
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Die Frage dahinter
#
# Der wichtigste Test für ein Erklärungsverfahren lautet: **Ändert sich die Erklärung, wenn ich
# nach einer anderen Klasse frage?** Eine Methode, die für jedes Ausgabeneuron dieselbe Karte
# malt, erklärt gar nichts — sie zeigt nur, wo im Bild „etwas ist" (eine Kantendetektion). Diese
# Prüfung ist der Kern der einflussreichen Arbeit *„Sanity Checks for Saliency Maps"* (Adebayo et
# al., 2018), an der mehrere populäre Verfahren gescheitert sind.
#
# ### Was passiert hier?
#
# Es werden **vier Erklärer** gebaut, einer pro Zielklasse, alle mit der Composite-Strategie aus
# Abschnitt 7:
#
# | Schlüssel | `idx` | ImageNet-Klasse |
# |---|---|---|
# | `fish` | 1 | `goldfish` |
# | `dog` | 232 | `Border_collie` |
# | `cat` | 281 | `tabby` |
# | `bird` | 94 | `hummingbird` |
#
# `layer=25` ist dabei identisch mit `len(model.layers) - 1` aus Teil A.
#
# Dann die Doppelschleife: **jeder Erklärer wird auf jedes Bild angewandt**, also 16 Heatmaps.
#
# ```python
# explanations[i][j] = explainers[keys[i]](np.expand_dims(images[keys[j]], axis=0))
# #                                ↑ Zielklasse            ↑ Eingabebild
# ```
#
# Das ergibt eine Matrix mit
#
# * **Zeilen $i$** = erklärte **Klasse**,
# * **Spalten $j$** = **Eingabebild**.
#
# Auf der **Diagonale** ($i = j$) passen Klasse und Bild zusammen. Alle anderen Felder sind
# **kontrafaktisch**: „Was in diesem Hundebild hätte für einen Goldfisch gesprochen?"
#
# ### Wie die Abbildung aufgebaut ist
#
# 5 Zeilen × 4 Spalten. Zeile 0 zeigt die Originalfotos, Zeilen 1–4 die Heatmaps.
#
# | Zeile | Inhalt |
# |---|---|
# | 0 | Originalbilder: Fisch, Hund, Katze, Vogel |
# | 1 | Relevanz für **`goldfish`** (idx 1) |
# | 2 | Relevanz für **`Border_collie`** (idx 232) |
# | 3 | Relevanz für **`tabby`** (idx 281) |
# | 4 | Relevanz für **`hummingbird`** (idx 94) |
#
# Die Spaltenreihenfolge ist in allen Zeilen dieselbe: **Fisch, Hund, Katze, Vogel**.
#
# Diese Tabelle ist nötig, weil die Zeilenbeschriftungen in der Abbildung **fehlen**:
# `ax[i+1][j].set_ylabel(keys[i])` wird nach `ax[i+1][j].axis('off')` aufgerufen, und mit
# abgeschalteter Achse zeichnet matplotlib kein Label. Ohne die Zuordnung oben ist die Abbildung
# schwer zu lesen.
#
# ### Was man auf der Abbildung sieht
#
# **Die Diagonale ist durchgehend rot und klar strukturiert:**
#
# * `goldfish` auf dem Fischbild: beide Fischkonturen rot nachgezeichnet, kräftige rote Punkte auf
#   den Augen.
# * `Border_collie` auf dem Hundebild: rote Konzentration auf **Kopf, Ohren und Gesicht** — genau
#   die rassetypischen Merkmale.
# * `tabby` auf dem Katzenbild: rote Silhouette, rote Augen, rote Ohren.
# * `hummingbird` auf dem Vogelbild: rot auf Vogelkörper **und** auf der Blütenrispe.
#
# **Die Gegendiagonale zeigt sauberes Blau:**
#
# * Zeile 1 / Spalte Hund: Kopf und Ohren des Collies sind **kräftig blau** — Hundemerkmale
#   sprechen aktiv **gegen** „Goldfisch".
# * Zeile 2 / Spalte Fisch: spiegelbildlich, die Fischkonturen und -augen werden blau.
#
# Dieses **antisymmetrische Paar** ist das eigentliche Ergebnis der Zelle: Dasselbe Pixel bekommt
# je nach gefragter Klasse das entgegengesetzte Vorzeichen. LRP ist also **klassenspezifisch** und
# besteht den Sanity-Check.
#
# **Interessante Zwischenfälle abseits der Diagonale:**
#
# * Zeile 2 / Spalte Katze: der Collie-Erklärer markiert **Ohren und Körperkontur der Katze rot**,
#   die **Augenregion aber blau**. Fell und spitze Ohren sind gemeinsame Merkmale von Katze und
#   Hund; die Augen sind das Unterscheidungsmerkmal. Genau dieselbe Beobachtung beschreibt das
#   Fraunhofer-HHI-Video im
#   [Repo-Dokument](../../doc/LRP_Fraunhofer_HHI_Video_Zusammenfassung.md).
# * Zeile 3 / Spalte Hund: der Tabby-Erklärer markiert den **Rumpf des Hundes stark rot**, den
#   **Kopf aber blau** — der behaarte Körper „könnte" eine Katze sein, der Hundekopf spricht
#   dagegen.
# * Zeile 4 / Spalte Fisch: der Kolibri-Erklärer markiert die Goldfische **kräftig rot**. Ein
#   stromlinienförmiger Körper mit Flossen/Flügeln und warmen Farben ist für das Netz
#   offensichtlich ein gemeinsames Merkmal. Solche **Cross-Talks** sind ein realistisches Bild
#   dessen, wie ein CNN Klassen tatsächlich organisiert — nicht in getrennten Schubladen, sondern
#   in geteilten Merkmalsräumen.
# * Zeile 1–4 / Spalte Vogel: in allen vier Zeilen ist die **rote Blütenrispe** mitmarkiert. Sie
#   ist das kontrastreichste Objekt im Bild und zieht in jeder Klasse Relevanz an. Ein
#   Warnsignal-Kandidat: läge die Rispe *nur* in der Kolibri-Zeile rot, wäre das ein klarer
#   Clever-Hans-Verdacht („Vogel = rote Blüten").
#
# ### ⚠️ Was man aus dieser Abbildung **nicht** ablesen darf
#
# Jedes Panel wird **einzeln** normalisiert:
#
# ```python
# explanation = explanation / np.amax(np.abs(explanation))
# ```
#
# Der Kontrast ist deshalb **zwischen den Panels nicht vergleichbar**. Ein Feld kann kräftig rot
# aussehen, obwohl die absolute Relevanz um Größenordnungen kleiner ist als im Nachbarfeld — bei
# Softmax-Werten von 1,0 bis $10^{-13}$ (Abschnitt 9) ist genau das der Fall. Wer Panels
# vergleichen will, muss eine **gemeinsame Skala** über alle 16 Karten verwenden.
#
# Zweitens: die Karten für nicht vorhergesagte Klassen erklären einen **kleinen oder negativen
# Logit**. Sie beantworten „was spräche hypothetisch für diese Klasse", nicht „so hat das Modell
# entschieden".
#
# ### Einordnung
#
# Diese Matrix ist das Werkzeug, mit dem man in der Praxis prüft, ob eine XAI-Pipeline
# funktioniert. Der Ablauf ist immer derselbe:
#
# 1. **Diagonale plausibel?** Zeigt die Erklärung auf das Objekt der jeweiligen Klasse?
# 2. **Off-Diagonale unterschiedlich?** Wenn alle 16 Karten gleich aussehen, ist die Erklärung
#    nicht klassenspezifisch und damit wertlos.
# 3. **Vorzeichen sinnvoll?** Sprechen fremde Objekte gegen die Klasse (blau)?
# 4. **Verdächtige Hotspots?** Hintergrund, Bildränder, Wasserzeichen, Text — der
#    Clever-Hans-Test.
#
# Erst wenn diese vier Punkte sitzen, lohnt sich der Sprung in eine Domäne, in der man die
# richtige Antwort **nicht** kennt — etwa medizinische Bildgebung. Genau dort setzen die anderen
# Notebooks dieses Repositories an.

# %%
from explainability import LRP

idx = [
    ('fish', 1),
    ('dog', 232),
    ('cat', 281),
    ('bird', 94)
]

explainers = {
    p[0]: LRP(model, layer=25, idx=p[1], strategy=strategy) \
    for p in idx
}

fig, ax = plt.subplots(5, 4, figsize=(15, 15))

explanations = np.zeros((4, 4), dtype=object)

keys = [p[0] for p in idx]

ax[0][0].axis('off')

for i in range(4):
    ax[0][i].imshow(original_images[keys[i]])
    ax[0][i].axis('off')

for i in range(len(keys)):
    for j in range(len(keys)):
        explanations[i][j] = explainers[keys[i]](np.expand_dims(images[keys[j]], axis=0))

for i in range(len(explanations)):
    for j in range(len(explanations[i])):
        explanation = np.sum(explanations[i][j][0], axis=-1)
        explanation = explanation / np.amax(np.abs(explanation))
        ax[i+1][j].imshow(explanation, cmap='seismic', clim=(-1, 1))
        ax[i+1][j].axis('off')
        ax[i+1][j].set_ylabel(keys[i])
        
        
plt.show()

# %% [markdown]
# <a id="sec-11"></a>
# ## 11. Fazit und Fallstricke
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Die Regeln im Überblick
#
# | Regel | Formel (Kern) | Charakter der Heatmap | Wofür geeignet |
# |---|---|---|---|
# | **LRP-0** (z-Regel) | $\dfrac{a_j w_{jk}}{\sum a w}$ | extrem rauschig, Hintergrund dominiert | Referenz/Theorie; entspricht Gradient × Input |
# | **LRP-ε** | Nenner $+\,\epsilon\,\mathrm{sign}(z)$ | etwas ruhiger, gleiche Struktur | obere/dichte Schichten |
# | **LRP-αβ**, α=1 β=0 | nur $(aw)^{+}$ | sauber, nur positiv, konturbetont | Präsentation; „woraus baut sich die Evidenz auf?" |
# | **LRP-αβ**, α=2 β=1 | $\alpha (aw)^{+} - \beta (aw)^{-}$ | kontrastreich, mit Blau, ausreißeranfällig | Contra-Evidenz sichtbar machen |
# | **flat** | $1 / \#\{j'\}$ | räumlich glatt, wertunabhängig | erste Faltungsschicht |
# | **Composite** | pro Schicht die passende | scharf, fokussiert, lesbar | **Standardempfehlung** |
#
# Nebenbedingung für αβ: $\alpha - \beta = 1$ (erzwingt die Relevanzerhaltung).
#
# ### Die sieben Punkte, die man mitnehmen sollte
#
# 1. **Es gibt nicht „die" LRP-Heatmap.** Abschnitte 3 und 7 zeigen dasselbe Modell, dasselbe
#    Bild, dieselbe Klasse — und ein völlig anderes Ergebnis. Die Regelwahl gehört deshalb
#    genauso dokumentiert wie die Modellarchitektur.
# 2. **Composite ist die Standardempfehlung**: flat unten, αβ in der Mitte, ε oben. Eine einzige
#    Regel für das ganze Netz ist in der Regel die schlechtere Wahl.
# 3. **Rot heißt „pro erklärte Klasse", blau „contra".** Immer relativ zur gewählten Klasse —
#    dasselbe Pixel kann in beiden Farben erscheinen (Abschnitt 10).
# 4. **Normalisierung ist Teil der Methode, nicht Kosmetik.** Ein leeres Panel bedeutet meist
#    einen Skalierungsfehler (Abschnitt 5), und Max-Normalisierung kann von drei Pixeln zerstört
#    werden (Abschnitt 6). Perzentil-basiertes Clipping ist robuster.
# 5. **Absolutwerte sind nicht vergleichbar** — nicht zwischen Regeln, nicht zwischen Bildern,
#    nicht zwischen Klassen, und bei panelweiser Normalisierung auch nicht innerhalb einer
#    Abbildung.
# 6. **Artefakte kennen und benennen**: blaue Ränder durch Padding (Abschnitt 7), Rastermuster
#    durch *winner-takes-all*-Pooling, Relevanz-Leck durch Bias-Terme (Abschnitt 6). Nicht jede
#    Struktur in einer Heatmap ist ein Befund über das Modell.
# 7. **Gegen eine Referenz validieren.** Der Vergleich mit iNNvestigate ist der eigentliche
#    Existenzgrund von Teil A. Ein XAI-Verfahren liefert immer ein plausibel aussehendes Bild —
#    auch wenn es falsch implementiert ist.
#
# ### Wie es weitergeht
#
# Dieses Notebook ist der kontrollierte Laborfall: bekanntes Modell, bekannte Klassen, bekannte
# Referenzwerte. Die anderen Notebooks dieses Repositories wenden dieselben Regeln auf Fälle an,
# in denen die richtige Antwort **nicht** bekannt ist:
#
# | Notebook | Aufgabe |
# |---|---|
# | `Train_and_explain_dummy_geometric_data` | synthetische Formen mit bekannter Ground Truth — der Test, ob LRP die *tatsächlich* eingebaute Ursache findet |
# | `Train_and_explain_3D_mnist_model` | Übergang von 2D zu 3D-Volumendaten |
# | `Train_and_explain_synthetic_brain_regression_model` | Regression statt Klassifikation, auf synthetischen Hirnbildern |
# | `Explain_brain_age_predictions` | der echte Anwendungsfall: Brain Age auf MRT-Daten, mit Auswertung pro Hirnregion |
#
# Wer LRP über dieses Repository hinaus verwenden will: [**Zennit**](https://github.com/chr5tphr/zennit)
# (PyTorch), [**iNNvestigate**](https://github.com/albermax/innvestigate) (Keras/TensorFlow) und
# **LXT** (LRP für Transformer). Die interaktive Demo unter
# [lrpserver.hhi.fraunhofer.de](https://lrpserver.hhi.fraunhofer.de/) erlaubt es, die
# Regelparameter im Browser zu verschieben und den Effekt sofort zu sehen.

# %%
