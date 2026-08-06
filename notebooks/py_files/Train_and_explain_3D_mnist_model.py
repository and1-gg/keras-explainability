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
# # Ein 3D-CNN für Ziffern-Volumen — trainieren und erklären
#
# ## Worum geht es in diesem Notebook?
#
# Wir trainieren ein **dreidimensionales Convolutional Neural Network** (3D-CNN) darauf, handgeschriebene
# Ziffern zu erkennen — allerdings nicht als flaches Bild, sondern als **Würfel aus 16×16×16
# Volumenpixeln** („Voxel"). Danach stellen wir dem fertigen Netz die entscheidende Frage:
#
# > **„Welche Voxel haben dich zu dieser Entscheidung gebracht?"**
#
# Die Antwort kommt als **Relevanzkarte** (Heatmap): jedem der 4096 Eingabe-Voxel wird eine Zahl
# zugewiesen — *wie stark hat dieses Voxel für (rot) oder gegen (blau) die erklärte Ziffer
# gesprochen?* Das Verfahren dazu heißt **Layer-wise Relevance Propagation (LRP)**.
#
# ## Der größere Kontext: warum überhaupt XAI?
#
# Ein neuronales Netz lernt seine Regeln aus Daten und versteckt sie in Millionen von Gewichten.
# Der Kernsatz der XAI-Community („eXplainable AI") lautet:
#
# > *„Just because a model is right doesn't mean it got there for the right reason."*
#
# Ein Netz könnte eine Ziffer auch an einem Bildrand-Artefakt, an der Schreibdicke oder an der
# Orientierung im Raum erkennen. Solche Fälle heißen **Clever-Hans-Prädiktoren** (nach dem
# „rechnenden Pferd", das in Wahrheit die Körpersprache seines Besitzers las). Erklärungsverfahren
# wie LRP machen so etwas sichtbar.
#
# ## Warum ausgerechnet 3D und ausgerechnet Ziffern?
#
# Dieses Notebook ist ein **Trainingslager für medizinische Bildgebung**. MRT-, CT- und
# PET-Aufnahmen sind grundsätzlich Volumen, keine Bilder — und genau dafür sind die anderen
# Notebooks in diesem Repository gedacht (Hirnalter-Vorhersage, Demenz-Klassifikation). Nur ist ein
# Gehirnvolumen mit 256³ Voxeln unhandlich: das Training dauert Stunden, und **niemand weiß, was die
# richtige Antwort wäre**.
#
# 3D-MNIST löst beide Probleme:
#
# | Eigenschaft | Vorteil |
# |---|---|
# | 16³ = 4096 Voxel statt Millionen | Training in Minuten statt Tagen, läuft auf einer Laptop-GPU |
# | Wir wissen, was drin ist | Zeigt die Heatmap auf leeren Raum, ist etwas faul |
# | Zehn klar getrennte Klassen | Man kann fragen: „warum *diese* Ziffer und nicht jene?" |
# | Dieselbe `Conv3D`-Maschinerie wie beim MRT | Alles hier Gelernte überträgt sich 1:1 |
#
# Kurz: **Wenn die Erklärungs-Pipeline hier nicht funktioniert, funktioniert sie beim Gehirn erst
# recht nicht.**
#
# ## Der Unterschied zwischen 2D und 3D in einem Bild
#
# ```text
#   2D-Bild (klassisches MNIST)          3D-Volumen (dieses Notebook)
#   ┌───────────────┐                    ┌───────────────┐
#   │               │                    │  ╱───────────╲│   16 Schichten
#   │      ███      │  28 × 28 Pixel     │ ╱   ███  ╱   ╱│   hintereinander
#   │       █       │  = 784 Zahlen      │╱    █   ╱   ╱ │   16 × 16 × 16
#   │      ███      │                    │    ███ ╱   ╱  │   = 4096 Zahlen
#   └───────────────┘                    └───────────────┘
#   Filter: 3 × 3 = 9 Gewichte           Filter: 3 × 3 × 3 = 27 Gewichte
# ```
#
# Ein 3D-Filter „sieht" also nicht nur nach links/rechts/oben/unten, sondern auch **nach vorn und
# hinten**. Das ist der ganze konzeptionelle Unterschied — und gleichzeitig der Grund, warum
# 3D-Netze so viel teurer sind.
#
# ## LRP in drei Sätzen
#
# 1. **Observe** — ein normaler Vorwärtsdurchlauf liefert die Vorhersage.
# 2. **Redistribute** — der Ausgabewert der Zielklasse wird Schicht für Schicht **rückwärts**
#    verteilt, jeweils proportional dazu, wie stark ein Neuron zur Aktivierung des nachfolgenden
#    beigetragen hat.
# 3. **Reveal** — am Ende liegt die gesamte „Relevanz" auf den Eingabe-Voxeln und ergibt die Heatmap.
#
# Die zentrale Eigenschaft ist die **Erhaltung (Conservation)**: LRP erzeugt und vernichtet keine
# Relevanz, es verteilt sie nur um.
#
# $$\sum_j R_j \;=\; \sum_k R_k \qquad \text{für jede Schicht}$$
#
# Wichtig zu verinnerlichen: **Relevanz ist nicht dasselbe wie Aktivierung.** Ein schwach
# aktiviertes Neuron kann sehr relevant sein, wenn genau es den Ausschlag gab.
#
# ## Ablauf dieses Notebooks
#
# ```text
#   ┌─ Teil A: Modell bauen ───────────────────────────────────────────────┐
#   │  10.000 Volumen ──► [3D-CNN, 3,5 Mio. Parameter] ──► 10 Wahrscheinl. │
#   │  (Abschnitte 1–5)                                                    │
#   └──────────────────────────────────────────────────────────────────────┘
#                                    │
#   ┌─ Teil B: Modell erklären ──────▼─────────────────────────────────────┐
#   │  1 Volumen (eine „3") ──► LRP rückwärts ──► 10 Relevanz-Volumen      │
#   │  (Abschnitte 6–8)          einmal je Zielklasse 0…9                  │
#   └──────────────────────────────────────────────────────────────────────┘
#                                    │
#   ┌─ Teil C: Interaktiv in 3D ─────▼─────────────────────────────────────┐
#   │  Punktwolke: Eingabe  ‖  Punktwolke: Relevanz  (drehbar)             │
#   │  (Abschnitte 9–11)                                                   │
#   └──────────────────────────────────────────────────────────────────────┘
# ```
#
# ---
#
# <a id="toc"></a>
# ## Inhaltsverzeichnis
#
# | # | Abschnitt | Inhalt |
# |---|---|---|
# | 1 | [Die Daten: 3D-MNIST laden und ansehen](#sec-01) | HDF5, Voxelgitter, One-Hot, Slice-Ansicht |
# | 2 | [Das 3D-CNN: Architektur](#sec-02) | `Conv3D`, BatchNorm, Pooling, 3,5 Mio. Parameter |
# | 3 | [Exkurs: Farbskalen richtig lesen](#sec-03) | warum `jet` für Heatmaps ungeeignet ist |
# | 4 | [Training: Verlust, Optimierer, Callbacks](#sec-04) | Cross-Entropy, Adam, `ReduceLROnPlateau` |
# | 5 | [Das Modell: laden oder trainieren](#sec-05) | `.keras`-Format, Trennung Training / Analyse |
# | 6 | [LRP: zehn Erklärungen für ein Volumen](#sec-06) | Regelstrategie, Relevanzsummen vs. Logits |
# | 7 | [Die Slice-Galerie: 10 Klassen × 16 Schichten](#sec-07) | die zentrale Abbildung, Rot/Blau lesen |
# | 8 | [Die Differenzmatrix](#sec-08) | wie klassenspezifisch sind die Erklärungen wirklich? |
# | 9 | [Das Volumen als 3D-Punktwolke](#sec-09) | Plotly, Schwellwert, Rendering-Hinweis |
# | 10 | [Eine 3D-Zahl rein → Vorhersage raus](#sec-10) | die Minimal-Pipeline in fünf Zeilen |
# | 11 | [Input und Erklärung nebeneinander in 3D](#sec-11) | die anschaulichste Darstellung |
# | 12 | [Anhang: Rohdaten-Zelle](#sec-12) | Sandbox-Zelle, ohne Wirkung auf das Notebook |
# | 13 | [Fazit: was man mitnehmen sollte](#sec-13) | Stärken, Grenzen, Fallstricke |
#
# **Hintergrunddokument im Repo:**
# [`doc/LRP_Fraunhofer_HHI_Video_Zusammenfassung.md`](../../doc/LRP_Fraunhofer_HHI_Video_Zusammenfassung.md)
# — die LRP-Theorie in Worten, inklusive der Rot/Blau-Semantik.
#
# **Verwandtes Notebook:** [`Explain_2D_VGG_predictions`](Explain_2D_VGG_predictions.py) vergleicht
# die einzelnen LRP-Regeln (LRP-0, ε, αβ, flat, Composite) systematisch am 2D-Fall. Wer die
# Regelwahl in Abschnitt 6 genauer verstehen will, findet dort die Herleitungen.

# %% [markdown]
# <a id="sec-01"></a>
# ## 1. Die Daten: 3D-MNIST laden und ansehen
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Fünf Schritte in einer Zelle:
#
# 1. **Repository finden.** `find_repo_root()` läuft vom aktuellen Arbeitsverzeichnis nach oben, bis
#    es `pyproject.toml` oder den Ordner `explainability/` sieht, und hängt das Ergebnis an
#    `sys.path`. Erst dadurch ist später `from explainability import LRP` importierbar — egal, aus
#    welchem Ordner der Jupyter-Kernel gestartet wurde.
# 2. **HDF5-Datei öffnen.** `full_dataset_vectors.h5` ist der
#    [3D-MNIST-Datensatz von Kaggle](https://www.kaggle.com/daavoo/3d-mnist). HDF5 ist das
#    Standardformat für große wissenschaftliche Arrays; `h5py` liest es wie ein `dict`.
# 3. **Umformen zu Volumen.** In der Datei liegt jedes Beispiel als **flacher Vektor mit 4096
#    Zahlen**. `np.reshape(..., (-1, 16, 16, 16, 1))` faltet ihn zum Würfel auf:
#
#    $$4096 \;=\; 16 \times 16 \times 16$$
#
#    Die `-1` heißt „so viele Beispiele, wie eben da sind", die abschließende `1` ist die
#    **Kanaldimension** (bei einem Farbbild stünde dort 3 für RGB; hier gibt es nur einen
#    Graustufen-Kanal). Keras erwartet für `Conv3D` immer genau diese fünf Achsen:
#    `(Batch, Tiefe, Höhe, Breite, Kanäle)`.
# 4. **Spiegeln.** `train_X[:, ::-1, :, :]` dreht die **Achse 1** um. Reiner Anzeige-Kosmetik-Schritt:
#    `plt.imshow` zeichnet Zeile 0 ganz oben, das Voxelgitter zählt aber von unten. Ohne diese Zeile
#    stünden alle Ziffern auf dem Kopf. Weil die Spiegelung auf **Trainings- und Testdaten gleich**
#    angewandt wird, ändert sie am Lernproblem nichts.
# 5. **One-Hot-Kodierung.** Aus dem Label `3` wird der Vektor
#
#    $$\mathbf{y} = (0,0,0,\mathbf{1},0,0,0,0,0,0)$$
#
#    Warum? Die Verlustfunktion `categorical_crossentropy` vergleicht zwei
#    Wahrscheinlichkeitsverteilungen. Ohne One-Hot würde das Netz die Labels als **Zahlen** deuten
#    und „7 ist größer als 3" lernen — Unsinn, denn Ziffern sind Kategorien, keine Messwerte.
#
# ### Wie sehen die Daten konkret aus?
#
# | Größe | Wert |
# |---|---|
# | Trainingsbeispiele | 10.000 |
# | Testbeispiele | 2.000 |
# | Form pro Beispiel | (16, 16, 16, 1) = 4096 Voxel |
# | Wertebereich | 0,0 bis 1,0 (bereits normiert) |
# | Belegte Voxel | im Schnitt rund **15 %** — die Volumen sind also sehr **dünn besetzt** |
# | Klassenverteilung | 868 bis 1126 pro Ziffer, also näherungsweise ausgeglichen |
#
# Die Volumen entstehen, indem die 2D-MNIST-Ziffer zu einer **Punktwolke** gemacht, im Raum zufällig
# **rotiert** und wieder auf ein 16³-Gitter gerastert wird. Deshalb liegt dieselbe Ziffer in jedem
# Beispiel in einer anderen Lage — das Netz muss **rotationsrobust** werden, was die Aufgabe
# deutlich schwerer macht als klassisches 2D-MNIST.
#
# ### Was man auf den Abbildungen sieht
#
# Die Schleife zeigt die ersten fünf Trainingsbeispiele. Jede Abbildung ist ein **4×4-Raster mit den
# 16 Schnittebenen** eines einzigen Volumens; die Zahl über dem Raster ist das wahre Label. Man
# blättert also durch den Würfel wie durch die Schichten eines MRT-Scans (`cmap='Greys'`: schwarz =
# Voxel belegt, weiß = leer).
#
# | Beispiel | Label | Was zu sehen ist |
# |---|---|---|
# | 0 | 5 | Eine klare, leicht schräg stehende **5** wiederholt sich über ~7 Schichten. Die Ziffer liegt fast **parallel** zur Schnittebene. |
# | 1 | 5 | Nur kleine, kompakte Flecken pro Schicht. Dieselbe Ziffer steht hier fast **senkrecht** zur Schnittebene — wir sehen sie „von der Kante". |
# | 2 | 0 | Ein deutlicher **Ring**, ebenfalls schichtweise wiederholt. |
# | 3 | 0 | Dieselbe Null, aber schräg gekippt: pro Schicht nur noch kleine Ringfragmente. |
# | 4 | 4 | Eine gut lesbare **4**, wieder nahezu parallel zur Schnittebene. |
#
# ### Interpretation
#
# Die fünf Abbildungen sind **die wichtigste Lektion dieses Abschnitts**: Beispiel 0 und Beispiel 1
# tragen dasselbe Label „5", sehen als Schnittbilder aber **völlig unterschiedlich** aus. Beispiel 1
# würde kein Mensch als Fünf erkennen — die Information steckt nicht in einer einzelnen Schicht,
# sondern **erst in ihrer Kombination**.
#
# Genau deshalb braucht man hier `Conv3D` und nicht `Conv2D`: Ein 2D-Filter sieht immer nur eine
# Schicht und hätte bei Beispiel 1 keine Chance. Ein 3D-Filter sieht 3×3×**3** Voxel und kann die
# Ziffer über die Tiefe hinweg zusammensetzen.
#
# ### Einordnung
#
# Das ist exakt dieselbe Situation wie bei einem MRT: Ein einzelner axialer Schnitt zeigt einen
# Hippocampus nur als kleinen grauen Fleck; seine Form ergibt sich erst über 20 Schichten. Wer
# medizinische Volumen mit 2D-Netzen scheibenweise verarbeitet (was durchaus üblich ist), wirft
# diese Information weg — und muss sie durch andere Tricks wieder hereinholen.

# %%
import sys
import os
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

import h5py
import matplotlib.pyplot as plt
import numpy as np

import ipynbname


def _process_ancestors(pid: int, levels: int = 5) -> list[int]:
    pids = []
    for _ in range(levels):
        pids.append(pid)
        try:
            stat = Path(f"/proc/{pid}/stat").read_text()
            # Feld 4 von stat ist die PPID; der Prozessname in Feld 2 kann
            # Leerzeichen enthalten, deshalb erst hinter ")" trennen
            pid = int(stat[stat.rindex(")") + 1:].split()[1])
        except Exception:
            break
        if pid <= 1:
            break
    return pids


def find_notebook_name() -> str:
    """Ermittelt den Namen des laufenden Notebooks ohne .ipynb-Endung."""
    # 1. Interaktiv (JupyterLab, VS Code): ipynbname fragt den Jupyter-Server
    #    bzw. liest __vsc_ipynb_file__
    try:
        return ipynbname.name()
    except Exception:
        pass

    # 2. jupyter-server >= 2 stellt den Notebook-Pfad in die Kernel-Umgebung,
    #    papermill & Co. setzen __session__
    for candidate in (os.environ.get("JPY_SESSION_NAME"), globals().get("__session__")):
        if candidate and candidate.endswith(".ipynb"):
            return Path(candidate).stem

    # 3. Als reines Skript ausgeführt (python notebooks/py_files/<name>.py)
    if "__file__" in globals():
        return Path(globals()["__file__"]).stem

    # 4. Unter nbconvert/quarto gibt es keine der obigen Quellen — dort steht der
    #    Notebook-Pfad aber in der Kommandozeile des aufrufenden Prozesses
    #    (JPY_PARENT_PID zeigt auf nbconvert, dessen Vorfahren auf make/quarto)
    start_pid = int(os.environ.get("JPY_PARENT_PID") or os.getppid())
    for pid in _process_ancestors(start_pid):
        try:
            args = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
        except Exception:
            continue
        for arg in args:
            if arg.endswith(".ipynb"):
                return Path(arg).stem

    raise RuntimeError(
        "Notebook-Name konnte nicht ermittelt werden — bitte NOTEBOOK_NAME setzen."
    )


notebook_name = os.environ.get("NOTEBOOK_NAME") or find_notebook_name()

target_dir = repo_root / "output" / "notebooks" / notebook_name
target_dir.mkdir(parents=True, exist_ok=True)

print(f"Notebook-Name ist: {notebook_name}")
print(f"Zielordner ist: {target_dir}")

data_path = repo_root / 'data' / '3d-mnist' / 'full_dataset_vectors.h5'

assert os.path.isfile(data_path), \
    'Download the 3d-mnist data from https://www.kaggle.com/daavoo/3d-mnist'

with h5py.File(data_path, 'r') as f:
    train_X = np.reshape(f["X_train"][:], (-1, 16, 16, 16, 1))
    train_y = f["y_train"][:]    
    test_X = np.reshape(f["X_test"][:]  , (-1, 16, 16, 16, 1))
    test_y = f["y_test"][:]

train_X = train_X[:,::-1,:,:]
test_X = test_X[:,::-1,:,:]

def onehot(values: np.ndarray) -> np.ndarray:
    encoded = np.zeros((len(values), 10))

    for i in range(len(values)):
        encoded[i,values[i]] = 1

    return encoded

train_y = onehot(train_y)
test_y = onehot(test_y)

for i in range(5):
    fig, ax = plt.subplots(4, 4)
    fig.suptitle(str(np.argmax(train_y[i])))
    ax = ax.ravel()

    for j in range(16):
        ax[j].imshow(train_X[i,:,:,j], cmap='Greys')
        ax[j].axis('off')

    plt.show()

# %% [markdown]
# <a id="sec-02"></a>
# ## 2. Das 3D-CNN: Architektur
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Diese Zelle **baut** das Netz — sie trainiert noch nichts. Alle Gewichte sind danach zufällig
# initialisiert. Verwendet wird die **Functional API** von Keras: man reicht einen Tensor durch
# Schichten hindurch (`x = Schicht(...)(x)`) und übergibt am Ende Ein- und Ausgang an `Model`.
#
# ### Die vier Bausteine
#
# **1. `Conv3D` — die Faltung.** Ein kleiner Würfel von Gewichten (hier immer 3×3×3 = **27
# Gewichte**) wird über das gesamte Volumen geschoben. An jeder Position berechnet er
#
# $$z \;=\; \sum_{u,v,w} a_{u,v,w}\, k_{u,v,w} \;+\; b$$
#
# also eine gewichtete Summe der 27 Nachbarvoxel. Der Filter lernt dadurch ein **lokales Muster**
# (eine Kante, eine Ecke, eine Krümmung), und weil derselbe Filter überall angewandt wird, findet er
# dieses Muster **an jeder Stelle** des Volumens. Das ist das Prinzip der *Translationsinvarianz*
# und der Grund, warum CNNs mit so wenig Parametern auskommen.
#
# `padding='SAME'` füllt den Rand mit Nullen auf, damit das Ausgabevolumen dieselbe Kantenlänge
# behält wie das Eingabevolumen.
#
# **2. `BatchNormalization` — die Normalisierung.** Nach jeder Faltung werden die Werte über den
# Batch auf Mittelwert 0 und Varianz 1 gebracht und dann mit zwei gelernten Zahlen $\gamma, \beta$
# wieder skaliert:
#
# $$\hat a \;=\; \gamma\,\frac{a - \mu}{\sqrt{\sigma^2 + \varepsilon}} \;+\; \beta$$
#
# Das stabilisiert das Training erheblich. **Für LRP ist das ein Sonderfall**: eine BatchNorm-Schicht
# ist zur Inferenzzeit eine affine Abbildung und lässt sich vollständig in die vorangehende Faltung
# hineinrechnen. Genau das tut `explainability/model/utils/fuse_batchnorm.py`, bevor die Relevanz
# rückwärts propagiert wird — deshalb tauchen die `normX`-Schichten in der LRP-Strategie
# (Abschnitt 6) gar nicht auf.
#
# **3. `Activation('relu')` — die Nichtlinearität.** $\mathrm{ReLU}(x) = \max(0, x)$. Ohne sie wäre
# das ganze Netz nur eine einzige große lineare Abbildung, egal wie viele Schichten man stapelt.
#
# **4. `MaxPooling3D((2,2,2))` — die Verkleinerung.** Aus jedem 2×2×2-Block wird nur das Maximum
# behalten. Die Kantenlänge halbiert sich, das **Volumen achtelt sich**. Nach drei Pooling-Stufen
# ist aus 16³ = 4096 nur noch 2³ = 8 geworden. Gleichzeitig wächst die Zahl der Filter
# (32 → 64 → 128 → 256): **räumliche Auflösung wird gegen semantischen Reichtum getauscht**.
#
# ### Der Aufbau im Überblick
#
# | Block | Schichten | Ausgabeform | Was hier entsteht |
# |---|---|---|---|
# | Eingabe | `inputs` | (16, 16, 16, 1) | das rohe Voxelgitter |
# | 1 | `conv1`, `conv2`, `pool1` | (8, 8, 8, 32) | Kanten und Oberflächen |
# | 2 | `conv3`, `conv4`, `pool2` | (4, 4, 4, 64) | Krümmungen, Strichkreuzungen |
# | 3 | `conv5`, `conv6`, `pool3` | (2, 2, 2, 128) | Ziffernteile (Bögen, Balken) |
# | 4 | `conv7`, `conv8` | (2, 2, 2, 256) | ganze Ziffernkonzepte |
# | Kopf | `pool4` (GlobalAvgPool) | (256,) | jedes Filter auf **eine Zahl** reduziert |
# | Kopf | `dense` → `preds` | (10,) | zehn Wahrscheinlichkeiten |
#
# `GlobalAveragePooling3D` mittelt jedes der 256 Filter über alle 8 Positionen. Das ersetzt das
# klassische `Flatten` und spart enorm Parameter: `Flatten` hätte 2·2·2·256 = 2048 Werte an die
# Dense-Schicht gegeben, so sind es nur 256. Es macht das Netz zusätzlich robust gegen
# Verschiebungen — allerdings **verwischt es auch die Ortsinformation**, was bei der Interpretation
# der Heatmaps in Abschnitt 7 noch eine Rolle spielt.
#
# ### Regularisierung: zwei Mechanismen gegen Auswendiglernen
#
# Bei 10.000 Beispielen und 3,5 Mio. Parametern könnte das Netz den Trainingssatz schlicht
# auswendig lernen. Zwei Gegenmaßnahmen sind eingebaut:
#
# * **L2-Regularisierung** (`weight_decay = 1e-3`): Zum Verlust wird $\lambda \sum_i w_i^2$ addiert.
#   Große Gewichte werden also bestraft, das Netz bevorzugt „glatte" Lösungen.
# * **Dropout** (`dropout = 0.3`): Beim Training werden 30 % der Neuronen zufällig auf 0 gesetzt.
#   Das Netz kann sich nicht auf einzelne Neuronen verlassen. Bei der Inferenz ist Dropout inaktiv —
#   deshalb stören die beiden `Dropout`-Schichten LRP nicht.
#
# ### Was die Ausgabe zeigt
#
# `model.summary()` listet **34 Schichten** und meldet:
#
# ```text
# Total params:         3.519.680 (13,43 MB)
# Trainable params:     3.517.740
# Non-trainable params:     1.940
# ```
#
# Die 1.940 nicht-trainierbaren Parameter sind die **laufenden Mittelwerte und Varianzen** der
# BatchNorm-Schichten: sie werden nicht durch Gradientenabstieg gelernt, sondern während des
# Trainings mitgezählt.
#
# Zwei Zahlen lohnen einen zweiten Blick:
#
# * `conv8` allein hat **1.769.728 Parameter** — die Hälfte des gesamten Netzes. Grund:
#   $256 \times 256 \times 27 + 256 = 1.769.728$. Die Parameterzahl einer Faltung wächst mit dem
#   **Produkt** von Eingangs- und Ausgangskanälen.
# * `preds` hat nur **110 Parameter** ($10 \times 10 + 10$). Die eigentliche Entscheidung fällt also
#   längst vorher; die letzte Schicht sortiert nur noch.
#
# ### ⚠️ Eine Auffälligkeit in der Architektur
#
# Nach `norm6` folgt **keine** `Activation('relu')` — `pool3` schließt direkt an. In allen anderen
# Blöcken steht dort ein ReLU. In der Summary sieht man es an der Nummerierung: es gibt nur
# `activation` bis `activation_7`, also **8 statt 9** Aktivierungen. Ob das Absicht oder ein
# Tippfehler ist, lässt sich von außen nicht sagen; das Netz trainiert dadurch nicht falsch, aber
# `conv6` und `conv7` sind faktisch **linear hintereinandergeschaltet** und ließen sich theoretisch
# zu einer Schicht zusammenfassen. Für LRP ist das unkritisch (die Relevanz fließt einfach durch),
# es ist aber ein gutes Beispiel dafür, dass ein Blick in `model.summary()` mehr verrät, als man
# denkt.
#
# ### Einordnung
#
# Diese Architektur — abwechselnd zwei Faltungen und ein Pooling, Kanäle verdoppeln,
# GlobalAveragePooling am Ende — ist ein **VGG-artiges Standardmuster**. Genau dieselbe Struktur
# findet sich in den Hirn-Notebooks dieses Repositories, nur mit größerem Eingabevolumen und mehr
# Blöcken. Wer sie hier versteht, versteht sie dort auch.

# %%
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Conv3D, Dense, \
                                    Dropout, Flatten, \
                                    GlobalAveragePooling3D, Input, \
                                    MaxPooling3D
from tensorflow.keras.regularizers import l2

inputs = Input((16, 16, 16, 1), name='inputs')

x = inputs

kernel = (3, 3, 3)
dropout = 0.3
weight_decay = 1e-3
regularizer = l2(weight_decay)

x = Conv3D(32, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv1')(x)
x = BatchNormalization(name='norm1')(x)
x = Activation('relu')(x)
x = Conv3D(32, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv2')(x)
x = BatchNormalization(name='norm2')(x)
x = Activation('relu')(x)
x = MaxPooling3D((2, 2, 2), name='pool1')(x)
x = Conv3D(64, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv3')(x)
x = BatchNormalization(name='norm3')(x)
x = Activation('relu')(x)
x = Conv3D(64, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv4')(x)
x = BatchNormalization(name='norm4')(x)
x = Activation('relu')(x)
x = MaxPooling3D((2, 2, 2), name='pool2')(x)
x = Conv3D(128, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv5')(x)
x = BatchNormalization(name='norm5')(x)
x = Activation('relu')(x)
x = Conv3D(128, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv6')(x)
x = BatchNormalization(name='norm6')(x)
x = MaxPooling3D((2, 2, 2), name='pool3')(x)
x = Conv3D(256, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv7')(x)
x = BatchNormalization(name='norm7')(x)
x = Activation('relu')(x)
x = Conv3D(256, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv8')(x)
x = BatchNormalization(name='norm8')(x)
x = Activation('relu')(x)
x = GlobalAveragePooling3D(name='pool4')(x)
x = Dropout(dropout)(x)
x = Dense(10, kernel_regularizer=regularizer, activation=None, name='dense')(x)
x = BatchNormalization(name='norm9')(x)
x = Activation('relu')(x)
x = Dropout(dropout)(x)
x = Dense(10, activation='softmax', name='preds')(x)

model = Model(inputs, x)
model.summary()


# %% [markdown]
# <a id="sec-03"></a>
# ## 3. Exkurs: Farbskalen richtig lesen
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Diese Zelle hat **nichts mit dem Modell zu tun** — sie ist ein didaktischer Einschub. Berechnet
# wird lediglich:
#
# 1. `np.linspace(0, 100, 100)` erzeugt 100 gleichmäßig verteilte Zahlen von 0 bis 100,
# 2. `.reshape((10, 10))` ordnet sie zeilenweise in ein 10×10-Gitter (Zeile 0 enthält 0…9,09,
#    Zeile 9 enthält 90,9…100),
# 3. `plt.imshow(data, cmap='jet')` malt jede Zahl als Farbe,
# 4. `plt.colorbar(img)` hängt die **Legende** daneben.
#
# ### Was man auf der Abbildung sieht
#
# Ein glatter Farbverlauf von **oben nach unten**: dunkelblau (0) → blau → cyan → grün → gelb →
# orange → rot → dunkelrot (100). Rechts die Colorbar mit der Beschriftung „Zahlenwerte".
#
# ### Interpretation und warum das wichtig ist
#
# Der Kernpunkt: **Ein Farbbild ohne Colorbar ist nicht interpretierbar.** Man sieht Muster, kann
# aber keine Größe zuordnen. In allen folgenden Heatmaps muss man deshalb immer wissen, welche Farbe
# welchen Zahlenwert bedeutet.
#
# Und noch etwas: `jet` ist für Heatmaps **die falsche Wahl**, obwohl (oder weil) sie so bunt ist:
#
# | Problem mit `jet` | Konsequenz |
# |---|---|
# | Nicht *perzeptuell uniform* — der Sprung cyan → grün wirkt viel größer als grün → gelb | Man „sieht" Strukturen, die in den Daten nicht existieren |
# | Kein ausgezeichneter Mittelpunkt | Bei Werten, die positiv *und* negativ sein können, ist nicht erkennbar, wo die Null liegt |
# | In Graustufen gedruckt nicht mehr monoton | Publikationen werden unlesbar |
#
# Deshalb verwendet dieses Notebook ab Abschnitt 7 die **divergierende** Skala `seismic`:
#
# $$\text{blau} \;\longleftarrow\; \text{weiß} \;\longrightarrow\; \text{rot}$$
#
# zusammen mit `clim=(-1, 1)`. Das `clim` ist entscheidend: Es fixiert die Skala so, dass **weiß
# immer exakt 0** bedeutet. Ohne diese Angabe würde matplotlib automatisch auf den vorhandenen
# Wertebereich skalieren — die Null läge dann irgendwo, und rot/blau hätten keine feste Bedeutung
# mehr.
#
# ### Einordnung
#
# Die Farbwahl ist bei XAI kein Schönheitsthema, sondern Teil der Aussage. Eine Relevanzkarte hat
# ein **Vorzeichen** (pro/contra) und einen **natürlichen Nullpunkt** (irrelevant) — beides muss die
# Farbskala abbilden. Für Größen ohne Nullpunkt (Voxelintensitäten, Wahrscheinlichkeiten) nimmt man
# dagegen eine sequentielle Skala wie `viridis` oder `Greys`.

# %%
import numpy as np
import matplotlib.pyplot as plt

# 1. Beispieldaten erstellen (z. B. Werte von 0 bis 100)
data = np.linspace(0, 100, 100).reshape((10, 10))

# 2. Daten mit der 'jet'-Farbpalette darstellen
plt.figure(figsize=(6, 5))
img = plt.imshow(data, cmap='jet')

# 3. Colorbar hinzufügen (Zahlenwerte werden automatisch angezeigt)
cbar = plt.colorbar(img)
cbar.set_label('Zahlenwerte', rotation=270, labelpad=15)

plt.title("Beispiel für Colorbar mit 'jet'")
plt.tight_layout()
plt.show()

# %% [markdown]
# <a id="sec-04"></a>
# ## 4. Training: Verlust, Optimierer, Callbacks
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# `model.compile(...)` legt fest, **wie** gelernt werden soll — es rechnet noch nichts. Drei
# Angaben:
#
# **1. Verlustfunktion `categorical_crossentropy`.** Sie misst den Abstand zwischen der vorhergesagten
# Verteilung $\hat{y}$ und dem One-Hot-Label $y$:
#
# $$L \;=\; -\sum_{c=0}^{9} y_c \,\log \hat{y}_c$$
#
# Weil $y$ One-Hot ist, bleibt davon nur **ein** Term übrig: $L = -\log \hat{y}_{\text{wahr}}$. Sagt
# das Netz für die richtige Klasse 0,99 voraus, ist $L \approx 0{,}01$; sagt es 0,01, ist
# $L \approx 4{,}6$. Der Logarithmus bestraft **selbstbewusste Fehler** also extrem hart — genau das
# will man.
#
# **2. Optimierer `Adam(1e-3)`.** Gradientenabstieg mit adaptiver Schrittweite pro Parameter. Die
# Lernrate $10^{-3}$ ist der übliche Startwert.
#
# **3. Metrik `accuracy`.** Nur zur Anzeige — der Anteil korrekt klassifizierter Beispiele. Die
# Metrik wird *nicht* optimiert (Genauigkeit ist nicht differenzierbar), sie ist aber die Zahl, die
# ein Mensch versteht.
#
# ### Der Callback: `ReduceLROnPlateau`
#
# ```python
# ReduceLROnPlateau(monitor="loss", factor=0.1, patience=5, verbose=1)
# ```
#
# Übersetzt: *„Wenn sich der Trainingsverlust 5 Epochen lang nicht verbessert, multipliziere die
# Lernrate mit 0,1."* Aus $10^{-3}$ wird also $10^{-4}$, dann $10^{-5}$.
#
# Die Idee ist anschaulich: Große Schritte bringen einen schnell ins Tal, aber am Talboden springt
# man darüber hinweg. Dann muss man kleiner treten. Ein solcher **Lernraten-Zeitplan** ist der
# billigste bekannte Weg zu ein paar Prozentpunkten mehr Genauigkeit.
#
# ⚠️ Hier wird `monitor="loss"` überwacht, also der **Trainings**verlust. Üblicher (und sicherer)
# wäre `val_loss`: Sinkt nur der Trainingsverlust weiter, während der Validierungsverlust steigt,
# lernt das Netz auswendig — das würde dieser Callback nicht bemerken.
#
# ### Wo bleibt `model.fit`?
#
# Diese Zelle **bereitet das Training nur vor**. Der eigentliche Aufruf steht in Abschnitt 5, denn
# erst dort wird entschieden, ob überhaupt trainiert werden muss: Liegt schon ein gespeichertes
# Modell vor, wird es geladen und `model.fit` übersprungen.
#
# Der Grund für diese Trennung: 100 Epochen dauern auf einer Laptop-GPU einige Minuten. Dieses
# Notebook ist in erster Linie ein **Analyse**-Notebook; man soll es von oben bis unten durchlaufen
# lassen können, ohne jedes Mal neu zu trainieren.
#
# Die Einstellungen, mit denen Abschnitt 5 im Trainingsfall arbeitet:
#
# | Parameter | Wert | Bedeutung |
# |---|---|---|
# | `epochs=100` | 100 Durchläufe durch alle 10.000 Beispiele | |
# | `batch_size=32` | 32 Volumen pro Gradientenschritt → 313 Schritte je Epoche | |
# | `shuffle=True` | Reihenfolge jede Epoche neu mischen | verhindert Ordnungseffekte |
# | `validation_data=(test_X, test_y)` | nach jeder Epoche auf den 2.000 Testdaten messen | ehrliche Zwischenbilanz |
#
# ### Einordnung
#
# Die strikte Trennung „Training einmal, Analyse beliebig oft" ist bei 3D-Daten keine Bequemlichkeit,
# sondern Notwendigkeit: Beim MRT-Pendant dieses Notebooks dauert ein Trainingslauf Stunden bis Tage
# auf dedizierter Hardware. Das Modell ist dort ein **Artefakt**, das man versioniert, ablegt und
# wiederverwendet — genau wie hier in `output/notebooks/<Notebook-Name>/`.

# %%
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-3),
              metrics=['accuracy'])

callbacks = [
    ReduceLROnPlateau(
        monitor="loss",
        factor=0.1,
        patience=5,
        verbose=1
    )
]

# %% [markdown]
# <a id="sec-05"></a>
# ## 5. Das Modell: laden oder trainieren
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Die Zelle sucht mit `MODEL_DIR.glob("*.keras")` nach einem bereits gespeicherten Modell und
# verzweigt danach:
#
# * **Datei gefunden** → `load_model(...)` **überschreibt** die Variable `model`. Das in Abschnitt 2
#   gebaute Netz mit seinen Zufallsgewichten wird verworfen und durch das trainierte ersetzt. Es
#   wird *nicht* trainiert.
# * **Keine Datei** → es wird 100 Epochen lang trainiert und das Ergebnis mit `model.save(...)`
#   abgelegt. Beim nächsten Durchlauf greift dann der obere Zweig.
#
# Architektur, Gewichte, Optimierer-Zustand und Trainingskonfiguration stecken alle in der einen
# Datei `3d_mnist_cnn.keras`. `MODEL_DIR.mkdir(parents=True, exist_ok=True)` legt den Ordner an,
# falls er fehlt, und beschwert sich nicht, wenn er schon existiert.
#
# Dieses **Caching-Muster** verwenden alle Trainings-Notebooks dieses Repos. Das Modell landet unter
# `output/notebooks/<Notebook-Name>/100_epochs/` — also im selben Ausgabeordner wie die Grafiken,
# jedes Notebook in seinem eigenen. Wer bewusst neu trainieren will, löscht oder verschiebt die
# `.keras`-Datei.
#
# ⚠️ Die Suche per `glob` nimmt die **erste** `.keras`-Datei in alphabetischer Reihenfolge — liegen
# dort mehrere Modelle, wird nicht unbedingt das geladen, das `MODEL_PATH` benennt.
#
# ### Ausgabe
#
# ```text
# Model geladen von: …/output/notebooks/Train_and_explain_3D_mnist_model/100_epochs/3d_mnist_cnn.keras
# ```
#
# ### Warum `.keras` und nicht `.h5`?
#
# `.keras` ist seit Keras 3 das empfohlene Format. Es ist intern ein ZIP-Archiv aus
# JSON-Konfiguration und Gewichtsdatei und speichert im Gegensatz zum alten HDF5-Format auch
# benutzerdefinierte Objekte zuverlässig. Der Ordnername `100_epochs` dokumentiert nebenbei die
# wichtigste Trainings-Einstellung — ein simples, aber wirksames Ordnungsprinzip.
#
# ### ⚠️ Was hier fehlt
#
# Es wird **keine Testgenauigkeit ausgegeben**. Ein `model.evaluate(test_X, test_y)` würde hier gut
# hinpassen, denn:
#
# > **Eine Erklärung ist nur so viel wert wie das Modell, das sie erklärt.**
#
# Eine hübsche Heatmap für ein Netz mit 40 % Genauigkeit erklärt vor allem, wie das Netz sich irrt.
# Das ist einer der häufigsten Anfängerfehler in der XAI: die Heatmap interpretieren, ohne vorher die
# Modellgüte zu prüfen. Immerhin liefert Abschnitt 10 später eine Stichprobe (Vorhersage vs. wahres
# Label für ein Beispiel) — eine echte Evaluation auf dem Testsatz ersetzt das nicht.
#
# ### Einordnung
#
# Ab hier ist das Notebook ein reines **Analyse-Notebook**. Alles Folgende — Vorhersagen,
# Erklärungen, 3D-Plots — sind Vorwärtsdurchläufe durch ein eingefrorenes Modell. Kein Gewicht
# ändert sich mehr.

# %%
from tensorflow.keras.models import load_model

MODEL_DIR = target_dir / "100_epochs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "3d_mnist_cnn.keras"

existing_model_path = next(iter(sorted(MODEL_DIR.glob("*.keras"))), None)

if existing_model_path is not None:
    model = load_model(existing_model_path)
    print(f"Model geladen von: {existing_model_path}")
else:
    model.fit(train_X, train_y,
              validation_data=(test_X, test_y),
              #epochs=2,
              epochs=100,
              batch_size=32,
              shuffle=True,
              callbacks=callbacks)

    model.save(MODEL_PATH)
    print(f"Model gespeichert unter: {MODEL_PATH}")

# %% [markdown]
# <a id="sec-06"></a>
# ## 6. LRP: zehn Erklärungen für ein Volumen
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Das ist die **Kernzelle des Notebooks**. Hier wird zum ersten Mal nicht gefragt „was sagt das
# Netz?", sondern „**warum** sagt es das?".
#
# ### 6.1 Die Mathematik: wie LRP rückwärts rechnet
#
# Jedes Neuron $k$ berechnet im Vorwärtspass
#
# $$z_k \;=\; \sum_j a_j \, w_{jk} \;+\; b_k$$
#
# LRP dreht das um: Hat Neuron $k$ die Relevanz $R_k$ zugewiesen bekommen, gibt es sie an seine
# Vorgänger $j$ weiter — **proportional zu deren Beitrag** $a_j w_{jk}$:
#
# $$R_j \;=\; \sum_k \frac{a_j\,w_{jk}}{\sum_{j'} a_{j'} w_{j'k}} \; R_k$$
#
# Der Bruch summiert sich über $j$ zu 1, deshalb gilt die **Erhaltung** $\sum_j R_j = \sum_k R_k$:
# Relevanz wird nur umverteilt, nie erzeugt. Das ist die **z-Regel** oder **LRP-0**.
#
# Startpunkt der Rückwärtsrechnung ist eine Maske auf der Ausgabeschicht: nur das Neuron der
# erklärten Klasse behält seinen Wert, alle anderen werden auf 0 gesetzt.
#
# $$R_k^{(\text{letzte Schicht})} \;=\; \begin{cases} z_{\text{idx}} & k = \text{idx} \\ 0 & \text{sonst}\end{cases}$$
#
# **Wichtig:** Verwendet wird der Wert **vor** der Softmax, das sogenannte *Logit*. `LRP` entfernt
# die Softmax-Aktivierung intern selbst (`remove_activation`). Das ist Absicht — die Softmax
# normalisiert über alle zehn Klassen, ihr Gradient vermischt sie und macht Erklärungen unschärfer.
#
# ### 6.2 Warum nicht überall dieselbe Regel? Die Composite-Strategie
#
# LRP-0 in Reinform rauscht stark: Der Nenner $\sum_{j'} a_{j'} w_{j'k}$ kann fast null werden, wenn
# sich positive und negative Beiträge aufheben — dann explodiert der Bruch. Über zehn Schichten
# hinweg multipliziert sich das auf. Deshalb setzt man **pro Schichtbereich eine eigene Regel** ein
# (*Composite-LRP*, Montavon et al. 2019):
#
# | Regel | Formel | Wirkung |
# |---|---|---|
# | **LRP-ε** | $R_j = \sum_k \dfrac{a_j w_{jk}}{z_k + \epsilon\,\mathrm{sign}(z_k)} R_k$ | $\epsilon$ dämpft Neuronen mit $\lvert z_k\rvert \approx 0$ gegen null → **Rauschfilter** |
# | **LRP-αβ** | $R_j = \sum_k \left(\alpha \dfrac{(a_j w_{jk})^{+}}{z_k^{+}} - \beta \dfrac{(a_j w_{jk})^{-}}{z_k^{-}}\right) R_k$ | trennt Pro- von Contra-Evidenz, sehr stabil |
# | **b-Regel** | $a_j \equiv 1$ statt der echten Aktivierung | macht die Verteilung **unabhängig vom Voxelwert** |
#
# Nebenbedingung für αβ: $\alpha - \beta = 1$. Der Code erzwingt das (`assert alpha == beta + 1`) —
# nur so bleibt die Relevanz erhalten. `alpha=2, beta=1` heißt also: positive Beiträge doppelt
# gewichten, negative einfach abziehen.
#
# ### 6.3 Die Strategie in dieser Zelle
#
# `LRPStrategy(layers=[...])` erwartet **genau einen Eintrag pro gewichtstragender Schicht**. Unser
# Netz hat 8 `Conv3D` + 2 `Dense` = **10** davon, also 10 Einträge. Die Liste ist **von der Eingabe
# zur Ausgabe** geordnet (intern wird sie mit `[::-1]` gedreht, weil das Erklärer-Netz rückwärts
# aufgebaut wird):
#
# | Eintrag | Regel | Schicht | Warum |
# |---|---|---|---|
# | 1 | `b=True`, α=1, β=0 | `conv1` | Aktivierungen durch Einsen ersetzen → die Heatmap hängt nicht mehr von der Voxel*helligkeit* ab, sondern nur von der *Lage*. Wirkt wie ein räumlicher Glätter. |
# | 2–8 | α=2, β=1 | `conv2`…`conv8` | stabile, kontrastreiche Verteilung durch die Faltungsblöcke |
# | 9–10 | ε = 0,25 | `dense`, `preds` | Rauschunterdrückung im Klassifikationskopf, wo sich viele Beiträge gegenseitig aufheben |
#
# Die vier Pooling-Schichten stehen nicht in der Liste. Sie behalten die Voreinstellung
# *winner-takes-all*: die gesamte Relevanz eines 2×2×2-Fensters geht an das Voxel, das im
# Vorwärtspass das Maximum lieferte.
#
# ### 6.4 Was die Zelle konkret rechnet
#
# ```python
# explainer = LRP(model, layer=len(model.layers)-1, idx=i, strategy=strategy)
# explanations[i] = explainer(train_X[654:655])
# ```
#
# | Argument | Bedeutung |
# |---|---|
# | `model` | das trainierte Netz |
# | `layer=33` | ab welcher Schicht rückwärts erklärt wird → `preds` (34 Schichten, Index 0-basiert) |
# | `idx=i` | **welches Ausgabeneuron** erklärt wird → Ziffer $i$ |
# | `strategy` | die Regelkombination aus 6.3 |
#
# `LRP` ist selbst ein **Keras-Modell**: Der Graph des Originalnetzes wird topologisch sortiert,
# umgedreht, und jede Schicht durch ihr LRP-Gegenstück ersetzt. Der Aufruf `explainer(...)` ist dann
# ein ganz normaler Vorwärtspass durch dieses Erklärer-Netz. Rechenaufwand: etwa wie ein
# Trainingsschritt.
#
# Die Schleife baut **zehn** solcher Erklärer — einen pro Zielklasse. Das Ergebnis ist ein Array
# `explanations` der Form `(10, 16, 16, 16, 1)`: für jede der zehn Ziffern ein vollständiges
# Relevanzvolumen. Das ist der entscheidende Punkt, den Anfänger oft übersehen:
#
# > **LRP erklärt nicht „die Vorhersage", sondern immer eine konkret gewählte Klasse.** Man kann
# > auch eine Klasse erklären lassen, die das Netz gar nicht gewählt hat — die Frage lautet dann:
# > „Was sprach für und gegen *diese* Ziffer?"
#
# `image_idx = 654` wählt ein Trainingsbeispiel aus. Es zeigt eine **3**, die etwa 615 der 4096
# Voxel belegt (15 %) und in den Schnittebenen 5 bis 10 liegt.
#
# ### 6.5 Was die Ausgabe zeigt
#
# ```text
# Predictions: [6.44e-08  3.85e-15  1.23e-03  9.99e-01  6.85e-15
#               4.55e-06  9.74e-08  3.67e-09  5.42e-07  7.86e-13]
# ```
#
# Das Netz ist sich mit **99,876 %** sicher, eine 3 zu sehen — und liegt richtig. Zweitbeste Klasse
# ist die 2 mit 0,12 %, alles andere ist praktisch null.
#
# Danach die zehn Relevanzsummen. Stellt man sie den tatsächlichen Logits gegenüber (dem Wert der
# Ausgabeneuronen vor der Softmax, also genau dem, womit LRP startet), ergibt sich:
#
# | Ziffer | Wahrscheinlichkeit | Logit $z_c$ | $\sum_j R_j$ | Anteil |
# |---|---|---|---|---|
# | **3** | 0,99876 | **+10,66** | **+8,34** | 78 % |
# | 2 | 1,23e-03 | +3,97 | +3,13 | 79 % |
# | 5 | 4,55e-06 | −1,64 | −1,34 | 82 % |
# | 8 | 5,42e-07 | −3,76 | −3,75 | 100 % |
# | 6 | 9,74e-08 | −5,48 | −4,05 | 74 % |
# | 0 | 6,44e-08 | −5,90 | −4,38 | 74 % |
# | 7 | 3,67e-09 | −8,76 | −7,34 | 84 % |
# | 9 | 7,86e-13 | −17,21 | −15,42 | 90 % |
# | 4 | 6,85e-15 | −21,95 | −20,87 | 95 % |
# | 1 | 3,85e-15 | −22,52 | −18,00 | 80 % |
#
# ### 6.6 Interpretation — drei Beobachtungen
#
# **1. Die Erhaltung funktioniert näherungsweise.** Die Relevanzsumme am Eingang entspricht dem
# Logit der erklärten Klasse — allerdings nur zu **74 bis 100 %**. Wo bleibt der Rest? Drei Lecks:
#
# * **Bias-Terme.** Der Code korrigiert mit `R = (R * z) / (z + bias)`; der Bias-Anteil an der
#   Aktivierung „verschluckt" Relevanz. Bei neun BatchNorm-Schichten, die in die Faltungen
#   hineingerechnet werden, summiert sich das.
# * **Die ε-Regel** vergrößert absichtlich den Nenner und verliert dadurch Relevanz.
# * **Die αβ-Regel** ist nur bei exakter Kompensation streng erhaltend.
#
# Für die Praxis heißt das: **Relevanzsummen sind eine gute Plausibilitätsprüfung, keine Bilanz.**
# Weicht die Summe um Größenordnungen vom Logit ab, ist etwas kaputt.
#
# **2. Die Reihenfolge bleibt erhalten.** Sortiert man die Ziffern nach Logit oder nach
# Relevanzsumme, ergibt sich fast dieselbe Reihenfolge (nur 1 und 4 tauschen die Plätze — deren
# Logits liegen mit −21,95 und −22,52 ohnehin fast gleichauf). LRP verzerrt die Rangfolge also
# nicht.
#
# **3. Das Vorzeichen der Summe ist das Vorzeichen des Logits.** Nur die Ziffern **3 und 2** haben
# eine positive Summe — es sind genau die beiden mit positivem Logit. Alle anderen sind negativ,
# manche stark (4: −20,87; 1: −18,00). Das ist zunächst nur Buchhaltung, hat aber eine drastische
# Folge für die Bilder im nächsten Abschnitt: **Eine Erklärung für eine Klasse mit negativem Logit
# ist zwangsläufig überwiegend blau.**
#
# ### Einordnung
#
# Ein negativer Logit bedeutet: *„Dieses Volumen spricht insgesamt **gegen** die Ziffer 4."* Die
# zugehörige Heatmap zeigt also nicht, wo eine 4 wäre, sondern **wo die Evidenz dagegen sitzt**.
# Anfänger interpretieren solche Karten regelmäßig falsch. Die Faustregel:
#
# > **Vor dem Deuten einer Heatmap immer die Vorhersage und das Vorzeichen des Logits ansehen.**

# %%
from explainability import LRP, LRPStrategy


alpha = 2
beta = 1

strategy = LRPStrategy(
    layers=[
        {'b': True, 'alpha': 1, 'beta': 0},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'epsilon': 0.25},
        {'epsilon': 0.25}
    ]
)

image_idx = 654

explanations = np.zeros((10, 16, 16, 16, 1))
predictions = model.predict(train_X[image_idx:image_idx + 1])
print(f'Predictions: {predictions[0]}')

for i in range(10):
    explainer = LRP(model, layer=len(model.layers)-1, idx=i, strategy=strategy)
    explanations[i] = explainer(train_X[image_idx:image_idx + 1])
    print(f'Sum evidence for {i}: {np.sum(explanations[i])}')

# %% [markdown]
# <a id="sec-07"></a>
# ## 7. Die Slice-Galerie: 10 Klassen × 16 Schichten
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Dies ist die **zentrale Abbildung des Notebooks**. Sie beantwortet die Frage: Wie sieht die
# Erklärung für jede der zehn Ziffern aus, Schicht für Schicht durch das Volumen?
#
# ### 7.1 Was passiert hier?
#
# Die Zelle baut ein Raster mit **20 Zeilen × 16 Spalten**. Je zwei Zeilen bilden einen Block:
#
# ```text
#   Zeile 2i     ── das Original-Volumen, Schnitte 0…15   (immer identisch!)
#   Zeile 2i+1   ── die Relevanzkarte für Ziffer i, dieselben Schnitte
# ```
#
# Die Beschriftung links (`ID i (p)`) nennt die Zielklasse und die vom Netz vergebene
# Wahrscheinlichkeit.
#
# Vor dem Zeichnen werden die Relevanzen in zwei Schritten aufbereitet:
#
# **1. Normierung auf $[-1, 1]$.**
#
# $$\tilde{R}^{(i)} \;=\; \frac{R^{(i)}}{\max_{xyz} \lvert R^{(i)}_{xyz}\rvert}$$
#
# Jede Klasse wird also **einzeln** auf ihr eigenes Maximum skaliert. Das macht die zehn Bilder
# überhaupt erst vergleichbar — die Rohsummen unterscheiden sich um mehr als eine Größenordnung
# (+8,34 bis −20,87). Der Preis: **Die absoluten Stärken sind danach nicht mehr ablesbar.** Eine
# knallrote Zelle in Block 2 und eine knallrote in Block 3 bedeuten nicht dasselbe.
#
# **2. Nullpunkt-Korrektur.** `explanations[i] -= explanations[i, 0, 0, 0, 0]` zieht den Wert des
# Eckvoxels ab. Die Annahme dahinter: Diese Ecke ist garantiert leer, ihre Relevanz sollte also 0
# sein; was übrig bleibt, ist ein systematischer Offset. In der Praxis liegt dieser Wert bei etwa
# $10^{-4}$ — die Korrektur ist also **praktisch wirkungslos**, aber als Gedanke sauber.
#
# ⚠️ Beide Operationen schreiben **in-place** in `explanations` zurück. Die nächste Zelle
# (Abschnitt 8) arbeitet deshalb bereits mit den normierten Werten, nicht mehr mit den Rohrelevanzen.
# Führt man diese Zelle zweimal aus, wird zweimal normiert — beim zweiten Mal ist das wirkungslos,
# weil das Maximum dann schon 1 ist.
#
# ### 7.2 Die beiden Farbskalen
#
# | Zeile | Aufruf | Skala | Bedeutung |
# |---|---|---|---|
# | Original | `imshow(...)` ohne `cmap` | `viridis` (Standard) | dunkelviolett = 0 (leer), grün/gelb = belegt |
# | Erklärung | `imshow(..., cmap='seismic', clim=(-1,1))` | divergierend | **rot = pro**, **weiß = irrelevant**, **blau = contra** |
#
# Dass die Originalschnitte violett statt weiß erscheinen, ist reine Konvention von `viridis`: Der
# **leere Raum ist der dunkelviolette Untergrund**. Nur die sechs Schichten mit Ziffer zeigen
# grün-gelbe Strukturen.
#
# ### 7.3 Was man auf der Abbildung sieht
#
# **Die Originalzeilen (jeder zweite Block) sind zehnmal identisch** — es ist immer dasselbe
# Volumen. Nur in den **Schnitten 5 bis 10** ist etwas zu sehen: eine diagonal im Raum liegende
# **3**. Die Schnitte 5 und 10 zeigen die **gefüllten Endflächen** (je ~138 Voxel), die Schnitte
# 6 bis 9 nur noch den **Umriss** (je 85 Voxel) — das Volumen ist innen hohl, wie eine aus der
# 2D-Ziffer extrudierte Schale.
#
# **Die Erklärungszeilen zerfallen in zwei Gruppen:**
#
# | Blöcke | Farbe | Logit |
# |---|---|---|
# | **ID 2 und ID 3** | deutlich **rot** | positiv (+3,97 / +10,66) |
# | alle übrigen | **blau**, unterschiedlich kräftig | negativ |
#
# **Alle zehn Karten zeigen dieselbe räumliche Struktur** — die Kontur der 3. Sie unterscheiden sich
# im Wesentlichen nur durch das **Vorzeichen** und die Intensität.
#
# **Die Relevanz reicht über die Ziffer hinaus.** Sichtbare Struktur liegt in den Schnitten **3 bis
# 12**, obwohl die Ziffer nur in 5 bis 10 existiert. Es bekommen also auch **leere Voxel Relevanz**
# — bei einer Nachrechnung landen rund 47 % der Gesamtrelevanz auf Voxeln mit Wert 0.
#
# ### 7.4 Interpretation
#
# **Warum sind ID 2 und ID 3 rot?** Weil bei diesen beiden Klassen der Startwert der
# Rückwärtsrechnung — das Logit — positiv ist. Das Netz sagt: „Dieses Volumen spricht für eine 3
# (stark) und für eine 2 (schwach)." Die roten Voxel sind die **Belege dafür**.
#
# **Warum ist ID 4 knallblau?** Logit −21,95, also die am stärksten ausgeschlossene Ziffer. Die Karte
# zeigt: *„Genau diese Voxel sind der Grund, warum das hier **keine 4** ist."* Das ist eine
# inhaltlich völlig andere Aussage als bei ID 3, sieht aber bis auf die Farbe **fast gleich aus**.
#
# **Warum sehen 2 und 3 so ähnlich aus?** Handgeschriebene 2 und 3 teilen sich den oberen Bogen; das
# Netz findet Evidenz für beide an derselben Stelle. Die 2 verliert im unteren Teil — dort, wo eine
# 2 einen waagerechten Balken hätte und die 3 einen zweiten Bogen. Das ist **plausible Semantik**,
# also ein gutes Zeichen.
#
# **Warum bekommen leere Voxel Relevanz?** Das ist die direkte Folge der **b-Regel** in `conv1`
# (Abschnitt 6.3): Dort wird $a_j \equiv 1$ gesetzt, die tatsächliche Voxelintensität also
# ignoriert. Relevanz kann deshalb auf Positionen fließen, die im Input leer sind. Zwei Dinge folgen
# daraus:
#
# 1. **Es ist kein Bug.** Es ist genau der Zweck der Regel — die Karte soll räumlich zusammenhängend
#    sein statt aus Einzelvoxeln zu bestehen.
# 2. **Es ist trotzdem interessant.** „Hier ist nichts, und genau das spricht für eine 3" ist eine
#    legitime Aussage: Der leere Raum *neben* der Ziffer ist Teil ihrer Form. Man darf solche
#    Bereiche nur nicht als „das Netz schaut auf Rauschen" fehldeuten.
#
# ### 7.5 ⚠️ Was diese Abbildung *nicht* zeigt
#
# Weil jede Klasse einzeln normiert wurde, sehen ID 3 (Logit +10,66) und ID 5 (Logit −1,64)
# **gleich kräftig** aus, obwohl die Evidenz für die 5 sechsmal schwächer ist. Wer die absoluten
# Stärken vergleichen will, müsste alle zehn Karten auf **dasselbe** globale Maximum normieren —
# dann wären acht der zehn Blöcke fast weiß.
#
# ### Einordnung
#
# Genau so liest man auch eine LRP-Karte eines MRT-Volumens: schichtweise durchblättern, die
# Struktur mit der Anatomie abgleichen, und immer im Kopf behalten, **welche** Klasse gerade erklärt
# wird. Der Unterschied ist nur, dass man beim Gehirn keine „richtige Antwort" hat, gegen die man
# die Karte prüfen kann — hier schon.

# %%
import matplotlib.pyplot as plt
import numpy as np

num_explanations = len(explanations)

# 1. Erstelle EINE große Figur vor der Schleife
# Zeilen = 2 * Anzahl der Erklärungen, Spalten = 16
fig, ax = plt.subplots(2 * num_explanations, 16, figsize=(30, 5 * num_explanations))

for i in range(num_explanations):
    explanations[i] = explanations[i] / np.amax(np.abs(explanations[i]))
    explanations[i] -= explanations[i, 0, 0, 0, 0]
    
    # Berechne die korrekten Zeilen-Indizes für diese Erklärung
    row_orig = 2 * i
    row_expl = 2 * i + 1
    
    # Optional: Ein Titel pro Block links am Rand (da suptitle sonst alles überschreibt)
    ax[row_orig][0].set_ylabel(f'ID {i} ({round(predictions[0,i], 2)})', 
                               fontsize=16, rotation=0, labelpad=40, va='center')
    
    for j in range(16):
        # Originalbilder in die obere Zeile des Blocks
        ax[row_orig][j].imshow(train_X[image_idx, :, :, j])
        ax[row_orig][j].set_xticks([])
        ax[row_orig][j].set_yticks([])
        
        # Erklärungen in die untere Zeile des Blocks
        ax[row_expl][j].imshow(explanations[i, :, :, j], cmap='seismic', clim=(-1, 1))
        ax[row_expl][j].set_xticks([])
        ax[row_expl][j].set_yticks([])
        
        # Rahmen ausschalten (außer für das erste Bild wegen des Labels links)
        if j > 0:
            ax[row_orig][j].axis('off')
        ax[row_expl][j].axis('off')

# 2. Layout optimieren, damit sich die Zeilen nicht überschneiden
plt.tight_layout()

# 3. Als eine einzige, große Datei speichern
fig.savefig(target_dir / "all_slices_combined.png", bbox_inches='tight', dpi=150)

# 4. Am Ende einmal anzeigen und Speicher freigeben
plt.show()
plt.close(fig)

# %% [markdown]
# <a id="sec-08"></a>
# ## 8. Die Differenzmatrix: wie klassenspezifisch sind die Erklärungen?
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### 8.1 Was passiert hier?
#
# Ein 10×10-Raster. In Zeile $i$, Spalte $j$ steht die **Differenz** zweier Relevanzkarten, jeweils
# auf **Schnittebene 5** (mitten durch die Ziffer):
#
# $$D_{ij} \;=\; \tilde{R}^{(i)}_{:,:,5} \;-\; \tilde{R}^{(j)}_{:,:,5}$$
#
# Verwendet werden dabei die **bereits normierten** Karten aus Abschnitt 7 — diese Zelle setzt also
# voraus, dass die vorherige gelaufen ist. Dieselbe Farbskala: rot = $D_{ij} > 0$ (Klasse $i$
# bewertet dieses Voxel positiver), blau = $D_{ij} < 0$, weiß = beide gleich.
#
# **Die Idee dahinter** ist die schärfere Frage: Nicht *„welche Voxel sprechen für die 3?"*, sondern
# *„welche Voxel unterscheiden eine 3 von einer 8?"*. Das ist der Übergang von einer **absoluten**
# zu einer **kontrastiven** Erklärung — in der XAI-Literatur die deutlich informativere Variante,
# weil sie der Frage entspricht, die Menschen tatsächlich stellen („warum das und nicht jenes?").
#
# ### 8.2 Was man auf der Abbildung sieht
#
# **Die Diagonale ist weiß.** $D_{ii} = 0$ — eine Karte minus sich selbst. Das ist die
# Selbstkontrolle: Wäre die Diagonale nicht weiß, wäre der Code fehlerhaft.
#
# **Die Matrix ist antisymmetrisch.** Feld $(i,j)$ ist das exakte Farbnegativ von $(j,i)$, denn
# $D_{ij} = -D_{ji}$.
#
# **Die Zeilen 2 und 3 leuchten kräftig rot**, die **Spalten 2 und 3 kräftig blau**. Alle übrigen
# Felder sind blass. Innerhalb des Blocks „2 gegen 3" (Felder $(2,3)$ und $(3,2)$) ist es fast weiß.
#
# ### 8.3 Interpretation — der unbequeme Befund
#
# Die Matrix zerfällt praktisch in **zwei Gruppen**: {2, 3} gegen den Rest. Innerhalb der Gruppen
# passiert kaum etwas.
#
# Rechnet man die Korrelation zwischen je zwei der zehn normierten Relevanzvolumen aus, bestätigt
# sich das drastisch: **Der Betrag der Korrelation liegt bei allen 45 Paaren zwischen 0,82 und
# 1,00.** Die Karten für 2 und 3 korrelieren mit $r \approx +1{,}00$ miteinander und mit
# $r \approx -1{,}00$ mit allen anderen.
#
# > **Es gibt hier nicht zehn Erklärungen, sondern im Wesentlichen eine — einmal mit Plus, einmal
# > mit Minus davor.**
#
# Das ist ein **negatives Ergebnis**, und genau deshalb lehrreich. Drei Ursachen kommen zusammen:
#
# 1. **Das Netz ist zu sicher.** Bei 99,876 % für die 3 sind alle anderen Logits nur noch
#    verschieden stark negativ. Es gibt schlicht keine differenzierte Contra-Evidenz mehr — das
#    Netz „zweifelt" nirgends.
# 2. **Die αβ-Regel ist strukturkonservativ.** Sie verteilt Relevanz entlang der stärksten
#    Aktivierungspfade. Die sind für alle Klassen weitgehend dieselben, weil sich alle zehn
#    Ausgabeneuronen dieselben 256 Merkmale aus `pool4` teilen.
# 3. **`GlobalAveragePooling3D` verwischt den Ort.** Nach dem Mittelwert über alle 8 Positionen
#    enthält der 256er-Vektor kaum noch Ortsinformation. Die klassenspezifische Entscheidung fällt
#    in `dense`/`preds` — also **hinter** dem Punkt, an dem Ortsinformation existiert. Was danach
#    rückwärts fließt, verteilt sich fast unvermeidlich auf dieselben Voxel.
#
# ### 8.4 ⚠️ Grenzen dieser Darstellung
#
# * **Nur eine von 16 Schichten.** `[:,:,5]` zeigt die gefüllte Endfläche der Ziffer. Andere Schnitte
#   könnten anders aussehen; die Zelle prüft das nicht.
# * **Differenz normierter Größen.** Weil jede Karte auf ihr eigenes Maximum skaliert wurde, mischt
#   $D_{ij}$ Unterschiede in der *Form* mit Unterschieden im *Maßstab*. Ein Teil des Rot in Zeile 3
#   ist also nur Normierungseffekt.
# * **Keine Achsenbeschriftung.** Man muss wissen, dass Zeile/Spalte $i$ der Ziffer $i$ entspricht.
#
# ### Einordnung
#
# Der Befund „alle Klassen erklären sich gleich" ist ein bekanntes Problem und der Grund, warum
# **Sanity Checks** zum Standardrepertoire gehören. Der berühmteste ist der *Model Parameter
# Randomization Test* (Adebayo et al., 2018): Man randomisiert die Gewichte schichtweise und
# schaut, ob sich die Heatmap ändert. Tut sie es nicht, misst die Methode Bildstruktur statt
# Modellverhalten.
#
# Was man hier praktisch tun würde:
#
# * ein **unsicheres** Beispiel wählen (etwa eines mit 60 : 40 zwischen zwei Ziffern) — dort
#   werden die Unterschiede sichtbar;
# * ein Netz **ohne** GlobalAveragePooling gegentesten;
# * die Differenz der **Roh**relevanzen statt der normierten betrachten.

# %%
fig, ax = plt.subplots(10, 10, figsize=(20, 20))

for i in range(10):
    for j in range(10):
        ax[i][j].axis('off')
        ax[i][j].imshow(explanations[i,:,:,5] - explanations[j,:,:,5], cmap='seismic', clim=(-1, 1))
        
plt.show()

# %% [markdown]
# <a id="sec-09"></a>
# ## 9. Das Volumen als 3D-Punktwolke
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Bis hierher haben wir das Volumen nur **scheibenweise** betrachtet. Ab jetzt schauen wir es als
# Ganzes an — und zwar interaktiv drehbar.
#
# ### 9.1 Was passiert hier?
#
# Die Funktion `plot_digit_3d` macht drei Dinge:
#
# **1. Kanaldimension entfernen.** `vol[..., 0]` macht aus `(16,16,16,1)` ein `(16,16,16)`.
#
# **2. Belegte Voxel finden.**
#
# ```python
# z, y, x = np.where(vol > threshold)
# ```
#
# `np.where` liefert die **Koordinaten** aller Voxel über dem Schwellwert 0,1 — als drei parallele
# Index-Arrays. Bei unserem Beispiel sind das **615 von 4096** Voxeln (15 %).
#
# Das ist der entscheidende Trick: Statt 4096 Würfel zu rendern (von denen 85 % unsichtbar leer
# wären), zeichnet man nur die 615 belegten als Punkte. Diese Umwandlung von einer **dichten
# Gitterdarstellung** in eine **dünne Punktliste** ist der Standardweg, 3D-Daten im Browser
# darstellbar zu machen.
#
# **3. Als `Scatter3d` zeichnen.** Jeder Punkt bekommt seine Voxelintensität als Farbe
# (`colorscale="Viridis"`), `opacity=0.85` macht ihn leicht durchscheinend, damit man auch das
# Innere sieht. `aspectmode="cube"` erzwingt gleiche Achsenskalierung — ohne das würde Plotly die
# Achsen an das Fenster anpassen und die Ziffer verzerren.
#
# ### 9.2 Was man auf der Abbildung sehen sollte
#
# Eine **frei drehbare Punktwolke** in einem 16×16×16-Würfel: die Ziffer **3**, als flache Schale in
# den Raum extrudiert und schräg im Würfel liegend. Genau die Struktur, die in Abschnitt 7 als sechs
# Schnittbilder zu sehen war — nur eben auf einen Blick. Mit der Maus lässt sie sich in die Lage
# drehen, in der die 3 als Ziffer lesbar wird.
#
# ### 9.3 ⚠️ Hinweis zum HTML-Export
#
# **In der exportierten HTML-Datei fehlt diese Abbildung.** Plotly-Grafiken sind JavaScript, kein
# Bild; beim Export mit `nbconvert` ohne passende Renderer-Einstellung bleibt die Ausgabe leer. Im
# laufenden Notebook erscheint die Grafik normal.
#
# Wer sie im HTML haben will, setzt vor dem Export
#
# ```python
# import plotly.io as pio
# pio.renderers.default = "notebook_connected"   # oder "notebook" für Offline-Einbettung
# ```
#
# Das ist ein typischer Stolperstein bei geteilten Notebooks: Was man selbst sieht, sieht der
# Empfänger des HTML-Exports nicht unbedingt.
#
# ### 9.4 Der Schwellwert als versteckte Entscheidung
#
# `threshold=0.1` blendet alle Voxel unter 10 % Intensität aus. Bei diesem Datensatz ist das
# unkritisch — von 616 Voxeln mit Wert > 0 überleben 615, es fällt also genau eines heraus. Bei
# medizinischen Daten ist die Schwelle dagegen **eine inhaltliche Entscheidung**: Sie bestimmt, was
# als „Gewebe" gilt und was als Hintergrund. Eine schlecht gewählte Schwelle kann eine ganze
# Struktur unsichtbar machen. Solche Parameter gehören dokumentiert.
#
# ### Einordnung
#
# Dieser Abschnitt leitet den letzten Teil des Notebooks ein: Ab jetzt geht es um **Darstellung**,
# nicht mehr um Berechnung. Für die Kommunikation von Ergebnissen — an Kliniker, an Gutachter, an
# das eigene Team — ist eine drehbare 3D-Ansicht oft überzeugender als 16 Schnittbilder, weil sie
# keine räumliche Vorstellungsleistung verlangt.

# %%
import plotly.graph_objects as go

def plot_digit_3d(vol, title="", threshold=0.1):
    vol = np.asarray(vol[..., 0] if vol.ndim == 4 else vol)
    z, y, x = np.where(vol > threshold)
    fig = go.Figure(go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=4, opacity=0.85, color=vol[z, y, x], colorscale="Viridis"),
    ))
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="cube",
                   xaxis=dict(range=[0, 16]), yaxis=dict(range=[0, 16]), zaxis=dict(range=[0, 16])),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.show()

idx = image_idx
plot_digit_3d(train_X[idx], title=f"Label: {np.argmax(train_y[idx])}")

# %% [markdown]
# <a id="sec-10"></a>
# ## 10. Eine 3D-Zahl rein → Vorhersage raus
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Die komplette Inferenz-Pipeline in fünf Zeilen — nützlich als Vorlage zum Kopieren:
#
# | Zeile | Was sie tut |
# |---|---|
# | `sample = train_X[idx:idx+1]` | ein Beispiel als **Batch der Größe 1** ausschneiden, Form `(1,16,16,16,1)` |
# | `probs = model.predict(sample, verbose=0)[0]` | zehn Wahrscheinlichkeiten; `[0]` holt das erste (einzige) Element des Batches |
# | `pred = int(np.argmax(probs))` | Index der größten Wahrscheinlichkeit = vorhergesagte Ziffer |
# | `true = int(np.argmax(train_y[idx]))` | One-Hot-Label zurück in eine Zahl übersetzen |
#
# Der **Slice** `idx:idx+1` statt `[idx]` ist kein Schönheitsfehler, sondern notwendig: `train_X[654]`
# hätte die Form `(16,16,16,1)`, und Keras würde die erste Achse als Batch-Dimension missverstehen.
# `verbose=0` unterdrückt lediglich den Fortschrittsbalken.
#
# ### Ausgabe
#
# ```text
# True: 3 | Pred: 3 | Confidence: 0.9988
# [6.4364016e-08 3.8472465e-15 1.2347297e-03 9.9875998e-01 6.8500914e-15
#  4.5543707e-06 9.7387264e-08 3.6674541e-09 5.4195624e-07 7.8604500e-13]
# ```
#
# ### Interpretation
#
# **Die Vorhersage ist korrekt** — und das ist die Voraussetzung dafür, dass die Erklärungen aus
# Abschnitt 7 überhaupt interpretierbar sind. Bei einer Fehlklassifikation würde die Heatmap
# erklären, *warum das Netz sich irrt* — auch spannend, aber eine ganz andere Frage.
#
# **99,876 % sind sehr viel.** Zum Vergleich: VGG19 gibt einem eindeutigen Katzenfoto nur rund 40 %
# für die Top-Klasse, weil es zwischen einem Dutzend Katzenrassen unterscheiden muss. Hier gibt es
# nur zehn gut getrennte Klassen, deshalb sind solche Werte normal.
#
# Trotzdem eine Warnung: Neuronale Netze sind bekanntermaßen **überkonfident**. 99,876 % heißen
# nicht „in 999 von 1000 Fällen richtig", sondern nur „dieser Ausgabewert war viel größer als die
# anderen". Ein Netz, das mit L2-Regularisierung und Dropout trainiert wurde, ist etwas besser
# kalibriert — verlassen sollte man sich darauf nicht.
#
# ### ⚠️ Ein methodischer Punkt
#
# `idx = image_idx = 654` ist ein **Trainings**beispiel. Das Netz hat es 100-mal gesehen. Die 99,876 %
# sagen daher wenig über die Generalisierungsfähigkeit aus, und auch die Erklärung könnte teilweise
# „auswendig gelernte" Struktur widerspiegeln. Für eine belastbare Analyse würde man
# `test_X`/`test_y` verwenden — die 2.000 Testvolumen liegen bereits geladen bereit.
#
# ### Einordnung
#
# Dieser Abschnitt schließt den Kreis zu Abschnitt 6: Dort wurde dieselbe Vorhersage berechnet, aber
# als Startwert für LRP. Hier steht sie für sich, als das, was ein Anwender vom Modell tatsächlich
# zu sehen bekommt. Die eine Zahl („3, mit 99,9 %") ist alles, was ohne XAI übrig bleibt — die
# nächsten Abschnitte zeigen, was dahintersteckt.

# %%
idx = image_idx
sample = train_X[idx:idx + 1]          # Shape (1, 16, 16, 16, 1)
probs = model.predict(sample, verbose=0)[0]
pred = int(np.argmax(probs))
true = int(np.argmax(train_y[idx]))

print(f"True: {true} | Pred: {pred} | Confidence: {probs[pred]:.4f}")
print(probs)

# %% [markdown]
# <a id="sec-11"></a>
# ## 11. Input und Erklärung nebeneinander in 3D
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Der Höhepunkt des Notebooks: Eingabe und Erklärung **in derselben Geometrie**, nebeneinander,
# gemeinsam drehbar.
#
# ### 11.1 Was passiert hier?
#
# Die Zelle fasst alles Vorherige zusammen:
#
# 1. **Vorhersage** für Beispiel 654 berechnen und die Top-Klasse `pred` bestimmen.
# 2. **Dieselbe Composite-Strategie** wie in Abschnitt 6 aufbauen, hier kompakter geschrieben:
#    `*[{'alpha': 2, 'beta': 1}] * 7` erzeugt sieben identische Einträge — inhaltlich identisch zur
#    ausgeschriebenen Liste oben.
# 3. **Erklärer bauen mit `idx=pred`**. Das ist der wichtige Unterschied zu Abschnitt 6: Dort wurden
#    alle zehn Klassen durchprobiert, hier wird gezielt die **vorhergesagte** Klasse erklärt — der
#    Normalfall in der Praxis.
# 4. **Relevanz normieren:**
#
#    $$\tilde{R} = \frac{R}{\max \lvert R \rvert + 10^{-12}}$$
#
#    Das $10^{-12}$ ist ein Schutz gegen Division durch null, falls die Erklärung komplett leer wäre.
# 5. **Dieselben Punkte zweimal zeichnen.** Entscheidend:
#
#    ```python
#    z, y, x = np.where(vol > 0.1)
#    ```
#
#    wird **einmal** berechnet und für **beide** Teilbilder verwendet. Links bekommen die Punkte die
#    Voxelintensität als Farbe, rechts ihre Relevanz. Es sind also exakt dieselben 615 Punkte an
#    exakt denselben Positionen — nur anders eingefärbt.
#
# ### 11.2 Warum das die beste Darstellung ist
#
# Weil sie die **Zuordnung erzwingt**. Bei den Schnittbildern in Abschnitt 7 muss das Auge zwischen
# zwei Zeilen hin- und herspringen und selbst herausfinden, welcher roter Fleck zu welchem Teil der
# Ziffer gehört. Hier liegt beides deckungsgleich übereinander — man dreht die Ansicht und sieht
# sofort, welcher Teil der Ziffer welche Relevanz trägt.
#
# Der Preis ist ein **blinder Fleck**: Weil nur Punkte mit `vol > 0.1` gezeichnet werden, ist die
# Relevanz auf **leeren** Voxeln unsichtbar. Genau die war in Abschnitt 7.3 aber ein interessanter
# Befund (rund 47 % der Gesamtrelevanz). Diese Darstellung zeigt also nur die Relevanz **auf** der
# Ziffer — sauberer anzusehen, aber unvollständig.
#
# ### 11.3 Die Farbskalen
#
# | Teilbild | Skala | Bedeutung |
# |---|---|---|
# | links, „Input" | `Viridis` | sequentiell, Voxelintensität 0…1 |
# | rechts, „LRP" | `RdBu_r`, `cmin=-1`, `cmax=+1` | divergierend: **rot = pro**, weiß = neutral, **blau = contra** |
#
# `RdBu_r` ist das Plotly-Gegenstück zu matplotlibs `seismic`; das `_r` steht für *reversed* — ohne
# es wäre Rot negativ und Blau positiv, also genau verkehrt herum. Die feste Skalierung
# `cmin=-1, cmax=+1` übernimmt die Rolle von `clim` und sorgt dafür, dass **weiß exakt bei null**
# liegt. Die Colorbar rechts macht die Werte ablesbar — genau der Punkt aus Abschnitt 3.
#
# ### 11.4 Was man sehen sollte
#
# Zwei drehbare Würfel nebeneinander, mit den Titeln `Input (true=3)` und `LRP for class 3 (p=1.00)`.
#
# * **Links:** die Ziffer 3 als grün-gelbe Punktwolke.
# * **Rechts:** dieselbe Wolke, **überwiegend rot** — denn der Logit der Klasse 3 ist mit +10,66
#   positiv, alle Beiträge sprechen also dafür. Die kräftigsten Rottöne liegen entlang der
#   **Kontur** der Ziffer, blasser wird es im Inneren der gefüllten Endflächen.
#
# **Interpretation:** Das Netz stützt seine Entscheidung auf die **Form** der Ziffer, nicht auf ihre
# Masse oder ihre Position im Würfel. Das ist genau das erwünschte Verhalten. Läge das kräftigste
# Rot am Rand des Würfels oder in einer Ecke, wäre das ein Warnsignal für einen
# Clever-Hans-Prädiktor.
#
# ### 11.5 ⚠️ Auch hier: im HTML-Export fehlt die Grafik
#
# Wie in Abschnitt 9.3 — Plotly-Ausgaben brauchen einen aktiven Renderer. Zusätzlich fällt auf, dass
# `probs` und `pred` hier **zum dritten Mal** berechnet werden (nach Abschnitt 6 und 10). Das kostet
# ein paar Millisekunden und macht die Zelle dafür **eigenständig lauffähig** — ein bewusster
# Kompromiss, der bei explorativen Notebooks üblich ist.
#
# ### Einordnung
#
# Genau diese Darstellung ist das Ziel der gesamten Pipeline und lässt sich unverändert auf
# medizinische Volumen übertragen: links das MRT, rechts dieselbe Anatomie eingefärbt danach, was
# das Netz zu seiner Diagnose bewogen hat. Der einzige Unterschied ist, dass man dort die
# Relevanzkarte mit anatomischem Wissen abgleicht statt mit einer Ziffernform — und dass die Frage
# „schaut das Netz auf die richtige Struktur?" dann keine Übung mehr ist, sondern eine
# Zulassungsvoraussetzung.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from explainability import LRP, LRPStrategy

idx = image_idx
sample = train_X[idx:idx + 1]
true_label = int(np.argmax(train_y[idx]))
probs = model.predict(sample, verbose=0)[0]
pred = int(np.argmax(probs))

strategy = LRPStrategy(layers=[
    {'b': True, 'alpha': 1, 'beta': 0},
    *[{'alpha': 2, 'beta': 1}] * 7,
    {'epsilon': 0.25},
    {'epsilon': 0.25},
])
explainer = LRP(model, layer=len(model.layers) - 1, idx=pred, strategy=strategy)
R = np.array(explainer(sample))[0, ..., 0]
R = R / (np.max(np.abs(R)) + 1e-12)

vol = sample[0, ..., 0]
z, y, x = np.where(vol > 0.1)

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=(
        f"Input (true={true_label})",
        f"LRP for class {pred} (p={probs[pred]:.2f})",
    ),
)
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z, mode="markers",
    marker=dict(size=4, opacity=0.85, color=vol[z, y, x], colorscale="Viridis"),
    showlegend=False,
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z, mode="markers",
    marker=dict(
        size=4, opacity=0.9,
        color=R[z, y, x],
        colorscale="RdBu_r",
        cmin=-1, cmax=1,
        showscale=True,
        colorbar=dict(title="Relevance", x=1.02),
    ),
    showlegend=False,
), row=1, col=2)
fig.update_layout(
    title=f"Sample #{idx}",
    scene=dict(aspectmode="cube"),
    scene2=dict(aspectmode="cube"),
    margin=dict(l=0, r=80, t=60, b=0),
)
fig.show()

# %% [markdown]
# <a id="sec-12"></a>
# ## 12. Anhang: Rohdaten-Zelle
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Diese Zelle lädt den Datensatz ein zweites Mal, diesmal in Variablen mit dem Präfix `d`
# (`dtrain_x`, `dtrain_y`, `draw_x`, `dtest_X`, `dtest_y`). Sie erzeugt **keine Ausgabe** und
# beeinflusst **nichts** von dem, was vorher berechnet wurde — es ist eine **Sandbox-Zelle**, in der
# jemand die Rohform der Daten untersuchen wollte.
#
# Der interessante Teil ist `draw_x = f["X_train"][:]`: die Daten in ihrer **Originalform**
# `(10000, 4096)` — also als flache Vektoren, **vor** dem `reshape` zum Würfel und **ohne** die
# Spiegelung aus Abschnitt 1. So liegen sie tatsächlich in der HDF5-Datei.
#
# ### ⚠️ Ein Tippfehler zum Lernen
#
# ```python
# dtrain_x = (f["X_train"][:], (-1, 16, 16, 16, 1))
# ```
#
# Hier fehlt `np.reshape(...)` um die Klammer herum. Das Ergebnis ist deshalb **kein umgeformtes
# Array**, sondern ein **Tupel** aus dem Array und dem Formtupel:
#
# ```python
# type(dtrain_x)     # tuple
# len(dtrain_x)      # 2
# dtrain_x[0].shape  # (10000, 4096)  ← unverändert
# ```
#
# Zum Vergleich die korrekte Zeile eine Ebene tiefer, für die Testdaten:
#
# ```python
# dtest_X = np.reshape(f["X_test"][:], (-1, 16, 16, 16, 1))   # ✓ echtes reshape
# ```
#
# Das ist ein klassischer Python-Stolperstein: Ein Komma erzeugt stillschweigend ein Tupel, es gibt
# **keine Fehlermeldung**. Der Fehler fällt erst auf, wenn man `dtrain_x` weiterverwendet — was hier
# nicht passiert, weshalb er folgenlos bleibt. Merksatz: Wenn ein Array „plötzlich keine `.shape`
# mehr hat", ist es meistens in ein Tupel gerutscht.
#
# ### Einordnung
#
# Solche Zellen entstehen beim explorativen Arbeiten ständig und sind völlig legitim. Für ein
# Notebook, das andere lesen sollen, gilt aber: **aufräumen oder kommentieren**. Ein Notebook ist
# nicht nur Code, sondern ein Dokument — und eine Zelle ohne Zweck und ohne Ausgabe kostet jeden
# Leser Zeit.

# %%
data_path = os.path.join(os.path.expanduser('~/git-repos/keras-explainability'), 'data', '3d-mnist', 'full_dataset_vectors.h5')

assert os.path.isfile(data_path), \
    'Download the 3d-mnist data from https://www.kaggle.com/daavoo/3d-mnist'

with h5py.File(data_path, 'r') as f:
    dtrain_x = (f["X_train"][:], (-1, 16, 16, 16, 1))
    dtrain_y = f["y_train"][:]    
    draw_x = f["X_train"][:]
    #dtrain_x = f["X_train"][:]  
    dtest_X = np.reshape(f["X_test"][:]  , (-1, 16, 16, 16, 1))
    dtest_y = f["y_test"][:]

# %% [markdown]
# <a id="sec-13"></a>
# ## 13. Fazit: was man mitnehmen sollte
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was dieses Notebook gezeigt hat
#
# Wir sind einmal die vollständige Kette gegangen: **Volumendaten laden → 3D-CNN bauen → trainieren
# → Vorhersage → Erklärung → Visualisierung**. Das Ergebnis:
#
# 1. **3D ist kein Sonderfall, sondern der Normalfall.** Für `Conv3D` gilt dieselbe Logik wie für
#    `Conv2D`, nur mit einer Achse mehr. Der einzige praktische Unterschied ist der Rechenaufwand.
# 2. **Ein einzelner Schnitt ist nicht das Volumen.** Die Beispiele 0 und 1 aus Abschnitt 1 zeigen
#    dieselbe Ziffer und sehen völlig verschieden aus. Diese Erkenntnis ist der ganze Grund, warum
#    es 3D-Netze gibt.
# 3. **LRP erklärt immer eine Klasse, nie „die Vorhersage".** Zehn Zielklassen ergeben zehn
#    Erklärungen. Rot heißt „spricht für die erklärte Klasse", blau „spricht dagegen" — und das
#    Vorzeichen der gesamten Karte folgt dem Vorzeichen des Logits.
# 4. **Die Regelwahl ist Teil des Ergebnisses.** Die Composite-Strategie (b-Regel unten, αβ in der
#    Mitte, ε oben) ist eine bewusste Entscheidung und gehört genauso dokumentiert wie die
#    Architektur.
# 5. **Die Erhaltung ist eine Näherung.** 74–100 % des Logits kommen am Eingang an; Bias-Terme und
#    die ε-Regel schlucken den Rest. Als Plausibilitätsprüfung taugt die Summe trotzdem.
#
# ### Der wichtigste Befund ist ein negativer
#
# Abschnitt 8 hat gezeigt: **Die zehn Erklärungen sind im Wesentlichen eine einzige Karte mit zwei
# Vorzeichen** (Korrelationsbetrag 0,82–1,00 zwischen allen Paaren). Die Heatmap zeigt hier vor
# allem, *wo die Ziffer ist* — nicht, *was eine 3 von einer 8 unterscheidet*.
#
# Drei Ursachen wirken zusammen: ein extrem sicheres Netz (99,876 %), die strukturkonservative
# αβ-Regel, und `GlobalAveragePooling3D`, das die Ortsinformation vor der eigentlichen
# Klassenentscheidung wegmittelt.
#
# Das ist keine Schwäche dieses Notebooks, sondern **die Lektion**: XAI-Verfahren liefern
# **immer** ein buntes Bild — auch wenn es nichts Klassenspezifisches enthält. Ohne quantitative
# Gegenprüfung merkt man das nicht.
#
# ### Checkliste für eigene Analysen
#
# | ✓ | Prüfung | Warum |
# |---|---|---|
# | ☐ | Modellgüte auf dem **Testsatz** messen, bevor eine Heatmap interpretiert wird | eine Erklärung ist nur so gut wie das Modell |
# | ☐ | **Testdaten** statt Trainingsdaten erklären | sonst erklärt man auswendig Gelerntes |
# | ☐ | Vorhersage **und Logit-Vorzeichen** vor dem Deuten ansehen | entscheidet, ob rot oder blau zu erwarten ist |
# | ☐ | Regelwahl **vorher** festlegen, nicht die schönste im Nachhinein auswählen | sonst ist die Heatmap ein Suchergebnis, kein Befund |
# | ☐ | Bei divergierenden Skalen immer `clim` / `cmin`+`cmax` symmetrisch setzen | nur so bedeutet weiß wirklich null |
# | ☐ | Klassenspezifität **quantitativ** prüfen (Differenzen, Korrelationen) | siehe Abschnitt 8 |
# | ☐ | Ein **unsicheres** Beispiel mit erklären | dort werden die Unterschiede erst sichtbar |
# | ☐ | Sanity Check: Gewichte randomisieren, Heatmap neu berechnen | ändert sie sich nicht, misst man Bildstruktur statt Modellverhalten |
#
# ### Naheliegende nächste Schritte
#
# * **`model.evaluate(test_X, test_y)`** ergänzen — die eine fehlende Zahl im ganzen Notebook.
# * Ein **falsch klassifiziertes** Beispiel suchen und erklären. Dort ist LRP am nützlichsten.
# * Ein Beispiel mit **unsicherer Vorhersage** (z. B. 60 : 40) durch Abschnitt 7 und 8 schicken.
# * Andere **Regelkombinationen** durchprobieren — die Systematik dazu steht in
#   [`Explain_2D_VGG_predictions`](Explain_2D_VGG_predictions.py).
# * Denselben Ablauf auf ein echtes Volumen anwenden:
#   [`Explain_brain_age_predictions`](Explain_brain_age_predictions.py).
#
# ### Weiterführend
#
# * [`doc/LRP_Fraunhofer_HHI_Video_Zusammenfassung.md`](../../doc/LRP_Fraunhofer_HHI_Video_Zusammenfassung.md)
#   — LRP-Theorie in Worten
# * Bach et al. (2015): *On Pixel-Wise Explanations…* — die Originalarbeit zu LRP
# * Montavon et al. (2019): *Layer-Wise Relevance Propagation: An Overview* — die Composite-Strategie
# * Adebayo et al. (2018): *Sanity Checks for Saliency Maps* — warum man XAI-Methoden misstrauen muss
#
# [↑ Zum Anfang](#top)
