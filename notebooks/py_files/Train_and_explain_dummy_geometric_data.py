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
# <a id="toc"></a>
# # Ein 3D-CNN trainieren und seine Entscheidungen erklären
#
# **Ein Einstieg in Deep Learning und Explainable AI (XAI) am Spielzeug-Beispiel**
#
# ## Worum geht es in diesem Notebook?
#
# Neuronale Netze sind hervorragende Mustererkenner, aber sie sind zunächst
# *Blackboxes*: Wir sehen die Vorhersage, nicht aber die Begründung. In
# sicherheitskritischen Anwendungen — allen voran der medizinischen Bildgebung —
# ist das ein Problem. Ein Netz, das Krankheiten auf MRT-Aufnahmen erkennt, mag
# 95 % Genauigkeit erreichen und trotzdem völlig unbrauchbar sein, weil es
# heimlich auf ein Artefakt des Scanners oder eine Textmarkierung im Bildrand
# achtet statt auf die Pathologie. Diesen Effekt nennt man
# *Clever-Hans-Verhalten* (Lapuschkin et al., 2019): Das Modell gibt die
# richtige Antwort aus dem falschen Grund.
#
# Das Forschungsfeld, das solche Begründungen sichtbar macht, heißt
# **Explainable AI (XAI)**. Dieses Notebook demonstriert eine der etablierten
# XAI-Methoden — **Layer-wise Relevance Propagation (LRP)** — in einem
# absichtlich winzigen, vollständig kontrollierten Szenario:
#
# 1. Wir **erzeugen** einen synthetischen 3D-Datensatz aus drei Klassen
#    (Würfel, Kugel, Rauschen). Weil wir die Daten selbst gebaut haben, wissen
#    wir *genau*, welche Bildbereiche relevant sein *sollten*.
# 2. Wir **trainieren** ein kleines 3D-Convolutional-Neural-Network (CNN) darauf.
# 3. Wir **erklären** die Vorhersagen mit LRP und prüfen, ob die Erklärungen mit
#    unserer Erwartung übereinstimmen.
# 4. Wir stellen dem Modell **Hybrid-Objekte** vor (halb Würfel, halb Kugel), die
#    es im Training nie gesehen hat, und beobachten, wie sich die Erklärungen
#    verändern.
#
# ## Warum ein Spielzeug-Datensatz?
#
# Das mag akademisch wirken, ist aber methodisch zentral. Für echte
# medizinische Bilder gibt es keine *Ground Truth für Erklärungen*: Niemand
# kann sagen, welches Voxel eine korrekte Erklärung hervorheben müsste. Bei
# einem Würfel, den wir selbst in ein leeres Volumen gezeichnet haben, wissen
# wir es. Synthetische Daten sind deshalb der Prüfstand, auf dem man eine
# XAI-Methode *validiert*, bevor man ihr auf echten Daten glaubt. Genau diese
# Rolle hat dieses Notebook im Repository: Es ist die Sanity-Check-Stufe vor den
# Notebooks zu MRT-Hirnalter-Vorhersage und VGG-Bildklassifikation.
#
# ## Voraussetzungen
#
# Grundkenntnisse in Python und NumPy. Deep-Learning-Vorwissen ist nicht nötig —
# die Konzepte (Convolution, Pooling, Softmax, Cross-Entropy) werden an der
# Stelle erklärt, an der sie zum ersten Mal auftauchen.
#
# ---
#
# ## Inhaltsverzeichnis
#
# | Abschnitt | Thema |
# |---|---|
# | [1. Setup und Reproduzierbarkeit](#kapitel-1) | Pfade, Imports, Zufallssaat |
# | [2. Den synthetischen 3D-Datensatz erzeugen](#kapitel-2) | Würfel, Kugeln und Rauschen in Voxelgittern |
# | [2.1 Interpretation: Übersicht über die Stichproben](#kapitel-2-1) | Was die Schnittbilder zeigen |
# | [2.2 Vorverarbeitung: One-Hot-Kodierung und Datensplit](#kapitel-2-2) | Labels als Vektoren, Trainings-/Testmenge |
# | [3. Das 3D-CNN aufbauen und trainieren](#kapitel-3) | Faltung, Normalisierung, Pooling, Klassifikation |
# | [3.1 Interpretation: Was beim Training passiert ist](#kapitel-3-1) | Modell laden statt neu trainieren |
# | [4. Die Schichten des Modells zählen](#kapitel-4) | Warum diese Zahl für LRP wichtig ist |
# | [5. Layer-wise Relevance Propagation (LRP)](#kapitel-5) | Die Theorie hinter den Heatmaps |
# | [5.1 Die LRP-Strategie dieses Notebooks](#kapitel-5-1) | Welche Regel auf welche Schicht wirkt |
# | [6. Die Reihenfolge der Klassen](#kapitel-6) | Eine kleine, aber fehleranfällige Stelle |
# | [7. Erklärungen für echte Stichproben](#kapitel-7) | Ein Erklärungsbild pro Klasse |
# | [7.1 Interpretation: Die Heatmaps der Stichproben](#kapitel-7-1) | Randartefakte, Rauschen, Signal |
# | [8. Das Gegenprobe-Experiment: Hybrid-Objekte](#kapitel-8) | Halb Würfel, halb Kugel |
# | [8.1 Interpretation: Vorhersagen und Erklärungen der Hybride](#kapitel-8-1) | Der Würfel-Bias des Modells |
# | [9. Alles in einer Abbildung](#kapitel-9) | Dieselbe Analyse als speicherbare Gesamtgrafik |
# | [9.1 Interpretation: Die Gesamtabbildung](#kapitel-9-1) | Muster über alle sechs Hybride |
# | [10. Interaktive 3D-Darstellung](#kapitel-10) | Von Schnittbildern zur Punktwolke |
# | [11. Fazit und weiterführende Schritte](#kapitel-11) | Was wir gelernt haben, was zu verbessern wäre |

# %% [markdown]
# <a id="kapitel-1"></a>
# # 1. Setup und Reproduzierbarkeit
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ## Was diese Zelle berechnet
#
# Diese Zelle rechnet noch nichts Inhaltliches, sondern richtet die
# Arbeitsumgebung ein. Vier Dinge passieren:
#
# **1. Das Repository-Wurzelverzeichnis finden.** `find_repo_root()` läuft vom
# aktuellen Arbeitsverzeichnis aus schrittweise nach oben (`p.parents`), bis es
# einen Ordner findet, der eine `pyproject.toml` oder einen `explainability/`-Ordner
# enthält. Dieser Pfad wird dann an `sys.path` gehängt. Ohne diesen Schritt
# könnte Python die lokale Bibliothek `explainability` (die LRP-Implementierung
# dieses Repos) nicht importieren, denn Notebooks starten je nach Umgebung in
# unterschiedlichen Verzeichnissen. Das ist ein typisches Muster in
# Forschungscode und macht das Notebook unabhängig davon, von wo aus es
# gestartet wird.
#
# **2. Bibliotheken importieren.** `numpy` für Zahlen-Arrays, `matplotlib` für
# Grafiken, aus `scikit-learn` die Funktion `euclidean_distances` (zum Zeichnen
# der Kugeln) und `OneHotEncoder` (zur Label-Kodierung).
#
# **3. Einen Ausgabeordner anlegen.** `find_notebook_name()` ermittelt den Namen
# des laufenden Notebooks — bewusst ohne fest verdrahteten Namen, damit ein
# Umbenennen des Notebooks automatisch auch den Ausgabeordner umbenennt. Nötig
# sind dafür mehrere Quellen, weil keine einzelne in allen Umgebungen
# funktioniert: `ipynbname.name()` fragt den Jupyter-Server und deckt den
# interaktiven Fall (JupyterLab, VS Code) ab, scheitert aber unter `nbconvert`,
# weil dort kein Server läuft. Dann greifen der Reihe nach die
# Kernel-Umgebungsvariable `JPY_SESSION_NAME`, `__file__` (falls die
# `.py`-Fassung direkt als Skript läuft) und schließlich die Kommandozeile des
# aufrufenden Prozesses, in der `nbconvert` bzw. Quarto den Notebook-Pfad
# stehen hat. Findet keine Quelle etwas, bricht die Zelle mit einer
# Fehlermeldung ab, statt stillschweigend in einen falschen Ordner zu
# schreiben; mit der Umgebungsvariablen `NOTEBOOK_NAME` kann man den Namen im
# Notfall vorgeben. Alle Grafiken und das trainierte Modell landen anschließend
# unter `output/notebooks/<Notebook-Name>/`.
#
# **4. Die Zufallssaat setzen.** `np.random.seed(42)` ist der wichtigste Aufruf
# dieser Zelle. Der Datensatz wird gleich zufällig generiert; ohne feste Saat
# bekäme man bei jedem Ausführen andere Würfel und Kugeln, und alle Ergebnisse
# wären nicht mehr vergleichbar. **Reproduzierbarkeit** ist in der
# Machine-Learning-Forschung keine Kür, sondern Pflicht — ein Ergebnis, das man
# nicht wiederherstellen kann, ist kein Ergebnis.
#
# > **Hinweis:** Eine feste NumPy-Saat macht die *Datengenerierung*
# > deterministisch, nicht aber das gesamte Training. TensorFlow hat einen
# > eigenen Zufallsgenerator (Gewichtsinitialisierung, Dropout,
# > Batch-Reihenfolge), und GPU-Kernel liefern nicht immer bit-identische
# > Ergebnisse. Für volle Determinismus bräuchte man zusätzlich
# > `tf.keras.utils.set_random_seed()` und `tf.config.experimental.enable_op_determinism()`.

# %%
import os
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

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple

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


# 2. Notebook-Namen aus der Laufzeitumgebung ermitteln
notebook_name = os.environ.get("NOTEBOOK_NAME") or find_notebook_name()

# 3. Pfad zusammensetzen: (root-dir-des-repos) / output/notebooks / notebook_name
target_dir = repo_root / "output/notebooks" / notebook_name

# 4. Ordner erstellen
target_dir.mkdir(parents=True, exist_ok=True)

print(f"Notebook-Name ist: {notebook_name}")
print(f"Zielordner ist: {target_dir}")

# für wiederholung
np.random.seed(42)


# %% [markdown]
# <a id="kapitel-2"></a>
# # 2. Den synthetischen 3D-Datensatz erzeugen
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ## Der größere Kontext: Was ist ein 3D-Bild?
#
# Ein gewöhnliches Graustufenbild ist eine Matrix aus **Pixeln** — eine
# zweidimensionale Anordnung von Helligkeitswerten. Medizinische Bildgebung wie
# MRT oder CT liefert stattdessen **Volumendaten**: ein dreidimensionales Gitter
# aus **Voxeln** (*volumetric pixels*). Jedes Voxel hat eine Position
# $(i, j, k)$ und einen Intensitätswert.
#
# In diesem Notebook arbeiten wir mit Volumen der Größe $16 \times 16 \times 16 = 4096$
# Voxeln. Das ist winzig — ein echtes MRT hat leicht $256^3 \approx 16.8$ Millionen
# Voxel — aber es genügt vollkommen, um die Prinzipien zu zeigen, und es läuft
# auch ohne GPU in Sekunden.
#
# ## Was diese Zelle berechnet
#
# Drei Generatorfunktionen erzeugen je eine Klasse von Volumen:
#
# **`generate_square`** — zeichnet einen achsenparallelen **Würfel** (der Name
# „square“ ist historisch bedingt; geometrisch ist es ein Würfel). Der Ablauf:
#
# - Eine zufällige Ecke $c$ wird gewählt, wobei $c = \lfloor 16/4 \rfloor + u$ mit
#   $u \sim \mathcal{U}\{0,\dots,7\}^3$, die Ecke liegt also irgendwo zwischen
#   Index 4 und 11.
# - Eine zufällige Kantenlänge $s \sim \mathcal{U}\{5,\dots,10\}$ (aus
#   `randint(16/3, 32/3)`).
# - `np.meshgrid` erzeugt alle Indexkombinationen des Würfels, die Ecke wird
#   addiert, und `idx[maxes < shape[0]]` verwirft alle Indizes, die aus dem
#   Volumen herausragen. Diese Zeile ist wichtiger, als sie aussieht: Sie
#   bedeutet, dass ein Würfel am Rand **abgeschnitten** wird — was gleich in den
#   Grafiken sichtbar wird.
# - Die gewählten Voxel werden auf 1 gesetzt, alles andere bleibt 0.
#
# **`generate_circle`** — zeichnet eine **Kugel** (auch hier ist der Name
# zweidimensional gedacht). Ein zufälliges Zentrum $z \sim \mathcal{U}\{0,\dots,15\}^3$
# und ein zufälliger Radius $r \sim \mathcal{U}\{4,\dots,7\}$ werden gezogen, dann
# wird für *jedes* Voxel $p$ des Gitters die euklidische Distanz
#
# $$d(p, z) = \sqrt{(p_1-z_1)^2 + (p_2-z_2)^2 + (p_3-z_3)^2}$$
#
# berechnet und das Voxel gesetzt, falls $d(p,z) \le r$. Weil das Zentrum
# überall liegen darf — auch direkt am Rand — sind viele Kugeln nur als Segment
# sichtbar.
#
# **`generate_noise`** — zieht jedes Voxel unabhängig aus einer stetigen
# Gleichverteilung, $v \sim \mathcal{U}(0,1)$. Diese Klasse ist der
# **Negativ-Fall**: Sie hat keine geometrische Struktur, nur Textur.
#
# Anschließend werden $n = 200$ Volumen pro Klasse erzeugt, zu einem Array
# $X$ mit 600 Elementen zusammengefügt, die Labels $y$ als Strings angelegt und
# beides mit derselben Permutation `idx` **gemeinsam durchmischt**. Das
# gleichzeitige Mischen von $X$ und $y$ ist essenziell — würde man nur eines von
# beiden mischen, wären alle Labels falsch zugeordnet.
#
# `X` wird auf die Form `(600, 16, 16, 16, 1)` umgeformt. Diese fünf Dimensionen
# bedeuten `(Stichproben, Tiefe, Höhe, Breite, Kanäle)`. Die letzte 1 ist der
# **Kanal**: Bei Farbbildern stünde hier 3 (Rot, Grün, Blau), bei uns gibt es nur
# einen Intensitätskanal. Keras erwartet diese Dimension immer, auch wenn sie
# die Länge 1 hat.
#
# ## Ein wichtiger Unterschied zwischen den Klassen
#
# Beachten Sie die Wertebereiche: Würfel und Kugeln sind **binär** (0 oder 1),
# Rauschen ist **kontinuierlich** (jeder Wert in $[0,1)$). Der mittlere
# Voxelwert unterscheidet sich damit systematisch zwischen den Klassen: Rauschen
# hat einen Erwartungswert von 0.5 über das ganze Volumen, Würfel und Kugeln
# füllen nur einen Bruchteil des Volumens. Das Modell könnte diese Klasse also
# theoretisch schon an der Gesamtintensität erkennen, ohne etwas über Form zu
# lernen — ein sogenannter **Shortcut**. Genau solche Abkürzungen aufzuspüren,
# ist die Aufgabe von XAI, und wir werden in Abschnitt 8 sehen, dass das Modell
# `noise` tatsächlich verdächtig sicher erkennt.

# %%
def generate_square(shape: Tuple[int] = (16, 16, 16)) -> np.ndarray:
    img = np.zeros(shape)
    corner = np.asarray(shape) // 4 + np.random.randint(0, shape[0] // 2, 3)
    side = np.random.randint(shape[1] / 3, shape[1] * 2 / 3)
    
    idx = np.asarray(np.meshgrid(*[np.arange(side) for _ in range(3)])).T.reshape(-1, 3)
    idx += corner
    maxes = np.amax(idx, axis=-1)
    idx = idx[maxes < shape[0]]
    
    img[tuple(idx.T)] = 1
    
    return img

def generate_circle(shape: Tuple[int] = (16, 16, 16)) -> np.ndarray:
    img = np.zeros(shape)
    center = np.random.randint(0, shape[0], 3)
    radius = np.random.randint(shape[0] // 4, shape[0] // 2)

    idx = np.asarray(np.meshgrid(*[np.arange(x) for x in shape])).T.reshape(-1, 3)
    distances = euclidean_distances(idx, center.reshape(1, -1))[:,0]
    inside = distances <= radius
    
    img[tuple(idx[inside].T)] = 1
    
    return img

def generate_noise(shape: Tuple[int] = (16, 16, 16)) -> np.ndarray:
    return np.random.uniform(0, 1, shape)

n = 200
shape = 16
squares = np.asarray([generate_square(shape=(shape, shape, shape)) for _ in range(n)])
circles = np.asarray([generate_circle(shape=(shape, shape, shape)) for _ in range(n)])
noise = np.asarray([generate_noise(shape=(shape, shape, shape)) for _ in range(n)])
X = np.concatenate([squares, circles, noise], axis=0)
y = np.asarray((['square'] * n) + (['circle'] * n) + (['noise'] * n))
idx = np.random.permutation(np.arange(len(X)))
X = np.reshape(X, (-1, shape, shape, shape, 1))
X = X[idx]
y = y[idx]

print(f'X.shape: {X.shape}')
print(f'y.shape: {y.shape}')

n = 10
fig, ax = plt.subplots(n, shape, figsize=(15, 2 * n))

for i in range(n):
    ax[i][shape // 2].set_title(str(y[i]))
    for j in range(shape):
        ax[i][j].imshow(X[i, j], cmap='Greys_r')
        ax[i][j].axis('off')

fig.savefig(target_dir / '1_geometric_samples_overview.png', bbox_inches='tight', dpi=150)
plt.show()
plt.close(fig)

encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

train_X = X[:300]
train_y = y[:300]
test_X = X[:300]
test_y = y[:300]

# %% [markdown]
# <a id="kapitel-2-1"></a>
# ## 2.1 Interpretation: Übersicht über die Stichproben
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Wie diese Abbildung zu lesen ist
#
# Ein 3D-Volumen kann man auf einem flachen Bildschirm nicht direkt zeigen. Die
# übliche Darstellung — dieselbe, die auch Radiologen am Befundungsmonitor
# nutzen — ist die **Schnittbildreihe**: Man schneidet das Volumen entlang einer
# Achse in Scheiben und legt diese nebeneinander.
#
# Die Abbildung hat also folgende Struktur:
#
# - **Jede Zeile** ist *ein* Trainingsbeispiel, mit dem Klassennamen als
#   Überschrift über der Mitte.
# - **Jede der 16 Spalten** ist eine Schnittebene $z = 0, 1, \dots, 15$ durch
#   dieses eine Volumen. Man liest eine Zeile also wie ein Daumenkino, das sich
#   durch das Objekt hindurchbewegt.
# - Weiß bedeutet Voxelwert 1, schwarz bedeutet 0 (Colormap `Greys_r`, das `_r`
#   steht für *reversed*).
#
# ### Was man konkret sieht
#
# **Kugeln (`circle`)** erscheinen als Kreisscheiben, die über die Schnitte
# hinweg **wachsen und wieder schrumpfen**. Das ist genau die Signatur einer
# Kugel: Schneidet man eine Kugel mit Radius $r$ in der Höhe $h$ über dem
# Zentrum, hat der Querschnitt den Radius $\sqrt{r^2 - h^2}$ — maximal in der
# Mitte, null an den Polen. In der ersten Zeile sieht man das mustergültig: Bei
# Schnitt 8 ein einzelner heller Punkt, bei Schnitt 12 die größte Scheibe, danach
# wieder kleiner.
#
# **Würfel (`square`)** erscheinen dagegen als Quadrate **konstanter Größe**, die
# über mehrere aufeinanderfolgende Schnitte unverändert stehen und dann abrupt
# verschwinden. Auch das ist geometrisch zwingend: Ein achsenparalleler
# Querschnitt durch einen Würfel ist immer dasselbe Quadrat.
#
# **Rauschen (`noise`)** ist in jedem einzelnen Schnitt körnig und sieht in allen
# 16 Schnitten gleich aus. Es gibt keine räumliche Kohärenz — der Wert eines
# Voxels sagt nichts über seine Nachbarn.
#
# ### Die wichtigste Beobachtung: abgeschnittene Objekte
#
# Mehrere Zeilen zeigen keine saubere Geometrie, sondern **Fragmente**. In der
# dritten Zeile (`circle`) ist nur ein Kugelsegment in der oberen linken Ecke zu
# sehen, in der neunten Zeile (`square`) ergibt der Würfel eine „C“- bzw.
# L-Form. Der Grund ist die Randbehandlung in den Generatorfunktionen: Zentrum
# bzw. Ecke werden zufällig im gesamten Volumen platziert, und alles, was über
# den Rand hinausragt, wird verworfen.
#
# Das hat eine echte Konsequenz für die Aufgabe:
#
# > Ein Teil der Stichproben trägt ein Label (`circle` oder `square`), obwohl das
# > sichtbare Objekt kaum noch Kugel- oder Würfelcharakter hat. Man spricht von
# > **Label-Rauschen**. Es setzt eine Obergrenze für die erreichbare Genauigkeit,
# > die *unterhalb* von 100 % liegt — kein Modell kann Information rekonstruieren,
# > die in den Daten nicht mehr vorhanden ist. Halten Sie das im Kopf, wenn wir
# > in Abschnitt 8 sehen, dass das Modell zwischen Kugel und Würfel oft nur
# > mäßig sicher entscheidet.
#
# <a id="kapitel-2-2"></a>
# ## 2.2 Vorverarbeitung: One-Hot-Kodierung und Datensplit
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### One-Hot-Kodierung
#
# Neuronale Netze rechnen mit Zahlen, nicht mit Strings. Die Labels
# `'circle'`, `'noise'`, `'square'` müssen also numerisch werden. Man könnte
# einfach 0, 1, 2 vergeben — das wäre aber falsch, denn es würde eine **Ordnung**
# suggerieren („noise liegt zwischen circle und square“), die es nicht gibt.
#
# Der `OneHotEncoder` bildet stattdessen jede Klasse auf einen Einheitsvektor ab:
#
# $$\texttt{circle} \mapsto (1,0,0), \qquad
#   \texttt{noise} \mapsto (0,1,0), \qquad
#   \texttt{square} \mapsto (0,0,1)$$
#
# Alle Klassen haben damit denselben Abstand voneinander. `y` bekommt die Form
# `(600, 3)`. Diese Kodierung passt genau zur Verlustfunktion, die wir gleich
# verwenden (kategorische Kreuzentropie), und zur Softmax-Ausgabeschicht mit drei
# Neuronen.
#
# ### Der Datensplit — und ein bewusst stehengelassener Fallstrick
#
# Die letzten vier Zeilen teilen die Daten in Trainings- und Testmenge auf.
# Schauen Sie genau hin:
#
# ```python
# train_X = X[:300]
# test_X  = X[:300]   # identischer Bereich!
# ```
#
# **Trainings- und Testmenge sind dieselben 300 Stichproben.** Das ist in
# produktivem Code ein schwerer Fehler und heißt *Datenleck* (data leakage): Die
# gemessene Validierungsgenauigkeit sagt dann nichts über die
# **Generalisierungsfähigkeit** aus, sondern nur darüber, wie gut das Modell
# seine Trainingsdaten auswendig gelernt hat. Korrekt wäre etwa
# `test_X = X[300:]`, sodass die zweiten 300 Stichproben ungesehen bleiben.
#
# Warum ist das hier vertretbar? Weil das Ziel dieses Notebooks nicht das
# Erreichen einer belastbaren Testgenauigkeit ist, sondern die Demonstration von
# LRP. Für die Erklärungen selbst spielt es keine Rolle, ob das Bild aus dem
# Training stammt. Wichtig ist nur, dass Sie **die genannten Genauigkeitswerte
# nicht als Leistungsaussage lesen** — und dass Sie den Fehler in Ihrem eigenen
# Code nicht wiederholen. Der Vollständigkeit halber: Die Stichproben mit Index
# 300 bis 599 werden in diesem Notebook überhaupt nicht verwendet.

# %% [markdown]
# <a id="kapitel-3"></a>
# # 3. Das 3D-CNN aufbauen und trainieren
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ## Der größere Kontext: Warum Faltungsnetze?
#
# Man könnte die 4096 Voxel einfach in einen langen Vektor auffalten und ein
# klassisches, vollverbundenes Netz darauf ansetzen. Das wäre aus zwei Gründen
# schlecht:
#
# 1. **Parameterexplosion.** Schon eine erste versteckte Schicht mit 100 Neuronen
#    bräuchte $4096 \times 100 \approx 410.000$ Gewichte — mehr als wir
#    Trainingsbeispiele haben.
# 2. **Verlust der Nachbarschaft.** Beim Auffalten geht die Information verloren,
#    welche Voxel räumlich benachbart sind. Genau diese Information *ist* aber die
#    Form eines Objekts.
#
# **Convolutional Neural Networks (CNNs)** lösen beides durch zwei Ideen:
# *lokale Verbindungen* (jedes Neuron sieht nur einen kleinen Bildausschnitt) und
# *Gewichtsteilung* (dasselbe Filter wird über das gesamte Bild geschoben). Ein
# Filter, das eine Kante erkennt, muss so nur einmal gelernt werden und
# funktioniert an jeder Position. Diese Eigenschaft heißt
# **Translationsäquivarianz** und ist der Grund, warum CNNs die Bildverarbeitung
# seit 2012 dominieren.
#
# Formal berechnet eine 3D-Faltung für Ausgabekanal $c$ an Position $(i,j,k)$:
#
# $$z^{c}_{ijk} = \sum_{d}\sum_{u=-1}^{1}\sum_{v=-1}^{1}\sum_{w=-1}^{1}
#   w^{c,d}_{uvw} \cdot a^{d}_{i+u,\,j+v,\,k+w} \;+\; b^{c}$$
#
# Dabei läuft $d$ über die Eingabekanäle und $w^{c,d}$ ist der $3\times3\times3$-Kern.
# Ein solcher Kern hat 27 Gewichte — unabhängig von der Bildgröße.
#
# ## Was diese Zelle berechnet
#
# Das Netz besteht aus **vier identisch aufgebauten Blöcken**, gefolgt von einem
# Klassifikationskopf. Jeder Block enthält:
#
# **`Conv3D(k, (3,3,3), padding='SAME', kernel_regularizer=l2(1e-3))`** — die
# Faltung mit $k$ Filtern. `padding='SAME'` legt einen Rahmen aus Nullen um das
# Volumen, damit die Ausgabe dieselbe räumliche Größe wie die Eingabe hat. Der
# `l2`-Regularisierer addiert $\lambda \sum_i w_i^2$ mit $\lambda = 10^{-3}$ zum
# Verlust und drängt die Gewichte so gegen null — eine Standardmaßnahme gegen
# **Overfitting** (das Auswendiglernen der Trainingsdaten).
#
# **`BatchNormalization()`** — normalisiert die Ausgaben über den Mini-Batch:
#
# $$\hat{z} = \gamma \cdot \frac{z - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}} + \beta$$
#
# mit den gelernten Parametern $\gamma$ (Skalierung) und $\beta$ (Verschiebung).
# Das stabilisiert und beschleunigt das Training erheblich. Merken Sie sich diese
# Schicht — sie wird in Abschnitt 5 für LRP eine besondere Rolle spielen.
#
# **`Activation('relu')`** — die Nichtlinearität $\text{ReLU}(x) = \max(0, x)$.
# Ohne Nichtlinearität wäre die Verkettung mehrerer Schichten mathematisch
# äquivalent zu einer einzigen linearen Abbildung, und das Netz könnte nichts
# lernen, was komplexer als eine lineare Funktion ist. Dass Faltung und
# Aktivierung hier *getrennte* Schichten sind (statt `Conv3D(..., activation='relu')`),
# ist kein Zufall: LRP muss zwischen der linearen Operation und der Aktivierung
# unterscheiden können.
#
# **`MaxPooling3D((2,2,2))`** — fasst jeden $2\times2\times2$-Block zu seinem
# Maximum zusammen und **halbiert damit jede Raumdimension**. Das reduziert den
# Rechenaufwand und vergrößert das **rezeptive Feld**: Nach mehreren Pooling-Stufen
# „sieht“ ein einzelnes Neuron einen großen Teil des Originalvolumens.
#
# Die räumliche Auflösung entwickelt sich also so:
#
# | Stufe | Form (D×H×W×Kanäle) |
# |---|---|
# | Eingabe | 16 × 16 × 16 × 1 |
# | nach Block 1 | 8 × 8 × 8 × 8 |
# | nach Block 2 | 4 × 4 × 4 × 16 |
# | nach Block 3 | 2 × 2 × 2 × 32 |
# | nach Block 4 | 1 × 1 × 1 × 32 |
#
# Das ist das typische CNN-Muster: **räumlich schrumpfen, in der Kanaltiefe
# wachsen**. Aus vielen Ortsangaben mit wenig Bedeutung werden wenige Ortsangaben
# mit viel Bedeutung.
#
# Der Kopf besteht aus:
#
# - **`Flatten()`** — macht aus dem $1\times1\times1\times32$-Tensor einen Vektor
#   mit 32 Einträgen.
# - **`Dropout(0.5)`** — schaltet während des Trainings zufällig die Hälfte der
#   Neuronen ab. Das verhindert, dass sich das Netz auf einzelne Neuronen
#   verlässt, und wirkt wie ein Ensemble vieler kleinerer Netze. Bei der
#   Inferenz ist Dropout automatisch inaktiv.
# - **`Dense(3, activation='softmax')`** — drei Ausgabeneuronen, eines pro Klasse.
#   Die Softmax-Funktion
#
#   $$p_c = \frac{e^{z_c}}{\sum_{c'} e^{z_{c'}}}$$
#
#   verwandelt die drei rohen Werte (*Logits*) in eine Wahrscheinlichkeitsverteilung,
#   die sich zu 1 summiert.
#
# Insgesamt hat das Modell rund 46.000 Parameter — für ein neuronales Netz sehr
# wenig (ein ResNet-50 hat 25 Millionen), aber angemessen für 300
# Trainingsbeispiele.
#
# ## Training
#
# `model.compile(...)` legt fest, *wie* gelernt wird:
#
# - **Verlustfunktion:** kategorische Kreuzentropie. Für ein Beispiel mit
#   One-Hot-Label $y$ und Vorhersage $p$ gilt
#
#   $$\mathcal{L} = -\sum_{c} y_c \log p_c$$
#
#   Weil nur eine Komponente von $y$ gleich 1 ist, reduziert sich das auf
#   $-\log p_{\text{korrekte Klasse}}$: Der Verlust ist null bei perfekter
#   Sicherheit und wächst unbeschränkt, je näher die vorhergesagte
#   Wahrscheinlichkeit der richtigen Klasse an null rückt.
# - **Optimierer:** Adam mit Lernrate $10^{-4}$. Adam passt die Schrittweite pro
#   Parameter adaptiv an und ist der De-facto-Standard.
# - **Metrik:** Accuracy, also der Anteil korrekt klassifizierter Beispiele. Anders
#   als der Verlust ist sie direkt interpretierbar, aber nicht differenzierbar und
#   deshalb nicht als Optimierungsziel geeignet.
#
# Der `if`-Block darunter implementiert **Caching**: Liegt unter
# `output/notebooks/.../100_epochs/` bereits eine `.keras`-Datei, wird sie
# geladen; andernfalls wird 100 Epochen lang trainiert und das Ergebnis
# gespeichert. Eine *Epoche* ist ein vollständiger Durchgang durch die
# Trainingsdaten; bei `batch_size=32` und 300 Beispielen sind das 10
# Gewichtsaktualisierungen pro Epoche. Dieses Caching-Muster macht das Notebook
# beim erneuten Ausführen schnell und die Erklärungen von Lauf zu Lauf
# vergleichbar — bei XAI-Experimenten ein wichtiger Vorteil, denn ein anderes
# Modell ergibt andere Heatmaps.

# %%
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Conv3D, Dense, Dropout, \
                                    Flatten, GlobalAveragePooling3D, Input, MaxPooling3D, Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

MODEL_DIR = repo_root / "output" / "notebooks" / notebook_name / "100_epochs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "geometric_3d_cnn.keras"

input = Input((shape, shape, shape, 1))
x = input

x = Conv3D(8, (3, 3, 3), padding='SAME', activation=None, kernel_regularizer=l2(1e-3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling3D((2, 2, 2))(x)

x = Conv3D(16, (3, 3, 3), padding='SAME', activation=None, kernel_regularizer=l2(1e-3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling3D((2, 2, 2))(x)

x = Conv3D(32, (3, 3, 3), padding='SAME', activation=None, kernel_regularizer=l2(1e-3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling3D((2, 2, 2))(x)

x = Conv3D(32, (3, 3, 3), padding='SAME', activation=None, kernel_regularizer=l2(1e-3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling3D((2, 2, 2))(x)

x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(3, activation='softmax')(x)

model = Model(input, x)

model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])

existing_model_path = next(iter(sorted(MODEL_DIR.glob("*.keras"))), None)

if existing_model_path is not None:
    model = load_model(existing_model_path)
    print(f"Model geladen von: {existing_model_path}")
else:
    model.fit(
        train_X,
        train_y,
        validation_data=(test_X, test_y),
        batch_size=32,
        epochs=100,
    )

    model.save(MODEL_PATH)
    print(f"Model gespeichert unter: {MODEL_PATH}")

# %% [markdown]
# <a id="kapitel-3-1"></a>
# ## 3.1 Interpretation: Was beim Training passiert ist
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Die Ausgabe dieser Zelle enthält keinen Trainingsverlauf, sondern nur die
# Meldung `Model geladen von: .../geometric_3d_cnn.keras`. Das ist die
# erwartete Ausgabe **beim zweiten und jedem weiteren Lauf**: Das Modell wurde
# bereits einmal trainiert und liegt als Datei vor, der `else`-Zweig mit
# `model.fit(...)` wird übersprungen.
#
# Beim allerersten Lauf sähen Sie hier stattdessen 100 Zeilen mit Fortschrittsbalken
# und Kennzahlen der Art `loss: 0.31 - accuracy: 0.92 - val_loss: 0.29 - val_accuracy: 0.93`.
# Die beiden Größen mit `val_`-Präfix beziehen sich auf die Validierungsdaten und
# sind hier — siehe Abschnitt 2.2 — mit den Trainingsdaten identisch, also nicht
# aussagekräftig.
#
# Zusätzlich erscheinen Meldungen von TensorFlow selbst, etwa zu oneDNN oder zur
# erkannten GPU (`Created device ... NVIDIA GeForce RTX 4070 Laptop GPU`). Das sind
# Informationsmeldungen, keine Fehler.
#
# > **Nebenbemerkung zur Praxis:** Ein Warnhinweis von TensorFlow über
# > verfügbare CPU-Instruktionen oder abweichende Fließkommaergebnisse ist
# > harmlos. Die Zeile über oneDNN weist allerdings auf einen realen Effekt hin:
# > Numerische Ergebnisse können sich zwischen Hardwarekonfigurationen minimal
# > unterscheiden. Für Erklärungen, die auf Vorzeichen und kleinen Differenzen
# > beruhen, ist das ein Grund mehr, das Modell zu cachen statt neu zu trainieren.

# %% [markdown]
# <a id="kapitel-4"></a>
# # 4. Die Schichten des Modells zählen
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ## Was diese Zelle berechnet
#
# `len(model.layers)` gibt die Anzahl der Keras-Schichten aus. Das Ergebnis ist
# **20**, und diese Zahl setzt sich so zusammen:
#
# $$\underbrace{1}_{\text{Input}} + \underbrace{4 \times 4}_{\text{Conv, BN, ReLU, Pool}}
#   + \underbrace{1}_{\text{Flatten}} + \underbrace{1}_{\text{Dropout}}
#   + \underbrace{1}_{\text{Dense}} = 20$$
#
# ## Warum das gleich wichtig wird
#
# Diese scheinbar triviale Zelle ist eine Vorbereitung: LRP muss wissen, *an
# welcher Schicht* die Erklärung beginnen soll. Der Index wird in der nächsten
# Zelle als `N_layers = len(model.layers) - 1 = 19` berechnet — also der Index
# der **letzten** Schicht, der `Dense(3, softmax)`-Ausgabeschicht (Python zählt ab
# 0, der letzte gültige Index ist demnach 19).
#
# Der Ausdruck `len(model.layers) - 1` ist robuster, als eine Zahl fest
# einzutippen: Wer der Architektur später einen Block hinzufügt, muss nichts
# anpassen. Die auskommentierten Zeilen `#N_layers = 9` und `#N_layers = 32` im
# nächsten Codeblock sind Überreste früherer Experimente mit anderen
# Architekturen — ein guter Anlass für den Hinweis, dass man Erklärungen
# prinzipiell auch an *Zwischen*schichten abgreifen kann, um zu untersuchen, was
# das Netz auf mittlerer Abstraktionsebene repräsentiert.

# %%
print(len(model.layers))

# %% [markdown]
# <a id="kapitel-5"></a>
# # 5. Layer-wise Relevance Propagation (LRP)
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ## Der größere Kontext: Die Landschaft der XAI-Methoden
#
# Wir haben ein Modell, das klassifiziert. Jetzt wollen wir wissen: **Welche
# Voxel haben zu dieser Entscheidung beigetragen?** Für diese Frage gibt es
# mehrere Antwortstrategien:
#
# | Ansatz | Idee | Bekannte Vertreter |
# |---|---|---|
# | Gradientenbasiert | Wie stark ändert sich die Ausgabe, wenn ich ein Voxel minimal ändere? | Saliency Maps, Grad-CAM, Integrated Gradients |
# | Perturbationsbasiert | Was passiert, wenn ich Bildbereiche verdecke? | Occlusion, LIME, SHAP |
# | Rückpropagation von Relevanz | Verteile die Ausgabe schichtweise rückwärts auf die Eingabe | **LRP**, DeepLIFT |
#
# Reine Gradienten haben einen bekannten Schwachpunkt: Sie messen *lokale
# Sensitivität*, nicht *Beitrag*. Ein Voxel kann für die Entscheidung
# entscheidend sein und trotzdem einen Gradienten von null haben, wenn die
# Aktivierung gesättigt ist. LRP wurde 2015 von Bach et al. genau deshalb
# entwickelt und liefert typischerweise deutlich weniger verrauschte Karten.
#
# ## Die Kernidee von LRP: Erhaltung
#
# LRP beruht auf einem einzigen Prinzip, dem **Konservierungsprinzip**. Man
# startet mit der Ausgabe des Netzes für die interessierende Klasse $c$ und
# verteilt diesen Wert rückwärts durch alle Schichten, wobei in jedem Schritt die
# Gesamtsumme erhalten bleibt:
#
# $$\sum_i R_i^{(1)} = \dots = \sum_j R_j^{(l)} = \sum_k R_k^{(l+1)} = \dots = f_c(x)$$
#
# Am Ende hat jedes Eingabevoxel $i$ einen **Relevanzwert** $R_i$. Positive
# Relevanz heißt „dieses Voxel hat für die Klasse gesprochen“, negative Relevanz
# „es hat dagegen gesprochen“. Diese Vorzeicheninformation ist ein wesentlicher
# Vorteil gegenüber Methoden, die nur Betragsstärken liefern.
#
# Die Aufteilung innerhalb einer Schicht geschieht proportional zum Beitrag jedes
# Neurons. Die Grundregel (LRP-0) lautet:
#
# $$R_i = \sum_j \frac{a_i w_{ij}}{\sum_{i'} a_{i'} w_{i'j} + b_j} \; R_j$$
#
# Dabei ist $a_i$ die Aktivierung von Neuron $i$, $w_{ij}$ das Gewicht zu Neuron
# $j$ der nächsten Schicht und $R_j$ die dort vorliegende Relevanz. Der Bruch ist
# genau der relative Anteil, den $i$ am Eingang von $j$ hatte.
#
# In dieser reinen Form ist LRP-0 numerisch instabil: Wird der Nenner nahe null,
# explodiert der Quotient. Deshalb existieren stabilisierte Varianten, von denen
# zwei in diesem Notebook zum Einsatz kommen:
#
# **Die $\varepsilon$-Regel** addiert einen kleinen Term im Nenner:
#
# $$R_i = \sum_j \frac{a_i w_{ij}}{z_j + \varepsilon \cdot \operatorname{sign}(z_j)} \; R_j,
#   \qquad z_j = \sum_{i'} a_{i'} w_{i'j} + b_j$$
#
# Das dämpft schwache, verrauschte Beiträge und ergibt schärfere Karten. Der Preis:
# Die Relevanzsumme ist nur noch näherungsweise erhalten (ein Teil wird
# „absorbiert“).
#
# **Die $\alpha\beta$-Regel** trennt positive und negative Beiträge und gewichtet
# sie unterschiedlich:
#
# $$R_i = \sum_j \left( \alpha \frac{(a_i w_{ij})^+}{\sum_{i'} (a_{i'} w_{i'j})^+}
#   \; - \; \beta \frac{(a_i w_{ij})^-}{\sum_{i'} (a_{i'} w_{i'j})^-} \right) R_j$$
#
# mit der Nebenbedingung $\alpha - \beta = 1$, die die Erhaltung sicherstellt.
# $(\cdot)^+$ bezeichnet den positiven, $(\cdot)^-$ den negativen Anteil. Bei
# $\alpha = 2, \beta = 1$ — den Werten dieses Notebooks — werden also Beiträge
# *für* die Klasse doppelt gewichtet und Beiträge *gegen* sie einfach abgezogen.
# Man erhält so kontrastreiche Karten, die klar zeigen, was die Entscheidung
# getragen hat. Die Implementierung erzwingt diese Bedingung übrigens per
# `assert alpha == beta + 1`.
#
# ## Was diese Zelle berechnet
#
# `LRP(model, layer=N_layers, idx=i, strategy=strategy)` baut ein **zweites
# Keras-Modell**, das rückwärts läuft. Diese Konstruktion ist der Kern des
# `explainability`-Pakets dieses Repositories, und sie ist elegant: Die
# Rückpropagation ist selbst ein Graph aus Keras-Schichten, lässt sich also mit
# `.predict()` aufrufen und auf der GPU ausführen. Intern passieren dabei drei
# Vorbereitungsschritte, die man kennen sollte:
#
# 1. **Die Softmax wird entfernt** (`remove_activation`). Erklärt wird der rohe
#    Logit $z_c$, nicht die Wahrscheinlichkeit $p_c$. Der Grund: Die Softmax
#    normalisiert über alle Klassen, ihr Wert für Klasse $c$ hängt damit auch von
#    den anderen Klassen ab — das würde die Zuordnung von Relevanz verfälschen.
# 2. **BatchNormalization wird in die Faltung hineingerechnet** (`fuse_batchnorm`).
#    Da BatchNorm zur Inferenzzeit eine feste affine Abbildung ist, kann man sie
#    in die Gewichte der vorangehenden Faltung einrechnen. Danach sieht LRP eine
#    saubere Kette aus lineare Operation → Aktivierung, für die die obigen Regeln
#    definiert sind.
# 3. **Die Zielklasse wird maskiert.** Die Startrelevanz ist
#    $R_j^{(L)} = z_j \cdot \delta_{j, \texttt{idx}}$, also der Logit der
#    gewünschten Klasse an ihrer Position und null überall sonst. Genau deshalb
#    braucht man **pro Klasse einen eigenen Erklärer** — und deshalb legt die
#    Dictionary-Comprehension am Ende drei Objekte an, eines für `circle`, `noise`
#    und `square`.
#
# Die Schleife, die `layer.rate = 0.0` für alle Dropout-Schichten setzt, ist eine
# Sicherheitsmaßnahme: Zufälliges Ausschalten von Neuronen würde Erklärungen von
# Aufruf zu Aufruf verändern. Bei der Inferenz ist Dropout ohnehin inaktiv, aber
# im rückwärts laufenden LRP-Graphen möchte man sich darauf nicht verlassen.
#
# <a id="kapitel-5-1"></a>
# ## 5.1 Die LRP-Strategie dieses Notebooks
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Ein wichtiges Ergebnis der LRP-Forschung (Montavon et al., 2019) lautet: **Es
# gibt nicht eine beste Regel, sondern eine beste Regel pro Schichttyp.** Die
# empfohlene Aufteilung für Faltungsnetze:
#
# - **Untere Schichten** (nahe der Eingabe): Regeln, die den Wertebereich der
#   Eingabe berücksichtigen, damit die Karte nicht verrauscht wird.
# - **Mittlere Schichten:** $\alpha\beta$ (bzw. $\gamma$), für kontrastreiche,
#   räumlich lokalisierte Erklärungen.
# - **Obere Schichten** (nahe der Ausgabe): $\varepsilon$, zur Rauschunterdrückung.
#
# Genau das kodiert `LRPStrategy`. Die Liste enthält einen Eintrag pro Schicht
# mit Gewichten und ist **in Vorwärtsrichtung** (Eingabe → Ausgabe) notiert;
# intern wird sie mit `[::-1]` umgedreht, weil der LRP-Graph rückwärts läuft.
# Das Modell hat fünf gewichtsbehaftete Schichten (vier `Conv3D` plus ein
# `Dense`), und die Liste muss genau fünf Einträge haben — andernfalls schlägt
# eine `assert`-Prüfung fehl. Die Zuordnung ist damit:
#
# | Eintrag | Schicht im Modell | Regel |
# |---|---|---|
# | `{'b': True}` | Conv3D #1 (8 Filter) | Aktivierungen werden durch Einsen ersetzt |
# | `{'alpha': 2, 'beta': 1}` | Conv3D #2 (16 Filter) | $\alpha\beta$-Regel |
# | `{'alpha': 2, 'beta': 1}` | Conv3D #3 (32 Filter) | $\alpha\beta$-Regel |
# | `{'alpha': 2, 'beta': 1}` | Conv3D #4 (32 Filter) | $\alpha\beta$-Regel |
# | `{'epsilon': 0.5}` | Dense (3 Ausgaben) | $\varepsilon$-Regel |
#
# Die erste Zeile verdient eine Erläuterung. Das Flag `b` bewirkt in der
# Implementierung `a = tf.ones_like(a)`: Die Eingabeaktivierungen der ersten
# Faltung werden durch Einsen ersetzt, sodass die Relevanzverteilung dort nur
# noch von den Gewichten abhängt. Der Effekt ist eine gleichmäßigere,
# weniger intensitätsabhängige Karte in der Eingabeschicht. Das ist verwandt mit
# der $z^{\mathcal{B}}$-Regel, die für Eingaben mit bekanntem Wertebereich
# entworfen wurde — bei unseren Daten liegt dieser Bereich sauber in $[0,1]$.
#
# Für die vier `MaxPooling3D`-Schichten wird keine Strategie angegeben, es gilt
# also der Standard **winner-takes-all**: Die Relevanz eines Pooling-Ausgangs
# geht vollständig an dasjenige Voxel, das im Vorwärtspass das Maximum war. Das
# ist konsistent mit der Vorwärtsoperation und der Grund, warum die Erklärungen
# räumlich präzise bleiben, obwohl vier Halbierungsschritte dazwischen liegen.

# %%
from explainability import LRP, LRPStrategy


alpha = 2
beta = 1

strategy = LRPStrategy(
    layers=[
        {'b': True},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'epsilon': 0.5}
    ]
)
#N_layers = 9
#N_layers = 32
N_layers = len(model.layers)-1

for layer in model.layers:
    if "dropout" in layer.name.lower():
        layer.rate = 0.0

explainers = {
    encoder.categories_[0][i]: LRP(model, layer=N_layers, idx=i, strategy=strategy) \
    for i in range(3)
}

# %% [markdown]
# <a id="kapitel-6"></a>
# # 6. Die Reihenfolge der Klassen
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ## Was diese Zelle berechnet
#
# `encoder.categories_[0]` gibt die Klassennamen in genau der Reihenfolge zurück,
# in der der `OneHotEncoder` sie den Spalten zugeordnet hat. Die Ausgabe ist
#
# ```
# array(['circle', 'noise', 'square'], dtype='<U6')
# ```
#
# Die Reihenfolge ist **alphabetisch** — nicht die Reihenfolge, in der die Daten
# erzeugt wurden (dort war es Würfel, Kugel, Rauschen). Spalte 0 des
# Ausgabevektors entspricht also `circle`, Spalte 1 `noise`, Spalte 2 `square`.
#
# ## Warum eine so kleine Zelle ihren eigenen Abschnitt bekommt
#
# Diese Zeile sieht nach Beiwerk aus, verhindert aber einen der häufigsten und
# unangenehmsten Fehler im Machine Learning: **die Verwechslung von
# Klassenindizes**. Ein solcher Fehler produziert keinen Programmabbruch. Das
# Modell trainiert normal, die Metriken sehen plausibel aus, und nur die
# Beschriftung der Ergebnisse ist falsch — bei einer Erklärungsanalyse bedeutet
# das, dass man die Heatmap der Klasse A betrachtet und über Klasse B
# nachdenkt.
#
# Der übrige Code schützt sich dagegen, indem er die Klassennamen konsequent
# aus dem Encoder ableitet, statt sie irgendwo neu einzutippen:
#
# ```python
# explainers = {encoder.categories_[0][i]: LRP(..., idx=i, ...) for i in range(3)}
# ```
#
# Der Index `i`, der an LRP als Zielklasse übergeben wird, und der Name, unter
# dem der Erklärer im Dictionary landet, stammen damit aus derselben Quelle und
# können nicht auseinanderlaufen. Das ist eine kleine Regel mit großem Nutzen:
# **Beschriftungen immer aus der Datenquelle ableiten, nie parallel pflegen.**

# %%
encoder.categories_[0]

# %% [markdown]
# <a id="kapitel-7"></a>
# # 7. Erklärungen für echte Stichproben
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ## Was diese Zelle berechnet
#
# Für die ersten zehn Stichproben aus `test_X` wird jeweils
#
# 1. das **Eingabevolumen** als Schnittbildreihe gezeichnet (Titel: `Image <Klasse>`),
# 2. und danach für **jede der drei Klassen** eine LRP-Erklärung berechnet und
#    ebenfalls als Schnittbildreihe gezeichnet.
#
# Pro Stichprobe entstehen also vier Bildreihen, insgesamt 40. Entscheidend ist,
# dass alle drei Erklärungen zum *gleichen* Eingabebild gehören. Wir fragen das
# Netz gewissermaßen dreimal etwas Verschiedenes:
#
# - *Was an diesem Bild spricht für eine Kugel?*
# - *Was spricht für Rauschen?*
# - *Was spricht für einen Würfel?*
#
# Diese **klassenspezifische** Betrachtung ist eine der Stärken von LRP. Man
# erklärt nicht „die Vorhersage“ als Ganzes, sondern kann jede Hypothese einzeln
# befragen — auch solche, die das Netz verworfen hat.
#
# ## Die Normalisierung der Darstellung
#
# ```python
# explanation = explanation / np.amax(np.abs(explanation))
# ```
#
# Diese Zeile teilt die Relevanzkarte durch ihren größten Absolutwert, sodass alle
# Werte in $[-1, 1]$ liegen:
#
# $$\tilde{R} = \frac{R}{\max_i |R_i|}$$
#
# Das ist nötig, weil die Rohwerte von der Größe des Logits abhängen und von Bild
# zu Bild um Größenordnungen schwanken können. Nach der Normalisierung passt jede
# Karte zur festen Farbskala `clim=(-1, 1)`, und die Bilder sind untereinander
# vergleichbar.
#
# Ein wichtiger Vorbehalt: Die Normalisierung ist **relativ zum jeweiligen Bild**.
# Ein sattes Rot bedeutet „das stärkste Signal *in diesem Bild*“ — nicht „ein
# starkes Signal absolut“. Zwei Karten können optisch gleich kräftig aussehen und
# doch zu völlig unterschiedlich sicheren Vorhersagen gehören.
#
# ## Die Farbskala verstehen
#
# Alle Erklärungsbilder verwenden die divergierende Colormap `seismic`, die um
# null zentriert ist:
#
# | Farbe | Relevanz | Bedeutung |
# |---|---|---|
# | **Rot** | $R > 0$ | Dieses Voxel **spricht für** die genannte Klasse |
# | **Weiß** | $R \approx 0$ | Kein nennenswerter Beitrag |
# | **Blau** | $R < 0$ | Dieses Voxel **spricht gegen** die Klasse |
#
# Divergierende Colormaps sind bei vorzeichenbehafteten Größen die richtige Wahl.
# Eine Skala wie `viridis` oder `jet` würde negative und positive Relevanz
# ununterscheidbar machen und damit die Hälfte der Information verschenken.

# %%
for i in range(10):
    img = test_X[i]
    label = encoder.categories_[0][np.argmax(test_y[i])]

    fig, ax = plt.subplots(1, shape, figsize=(15, 2))
    fig.suptitle(f'Image {label}')
    for i in range(shape):
        ax[i].imshow(img[i], cmap='Greys_r')
        ax[i].axis('off')
    plt.show()

    for classname in encoder.categories_[0]:
        explanation = explainers[classname].predict(np.expand_dims(img, axis=0))
        explanation = explanation / np.amax(np.abs(explanation))

        fig, ax = plt.subplots(1, shape, figsize=(15, 2))
        fig.suptitle(f'{classname} explanation')
        for j in range(shape):
            im = ax[j].imshow(explanation[0,j], cmap='seismic', clim=(-1, 1))
            ax[j].axis('off')
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        plt.show()

fig.savefig(target_dir / '2_geometric_samples_explanations.png', bbox_inches='tight', dpi=150)
plt.show()
plt.close(fig)        


# %% [markdown]
# <a id="kapitel-7-1"></a>
# ## 7.1 Interpretation: Die Heatmaps der Stichproben
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Die Ausgabe ist lang — 40 Bildreihen. Statt sie einzeln durchzugehen, lohnt es
# sich, auf vier Muster zu achten, die sich durch alle Beispiele ziehen.
#
# ### Beobachtung 1: Ein auffälliges Artefakt am Bildrand
#
# Das erste, was ins Auge fällt, ist nicht das erwartete Signal, sondern ein
# **kräftiger rot-blauer Streifen am linken Rand der Schnitte 0 und 1** — und
# zwar bei *jeder* Klasse und *jeder* Stichprobe, auch bei Bildern, in denen an
# dieser Stelle überhaupt kein Objekt liegt. Es ist die farbstärkste Struktur in
# fast allen Erklärungsbildern.
#
# Das ist kein Fehler in Ihrem Verständnis, sondern ein reales Artefakt. Es hat
# zwei ineinandergreifende Ursachen:
#
# **Zero-Padding.** Die Faltungen verwenden `padding='SAME'`, legen also einen
# Rahmen aus Nullen um das Volumen. Für ein Neuron am Rand ist ein Teil des
# rezeptiven Felds daher künstlich null, und der Nenner der LRP-Regel,
# $z_j = \sum_i a_i w_{ij}$, wird betragsmäßig klein. Da die Relevanz durch diesen
# Nenner geteilt wird, wächst der Quotient dort stark an — ein Randeffekt, der
# durch vier Faltungsschichten hindurch verstärkt wird.
#
# **Aggressives Pooling.** Vier `MaxPooling3D`-Stufen reduzieren $16^3$ auf
# $1^3$. Am Ende gibt es genau *eine* räumliche Position, und die
# winner-takes-all-Rückverteilung muss die gesamte Relevanz über vier Stufen auf
# einen einzigen Pfad zurückführen. Kleine Asymmetrien in der Indizierung — etwa
# welches Voxel bei Gleichstand als Maximum gilt — schlagen dabei systematisch
# auf bestimmte Randpositionen durch.
#
# Der Nebeneffekt ist tückisch: Weil die Darstellung auf
# $\max_i |R_i|$ normalisiert wird, **setzt das Artefakt die Farbskala** und
# drückt das eigentlich interessierende Signal in ein blasses Rosa. Man sieht das
# gut daran, dass die Objektregionen zwar Struktur zeigen, aber weit von den
# Extremwerten $\pm 1$ entfernt bleiben.
#
# > **Praxislehre:** Das ist keine Randnotiz, sondern die zentrale Lektion dieses
# > Abschnitts. Eine Heatmap ist nie eine neutrale Wahrheit, sondern das Produkt
# > aus Modell, LRP-Regelwahl, Architektur *und* Darstellungsentscheidungen.
# > Bevor man aus einer Erklärung eine Aussage über das Modell ableitet, muss man
# > wissen, welche Strukturen methodisch bedingt sind. Bei echten Anwendungen
# > begegnet man solchen Artefakten ständig — und die Versuchung, sie inhaltlich
# > zu deuten („das Modell achtet auf den Bildrand!“), ist groß.
#
# ### Beobachtung 2: Das Signal folgt trotzdem dem Objekt
#
# Sieht man von den Schnitten 0 und 1 ab, ist das Ergebnis erfreulich: Die
# Relevanz erscheint **genau in denjenigen Schnitten, in denen das Objekt liegt**.
# Bei der ersten Stichprobe (eine Kugel in den Schnitten 8 bis 15) sind die
# Schnitte 2 bis 7 nahezu weiß, während ab Schnitt 8 rote und blaue Strukturen an
# der Position der Kugel auftreten. Das Modell reagiert also auf das Objekt und
# nicht auf zufällige leere Regionen — die Grundannahme, die wir prüfen wollten,
# ist erfüllt.
#
# Interessant ist das Zusammenspiel von Rot und Blau innerhalb einer Objektregion.
# Bei der Kugel liegen die Farben oft dicht beieinander, statt die Kugel flächig
# rot zu füllen. Das ist charakteristisch: Das Netz hat mit $3\times3\times3$-Filtern
# vor allem **Kanten und Krümmung** gelernt, nicht „Fläche“. Die informative
# Struktur ist die Objektgrenze, und dort erzeugt ein Kantenfilter naturgemäß
# benachbarte positive und negative Antworten.
#
# ### Beobachtung 3: Rauschen sieht völlig anders aus
#
# Vergleichen Sie die Erklärungen für ein `noise`-Bild mit denen für ein
# geometrisches Objekt. Bei Rauschen ist die Relevanz **über das gesamte Volumen
# verstreut**, in allen 16 Schnitten, mit deutlich kräftigeren Farben und ohne
# jede räumliche Konzentration. Auch das ist genau richtig: Bei Rauschen *gibt* es
# keine lokalisierte Evidenz. Die Klasse ist an der Textur des ganzen Volumens
# erkennbar, nicht an einer Stelle.
#
# Diese qualitative Unterscheidung — *lokalisiert* versus *global verstreut* — ist
# ein Ergebnis, das man in einer reinen Genauigkeitszahl nicht sehen würde. Die
# Erklärung sagt uns nicht nur, *wo* das Modell hinschaut, sondern auch, *welcher
# Art* die Evidenz ist.
#
# ### Beobachtung 4: Die drei Klassenerklärungen ähneln sich
#
# Ein kritischer Punkt: Betrachtet man dasselbe Eingabebild dreimal (für
# `circle`, `noise` und `square`), sind die Karten sich strukturell **ähnlicher,
# als man erwarten würde**. Sie unterscheiden sich in Details und teils im
# Vorzeichen, aber die grobe räumliche Verteilung stimmt weitgehend überein.
#
# Dieses Phänomen ist in der Literatur gut dokumentiert und wird als mangelnde
# **Klassensensitivität** von Attributionsmethoden kritisiert. Es hat hier eine
# naheliegende Erklärung: Nach vier Pooling-Stufen bleiben nur 32 Zahlen übrig,
# und alle drei Klassen lesen dieselben 32 Merkmale — lediglich mit anderen
# Gewichten in der letzten Schicht. Die räumliche Herkunft der Relevanz ist damit
# zwangsläufig für alle Klassen sehr ähnlich. Wer klar getrennte
# klassenspezifische Karten will, braucht ein Modell mit mehr räumlicher Auflösung
# vor der Klassifikation.
#
# ### Ein technischer Hinweis zur gespeicherten Datei
#
# Der `savefig`-Aufruf steht *nach* der Schleife und bezieht sich damit auf die
# Variable `fig`, die zu diesem Zeitpunkt nur noch die **letzte** erzeugte Figur
# enthält. Die Datei `2_geometric_samples_explanations.png` zeigt deshalb nicht
# alle 40 Reihen, sondern eine einzige. Im Notebook selbst sind alle Reihen
# sichtbar, weil `plt.show()` innerhalb der Schleife aufgerufen wird. Genau dieses
# Problem löst Abschnitt 9, in dem alles in *eine* Figur mit vielen Zeilen
# gezeichnet und dann als Ganzes gespeichert wird.

# %% [markdown]
# <a id="kapitel-8"></a>
# # 8. Das Gegenprobe-Experiment: Hybrid-Objekte
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ## Der größere Kontext: Erklärungen sind Hypothesen, die man testen muss
#
# Bis hierhin haben wir Heatmaps *betrachtet*. Das ist eine notorisch unzuverlässige
# Form der Validierung, denn menschliche Mustererkennung findet auch dort Struktur,
# wo keine ist. Die Forschung hat gezeigt, dass plausibel *aussehende* Erklärungen
# von Modellen stammen können, die nachweislich Unsinn gelernt haben — und dass
# manche Attributionsmethoden Karten produzieren, die sich kaum ändern, wenn man
# die Modellgewichte randomisiert (Adebayo et al., 2018).
#
# Dieser Abschnitt geht deshalb einen Schritt weiter und führt ein **kontrolliertes
# Experiment** durch. Wir konstruieren Eingaben, für die wir eine klare Erwartung
# formulieren können, und prüfen, ob Vorhersage und Erklärung dieser Erwartung
# folgen. Das ist der Übergang von *beobachtender* zu *experimenteller*
# Interpretierbarkeit — methodisch dasselbe Vorgehen wie in der Physiologie, wo man
# einen Reiz gezielt variiert, statt nur zu beobachten.
#
# ## Was diese Zelle berechnet
#
# **Schritt 1: Zwei perfekte Referenzobjekte.** Anders als im Trainingsdatensatz
# werden hier ein Würfel und eine Kugel *deterministisch* und *exakt zentriert*
# erzeugt:
#
# - `square[4:12, 4:12, 4:12] = 1` — ein Würfel mit Kantenlänge 8, mittig im
#   Volumen.
# - Eine Kugel mit Zentrum $(8,8,8)$ und Radius 4, gezeichnet über eine dreifach
#   verschachtelte Schleife mit der Bedingung $\|p - z\|_2 \le 4$.
#
# Beide sind vollständig sichtbar, nicht abgeschnitten und haben dieselbe
# Ausdehnung. Das eliminiert die Störgrößen aus Abschnitt 2.1.
#
# **Schritt 2: Sechs Hybride.** Aus den beiden Objekten werden per
# `np.concatenate` **Chimären** gebaut: Die eine Hälfte des Volumens stammt vom
# Würfel, die andere von der Kugel. Das geschieht für alle drei Raumachsen und in
# beiden Reihenfolgen, also $3 \times 2 = 6$ Kombinationen. Zum Beispiel:
#
# ```python
# np.concatenate([square[:8], circle[8:]], axis=0)   # vorne Würfel, hinten Kugel
# np.concatenate([circle[:8], square[8:]], axis=0)   # umgekehrt
# ```
#
# Diese Objekte sind **out-of-distribution**: Sie gehören zu keiner Trainingsklasse
# und wurden dem Modell nie gezeigt. Genau das macht sie wertvoll — sie erzwingen
# eine Entscheidung zwischen zwei gleichzeitig vorhandenen Evidenzen und zeigen so,
# wie das Modell Beweise abwägt.
#
# **Schritt 3: Vorhersage und Erklärung.** Für jedes Hybrid wird die Vorhersage
# berechnet (und als Titel über der Bildreihe angezeigt), dann eine Erklärung pro
# Klasse.
#
# **Schritt 4: Das Differenzbild.** Der interessanteste Teil. Hier werden die
# Erklärungen für `square` und `circle` zunächst *unterschiedlich* normalisiert,
# nämlich auf den Bereich $[0,1]$ per Min-Max:
#
# $$R^{\text{norm}} = \frac{R - \min_i R_i}{\max_i (R_i - \min_i R_i)}$$
#
# und anschließend voneinander abgezogen:
#
# $$D = R^{\text{norm}}_{\text{square}} - R^{\text{norm}}_{\text{circle}}$$
#
# Die Idee ist eine **kontrastive Erklärung**: Nicht „was spricht für Würfel“,
# sondern „was spricht *eher* für Würfel als für Kugel“. Rot markiert dann
# Regionen, die die Würfel-Hypothese relativ stärker stützen, blau solche, die die
# Kugel-Hypothese stützen. Bei einem sauber arbeitenden Modell und einem Hybrid
# erwarten wir Rot in der Würfelhälfte und Blau in der Kugelhälfte.
#
# > **Methodische Warnung zur Differenz.** Die Min-Max-Normalisierung verschiebt
# > den Nullpunkt: Nach der Transformation ist der kleinste (also negativste)
# > Relevanzwert 0 und nicht mehr „kein Beitrag“. Die Differenz zweier so
# > skalierter Karten mischt daher echte Kontrastinformation mit
# > Skalierungseffekten, und beide Karten können unterschiedliche Skalenfaktoren
# > erhalten haben. Die Variablenbezeichnung `absolute_difference` ist
# > zusätzlich irreführend, denn es wird kein Absolutbetrag gebildet — das
# > Vorzeichen bleibt erhalten, und das ist gut so, sonst wäre die Richtung des
# > Kontrasts verloren. Das Differenzbild taugt als qualitativer Hinweis, nicht
# > als quantitatives Maß.

# %%
from scipy.spatial.distance import euclidean


labels = encoder.categories_[0]

square = np.zeros((16, 16, 16, 1))
square[4:12,4:12,4:12,0] = 1

fig, ax = plt.subplots(1, shape, figsize=(15, 2))
fig.suptitle('Square')
for j in range(shape):
    ax[j].imshow(square[j], cmap='Greys_r')
    ax[j].axis('off')
plt.show()

circle = np.zeros((16, 16, 16, 1))
center = (8, 8, 8)
radius = 4

for i in range(16):
    for j in range(16):
        for k in range(16):
            if euclidean((i, j, k), center) <= radius:
                circle[i,j,k,0] = 1
                
fig, ax = plt.subplots(1, shape, figsize=(15, 2))
fig.suptitle('Circle')
for j in range(shape):
    ax[j].imshow(circle[j], cmap='Greys_r')
    ax[j].axis('off')
plt.show()

combinations = [
    np.concatenate([square[:8], circle[8:]], axis=0),
    np.concatenate([circle[:8], square[8:]], axis=0),
    np.concatenate([square[:,:8], circle[:,8:]], axis=1),
    np.concatenate([circle[:,:8], square[:,8:]], axis=1),
    np.concatenate([square[:,:,:8], circle[:,:,8:]], axis=2),
    np.concatenate([circle[:,:,:8], square[:,:,8:]], axis=2),
]

for i in range(len(combinations)):
    fig, ax = plt.subplots(1, shape, figsize=(15, 2))
    prediction = model.predict(np.expand_dims(combinations[i], axis=0))[0]
    fig.suptitle(' '.join([f'{labels[i]}: {prediction[i]:.2f}' for i in range(len(labels))]))
    
    for j in range(shape):
        ax[j].imshow(combinations[i][j], cmap='Greys_r')
        ax[j].axis('off')
        
    plt.show()
        
    for j in range(3):
        fig, ax = plt.subplots(1, shape, figsize=(15, 2))
        classname = encoder.categories_[0][j]
        explanation = explainers[classname].predict(np.expand_dims(combinations[i], axis=0))[0]
        explanation = explanation / np.amax(np.abs(explanation))
        fig.suptitle(f'{classname} explanation')
        
        for k in range(shape):
            im = ax[k].imshow(explanation[k], cmap='seismic', clim=(-1, 1))
            ax[k].axis('off')
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        plt.show()
        
    square_explanation = explainers['square'].predict(np.expand_dims(combinations[i], axis=0))[0]
    square_explanation = square_explanation - np.amin(square_explanation)
    square_explanation = square_explanation / np.amax(square_explanation)
    circle_explanation = explainers['circle'].predict(np.expand_dims(combinations[i], axis=0))[0]
    circle_explanation = circle_explanation - np.amin(circle_explanation)
    circle_explanation = circle_explanation / np.amax(circle_explanation)
    absolute_difference = square_explanation - circle_explanation
    
    fig, ax = plt.subplots(1, shape, figsize=(15, 2))
    fig.suptitle('Square explanation - circle explanation')

    for j in range(shape):
        im = ax[j].imshow(absolute_difference[j], cmap='seismic', clim=(-1, 1))
        ax[j].axis('off')
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.show()


# %% [markdown]
# <a id="kapitel-8-1"></a>
# ## 8.1 Interpretation: Vorhersagen und Erklärungen der Hybride
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Die Referenzobjekte
#
# Die ersten beiden Bildreihen zeigen den perfekten Würfel und die perfekte Kugel.
# Der Würfel erscheint in den Schnitten 4 bis 11 als identisches Quadrat, die
# Kugel als Scheibe, die von Schnitt 4 bis 8 wächst und bis Schnitt 12 wieder
# schrumpft. Das sind die Idealformen, gegen die wir alles Weitere vergleichen.
#
# ### Wie man die Hybride in der Schnittdarstellung erkennt
#
# Die sechs Kombinationen sehen unterschiedlich aus, je nachdem, entlang welcher
# Achse geschnitten wurde — ein Punkt, der leicht verwirrt:
#
# - **`axis=0`** ist genau die Achse, entlang der auch die Schnittbilder angeordnet
#   sind. Man sieht daher *reine* Bilder: die Schnitte 4 bis 7 zeigen ein
#   Würfelquadrat, die Schnitte 8 bis 11 eine Kugelscheibe (oder umgekehrt).
# - **`axis=1` und `axis=2`** schneiden *innerhalb* jedes Bildes. Jeder einzelne
#   Schnitt ist deshalb selbst ein Mischbild: oben flach und eckig, unten
#   gerundet — bzw. links und rechts bei `axis=2`. Diese Formen sehen aus wie
#   halbe Quadrate mit angesetzter Kuppe.
#
# ### Beobachtung 1: `noise` ist immer exakt 0.00
#
# In allen sechs Titelzeilen steht `noise: 0.00`. Das Modell schließt Rauschen mit
# maximaler Sicherheit aus — bei Objekten, die es nie gesehen hat.
#
# Das ist gleichzeitig beruhigend und verdächtig. Beruhigend, weil die Hybride
# tatsächlich strukturiert sind. Verdächtig, weil eine Wahrscheinlichkeit von
# exakt null bei *out-of-distribution*-Eingaben auf eine sehr einfache
# Entscheidungsregel hindeutet. Erinnern Sie sich an Abschnitt 2: Rauschen war die
# einzige Klasse mit kontinuierlichen Werten und einer mittleren Intensität von
# etwa 0.5, während Würfel und Kugeln binär sind und nur einen Bruchteil des
# Volumens füllen. Das Modell hat mit hoher Wahrscheinlichkeit den
# **Intensitäts-Shortcut** gelernt statt eines echten Texturkonzepts. Für diesen
# Datensatz funktioniert das perfekt; auf Daten mit anderem Intensitätsprofil
# würde es sofort zusammenbrechen.
#
# ### Beobachtung 2: `square` gewinnt jedes Mal
#
# Die sechs Vorhersagen lauten:
#
# | # | Achse | Erste Hälfte | `circle` | `square` |
# |---|---|---|---|---|
# | 1 | 0 (Schnittachse) | Würfel | 0.41 | **0.59** |
# | 2 | 0 (Schnittachse) | Kugel | 0.27 | **0.73** |
# | 3 | 1 (Höhe) | Würfel | 0.20 | **0.80** |
# | 4 | 1 (Höhe) | Kugel | 0.14 | **0.86** |
# | 5 | 2 (Breite) | Würfel | 0.41 | **0.59** |
# | 6 | 2 (Breite) | Kugel | 0.36 | **0.64** |
#
# Das Modell entscheidet **in allen sechs Fällen für `square`**, mit
# Wahrscheinlichkeiten zwischen 0.59 und 0.86. Es gibt also einen deutlichen
# **Würfel-Bias**. Dafür gibt es mehrere plausible Gründe: Der Würfel hat mit
# Kantenlänge 8 ($8^3 = 512$ gesetzte Voxel) fast das Doppelte an Masse wie die
# Kugel mit Radius 4 (etwa 270 Voxel), und gerade Kanten und Ecken sind für
# $3\times3\times3$-Filter besonders leicht zu detektierende Merkmale, während
# eine gekrümmte Oberfläche auf einem $16^3$-Gitter nur grob aufgelöst ist.
#
# ### Beobachtung 3: Die fehlende Symmetrie ist das eigentlich Aufschlussreiche
#
# Vergleichen Sie die Paare in der Tabelle: Zeile 1 gegen Zeile 2 enthalten
# *exakt dieselben zwei Hälften*, nur in umgekehrter Anordnung. Wäre das Modell
# gegenüber der Position der Evidenz gleichgültig, müssten beide Zeilen dieselbe
# Vorhersage ergeben. Stattdessen springt `square` von 0.59 auf 0.73 — und bei
# `axis=1` von 0.80 auf 0.86.
#
# **Das Modell ist also positionsabhängig.** Wo die Würfelhälfte liegt,
# beeinflusst das Ergebnis, obwohl gleich viel Würfelevidenz vorhanden ist. Auf
# den ersten Blick widerspricht das dem Lehrbuchargument, CNNs seien
# translationsäquivariant. Der Widerspruch löst sich, wenn man die Architektur
# betrachtet: Vier Pooling-Stufen reduzieren $16^3$ auf $1^3$, und das
# anschließende `Flatten` mit `Dense`-Schicht verknüpft *feste* Positionen mit
# *festen* Gewichten. Die Äquivarianz der Faltung wird durch die
# Ausgabearchitektur wieder aufgegeben. Genau solche Einsichten sind der Grund,
# warum man kontrollierte Experimente macht: Diese Positionsabhängigkeit ist in
# keiner Genauigkeitszahl sichtbar.
#
# Ebenfalls auffällig ist, dass die Achse eine Rolle spielt: Der Schnitt entlang
# `axis=1` erzeugt die höchsten Würfel-Wahrscheinlichkeiten (0.80 und 0.86),
# `axis=0` und `axis=2` niedrigere. Ein isotropes Modell dürfte hier keinen
# Unterschied machen.
#
# ### Beobachtung 4: Das Differenzbild liefert nicht, was es sollte
#
# Die Erwartung war klar: Rot in der Würfelhälfte, Blau in der Kugelhälfte. Was
# man tatsächlich sieht, sind Bilder, die — abgesehen von den bekannten
# Randartefakten in den Schnitten 0 und 1 — **nahezu einfarbig blass** sind. Bei
# manchen Hybriden liegt ein durchgehend leicht rosafarbener, bei anderen ein
# leicht bläulicher Schleier über dem gesamten Volumen. Eine räumliche Trennung
# entlang der Schnittebene ist nicht zu erkennen.
#
# Dieses Negativergebnis ist lehrreich, und es hat zwei getrennte Ursachen:
#
# 1. **Methodisch:** Der gleichmäßige Farbstich ist ein Artefakt der Min-Max-Normalisierung.
#    Weil beide Karten vor der Subtraktion auf $[0,1]$ verschoben und *unabhängig*
#    voneinander skaliert werden, unterscheiden sich ihre Nullpunkte, und die
#    Differenz enthält einen konstanten Offset. Man betrachtet also zu einem
#    guten Teil einen Skalierungseffekt, nicht einen inhaltlichen Kontrast.
# 2. **Inhaltlich:** Die beiden Karten *sind* sich tatsächlich sehr ähnlich —
#    dieselbe Beobachtung wie in Abschnitt 7.1. Wenn `square`- und
#    `circle`-Erklärung nahezu identisch sind, ist ihre Differenz notwendigerweise
#    nahe null. Nach vier Pooling-Stufen bleiben nur 32 gemeinsam genutzte
#    Merkmale, aus denen alle Klassen lesen; eine klassenspezifische räumliche
#    Trennung ist bei dieser Architektur strukturell kaum möglich.
#
# ### Was wir aus dem Experiment mitnehmen
#
# Das Experiment hat funktioniert, auch wenn die Erklärungen nicht so hübsch
# geworden sind wie erhofft. Es hat drei konkrete Eigenschaften des Modells
# aufgedeckt: einen Würfel-Bias, eine Positionsabhängigkeit und einen mutmaßlichen
# Intensitäts-Shortcut für die Rauschklasse. Und es hat eine Grenze der Methode
# gezeigt: In dieser Architektur — mit vier Pooling-Stufen bis auf eine einzige
# räumliche Position — sind klassenspezifische Erklärungen kaum zu trennen. Wer
# diesen Aspekt untersuchen möchte, sollte weniger aggressiv poolen oder die
# Erklärung an einer *Zwischen*schicht abgreifen (siehe die Bemerkung zum
# `layer`-Parameter in Abschnitt 4).
#
# **Ein negatives Ergebnis ehrlich zu benennen, ist Teil der Methode.** Eine
# Erklärung, die nichts zeigt, ist eine Information über Modell und Methode — nicht
# ein Anlass, Parameter zu drehen, bis das erwartete Bild erscheint.

# %% [markdown]
# <a id="kapitel-9"></a>
# # 9. Alles in einer Abbildung
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ## Was diese Zelle berechnet
#
# Inhaltlich rechnet diese Zelle **exakt dasselbe** wie Abschnitt 8: dieselben
# Referenzobjekte, dieselben sechs Hybride, dieselben Vorhersagen und Erklärungen.
# Der Unterschied liegt ausschließlich in der Darstellung — und der ist praktisch
# bedeutsam.
#
# Statt viele einzelne Figuren zu erzeugen, wird **eine** Figur mit
#
# $$n_{\text{rows}} = \underbrace{2}_{\text{Referenzen}}
#   + \underbrace{6}_{\text{Hybride}} \times \underbrace{(1 + 3 + 1)}_{\text{Eingabe, 3 Erklärungen, Differenz}}
#   = 32$$
#
# Zeilen und 16 Spalten angelegt. Die Hilfsfunktion `plot_row` zeichnet eine Zeile
# und erhöht dabei einen Zeilenzähler; sie greift über `global row` auf diesen
# Zähler zu — ein pragmatisches, wenn auch nicht besonders elegantes Muster für
# Notebook-Code.
#
# ## Warum dieser Umweg?
#
# Aus dem Problem, auf das Abschnitt 7.1 am Ende hingewiesen hat: `fig.savefig()`
# kann nur die Figur speichern, auf die die Variable gerade zeigt. Werden in einer
# Schleife 30 Figuren erzeugt, landet nur die letzte in der Datei. Alles in eine
# Figur zu zeichnen, löst das — die vollständige Analyse liegt danach als eine
# einzige Datei `3_geometric_combinations_explanations.png` vor.
#
# Das ist keine kosmetische Frage. Reproduzierbare Forschung heißt unter anderem,
# dass Abbildungen **automatisch aus dem Code entstehen** und nicht per Screenshot
# aus einem Notebook gezogen werden. Nur so ist garantiert, dass die Abbildung in
# der Publikation tatsächlich zu dem Code gehört, der im Repository liegt. Der
# Parameter `dpi=150` sorgt zusätzlich dafür, dass die Auflösung für Druck und
# Präsentation ausreicht, und `bbox_inches='tight'` schneidet überflüssige Ränder
# ab.

# %%
from scipy.spatial.distance import euclidean

labels = encoder.categories_[0]

square = np.zeros((16, 16, 16, 1))
square[4:12, 4:12, 4:12, 0] = 1

circle = np.zeros((16, 16, 16, 1))
center = (8, 8, 8)
radius = 4
for i in range(16):
    for j in range(16):
        for k in range(16):
            if euclidean((i, j, k), center) <= radius:
                circle[i, j, k, 0] = 1

combinations = [
    np.concatenate([square[:8], circle[8:]], axis=0),
    np.concatenate([circle[:8], square[8:]], axis=0),
    np.concatenate([square[:, :8], circle[:, 8:]], axis=1),
    np.concatenate([circle[:, :8], square[:, 8:]], axis=1),
    np.concatenate([square[:, :, :8], circle[:, :, 8:]], axis=2),
    np.concatenate([circle[:, :, :8], square[:, :, 8:]], axis=2),
]

# Square + Circle + je Combination: Input + 3 Erklärungen + Differenz
n_rows = 2 + len(combinations) * (1 + 3 + 1)
fig, ax = plt.subplots(n_rows, shape, figsize=(16, 2 * n_rows))
row = 0

def plot_row(data, title, cmap='Greys_r', clim=None, colorbar=False):
    global row
    ax[row][shape // 2].set_title(title)
    im = None
    for j in range(shape):
        kwargs = {'cmap': cmap}
        if clim is not None:
            kwargs['clim'] = clim
        im = ax[row][j].imshow(data[j], **kwargs)
        ax[row][j].axis('off')
    if colorbar:
        fig.colorbar(im, ax=ax[row], fraction=0.015, pad=0.01)
    row += 1

plot_row(square, 'Square')
plot_row(circle, 'Circle')

for i, combo in enumerate(combinations):
    prediction = model.predict(np.expand_dims(combo, axis=0))[0]
    title = ' '.join([f'{labels[c]}: {prediction[c]:.2f}' for c in range(len(labels))])
    plot_row(combo, title)

    for classname in labels:
        explanation = explainers[classname].predict(np.expand_dims(combo, axis=0))[0]
        explanation = explanation / np.amax(np.abs(explanation))
        plot_row(explanation, f'{classname} explanation',
                 cmap='seismic', clim=(-1, 1), colorbar=True)

    square_explanation = explainers['square'].predict(np.expand_dims(combo, axis=0))[0]
    square_explanation = square_explanation - np.amin(square_explanation)
    square_explanation = square_explanation / np.amax(square_explanation)
    circle_explanation = explainers['circle'].predict(np.expand_dims(combo, axis=0))[0]
    circle_explanation = circle_explanation - np.amin(circle_explanation)
    circle_explanation = circle_explanation / np.amax(circle_explanation)
    absolute_difference = square_explanation - circle_explanation

    plot_row(absolute_difference, 'Square explanation - circle explanation',
             cmap='seismic', clim=(-1, 1), colorbar=True)

fig.savefig(target_dir / '3_geometric_combinations_explanations.png',
            bbox_inches='tight', dpi=150)
plt.show()
plt.close(fig)

# %% [markdown]
# <a id="kapitel-9-1"></a>
# ## 9.1 Interpretation: Die Gesamtabbildung
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Der Aufbau der Abbildung
#
# Die Abbildung ist 32 Zeilen hoch. Ihre Struktur — einmal verstanden, macht sie
# das Lesen sehr schnell:
#
# | Zeile | Inhalt |
# |---|---|
# | 1 | Referenzwürfel |
# | 2 | Referenzkugel |
# | 3 | Hybrid 1, Eingabe (Titel = Vorhersage) |
# | 4–6 | Hybrid 1, Erklärungen für `circle`, `noise`, `square` |
# | 7 | Hybrid 1, Differenzbild |
# | 8–12 | Hybrid 2, gleiche Reihenfolge |
# | … | … bis Hybrid 6 |
#
# Der große Vorteil dieser Anordnung gegenüber Abschnitt 8 ist das **vertikale
# Vergleichen**: Man kann eine Spalte — also eine feste Schnittebene $z$ — über
# alle 32 Zeilen hinweg mit dem Auge verfolgen. Genau das ist bei
# Einzelabbildungen, zwischen denen man scrollen muss, praktisch unmöglich, und
# genau so lassen sich systematische Muster von Einzelfällen unterscheiden.
#
# ### Was der Gesamtblick bestätigt
#
# Drei Muster, die man erst in dieser Zusammenschau sicher als *systematisch*
# einstufen kann:
#
# **Das Randartefakt betrifft ausnahmslos jede Erklärungszeile.** In den Spalten 0
# und 1 zeigt jede der 24 Erklärungszeilen dieselbe kräftige rot-blaue Struktur an
# derselben Stelle, unabhängig von Klasse, Hybrid und Schnittachse. Damit ist
# eindeutig belegt: Das Artefakt hängt an der **Architektur** und der
# **LRP-Konfiguration**, nicht am Bildinhalt. Wäre es inhaltlich bedingt, müsste es
# mit dem Eingabebild variieren.
#
# **Die Differenzzeilen sind durchgehend blass.** Jede siebte Zeile ist ein
# Differenzbild, und alle sechs sehen ähnlich aus: fast farblos, mit einem leichten
# gleichmäßigen Farbstich. Auch das bestätigt die Diagnose aus Abschnitt 8.1 —
# nicht ein einzelner Ausrutscher, sondern ein durchgängiges Verhalten.
#
# **Die relevanten Schnitte wandern mit dem Objekt.** Bei `axis=0`-Hybriden
# konzentriert sich Relevanz auf die Schnitte 4 bis 11, in denen überhaupt Objekt
# vorhanden ist, und die Schnitte 12 bis 15 bleiben nahezu weiß. Bei
# `axis=1`- und `axis=2`-Hybriden verteilt sie sich anders, weil dort in jedem
# Schnitt beide Formanteile vorkommen. Das Modell reagiert also auf die
# tatsächliche Objektgeometrie — die grundlegende Sanity-Prüfung ist bestanden.
#
# ### Warum man solche Übersichtsabbildungen erstellen sollte
#
# Ein einzelnes Erklärungsbild ist eine Anekdote. Erst die systematische
# Gegenüberstellung — dieselbe Analyse über mehrere Eingaben, mehrere Klassen und
# mehrere kontrollierte Variationen — erlaubt die Unterscheidung zwischen einem
# **Modelleigenschaft** und einem **Einzelfall**. Wer XAI-Ergebnisse anhand von
# zwei oder drei hübschen Heatmaps präsentiert, hat methodisch nichts gezeigt.
# Diese Zelle ist deshalb kein Darstellungs-Beiwerk, sondern der eigentliche
# Ergebnisteil des Notebooks.

# %% [markdown]
# <a id="kapitel-10"></a>
# # 10. Interaktive 3D-Darstellung
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ## Warum noch eine Darstellungsform?
#
# Schnittbildreihen sind präzise, aber sie stellen an den Betrachter eine
# anspruchsvolle Aufgabe: Man muss 16 flache Bilder im Kopf zu einem Körper
# zusammensetzen. Radiologen trainieren das über Jahre. Für alle anderen ist eine
# direkte räumliche Darstellung viel zugänglicher — und für die Frage „ist dieses
# abgeschnittene Objekt noch als Kugel erkennbar?“ auch schlicht besser geeignet.
#
# Diese Zelle rendert die Volumen deshalb als **interaktive 3D-Punktwolken** mit
# Plotly. Anders als bei Matplotlib-Grafiken kann man die Ansicht mit der Maus
# drehen, zoomen und verschieben.
#
# ## Was diese Zelle berechnet
#
# **`plot_volume_plotly`** zeichnet ein einzelnes Volumen. Der Kern ist die Zeile
#
# ```python
# z, y, x = np.where(vol > threshold)
# ```
#
# `np.where` liefert die Koordinaten aller Voxel oberhalb der Schwelle
# (`threshold=0.5`). Nur diese werden als Marker gezeichnet — ein wichtiger
# Effizienzgedanke: Von 4096 Voxeln sind bei einem Würfel nur einige hundert
# gesetzt, und der leere Raum muss nicht gerendert werden. Bei Rauschen greift die
# Schwelle anders: Rund die Hälfte aller Voxel liegt über 0.5, sodass etwa 2000
# Punkte gezeichnet werden — die Punktwolke füllt das gesamte Volumen.
#
# Die Reihenfolge der Rückgabewerte, `z, y, x`, ist bewusst so gewählt: NumPy
# indiziert Arrays in der Reihenfolge `(Tiefe, Höhe, Breite)`, während Plotly die
# Argumente `x, y, z` erwartet. Wer hier die Achsen verdreht, erhält eine
# gespiegelte oder transponierte Darstellung — eine klassische Fehlerquelle bei
# 3D-Daten.
#
# `aspectmode="cube"` erzwingt gleiche Achsenskalierung. Ohne diese Einstellung
# würde Plotly die Achsen an das Datenausmaß anpassen, und eine Kugel sähe wie ein
# Ellipsoid aus.
#
# **`plot_random_volumes_plotly`** zieht mit einem eigenen Zufallsgenerator
# (`np.random.default_rng(seed)`, dem modernen NumPy-Interface) zehn Stichproben
# und ordnet sie als $2 \times 5$-Raster interaktiver Teilbilder an. Die
# Klassennamen werden per `np.argmax(y[i])` aus der One-Hot-Kodierung
# zurückgewonnen — der Index der 1 im Vektor ist der Klassenindex — und über
# `class_names` in Text übersetzt.
#
# ## Was man in der Ausgabe sieht
#
# Zehn drehbare Teilbilder mit Titeln der Form `#123: circle`. Die Klassen sind
# hier deutlich unmittelbarer zu unterscheiden als in den Schnittbildern:
#
# - **Würfel** sind als kompakte Blöcke mit sichtbaren Kanten und Ecken erkennbar.
# - **Kugeln** erscheinen als runde Punktwolken. Jetzt wird auch klar sichtbar,
#   dass viele von ihnen **an einer oder mehreren Seiten flach abgeschnitten**
#   sind — das Randproblem aus Abschnitt 2.1, hier viel offensichtlicher als in
#   der Schnittdarstellung.
# - **Rauschen** ist eine gleichmäßig das ganze Volumen füllende Punktwolke ohne
#   erkennbare Form.
#
# Die Farbgebung (`colorscale="Viridis"`) kodiert den Voxelwert. Bei Würfeln und
# Kugeln sind alle Werte gleich 1, die Farbe ist daher konstant und trägt keine
# Information; bei Rauschen variiert sie, weil die Werte kontinuierlich sind.
#
# ## Zwei praktische Anmerkungen
#
# `fig.write_html(...)` speichert die Grafik als eigenständige, im Browser
# interaktive HTML-Datei — nützlich, um Ergebnisse mit Kollegen zu teilen, die
# keine Python-Umgebung haben. Die Dateigröße ist allerdings beträchtlich, weil
# alle Punktkoordinaten mitgeschrieben werden. Die auskommentierte Zeile
# `fig.write_image(...)` würde stattdessen ein statisches PNG erzeugen; sie
# benötigt das zusätzliche Paket `kaleido` und ist deshalb deaktiviert.
#
# `fig.show(renderer="notebook")` erzwingt den Notebook-Renderer. Ohne diese
# explizite Angabe bleibt die Grafik in manchen Umgebungen — insbesondere in
# exportierten HTML-Dateien — leer.

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot_volume_plotly(vol: np.ndarray, title: str = "", threshold: float = 0.5):
    vol = np.asarray(vol)
    if vol.ndim == 4:
        vol = vol[..., 0]
    z, y, x = np.where(vol > threshold)
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=4, opacity=0.8, color=vol[z, y, x], colorscale="Viridis"),
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="z",
            aspectmode="cube",
            xaxis=dict(range=[0, vol.shape[2]]),
            yaxis=dict(range=[0, vol.shape[1]]),
            zaxis=dict(range=[0, vol.shape[0]]),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_random_volumes_plotly(
    X: np.ndarray,
    y: np.ndarray,
    n: int = 10,
    threshold: float = 0.5,
    seed: int = 42,
    class_names=None,
):
    """n zufällige 3D-Volumen als eine interaktive Plotly-Figur."""
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(X), size=min(n, len(X)), replace=False)

    titles = []
    for i in idxs:
        if y.ndim == 2:
            cls = class_names[np.argmax(y[i])] if class_names is not None else int(np.argmax(y[i]))
        else:
            cls = y[i]
        titles.append(f"#{i}: {cls}")

    n_plots = len(idxs)
    n_cols = 5
    n_rows = int(np.ceil(n_plots / n_cols))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)],
        subplot_titles=titles,
    )

    for k, i in enumerate(idxs):
        row, col = divmod(k, n_cols)
        vol = np.asarray(X[i])
        if vol.ndim == 4:
            vol = vol[..., 0]
        z, yy, x = np.where(vol > threshold)
        fig.add_trace(
            go.Scatter3d(
                x=x, y=yy, z=z,
                mode="markers",
                marker=dict(size=3, opacity=0.85, color=vol[z, yy, x], colorscale="Viridis"),
                showlegend=False,
                name=titles[k],
            ),
            row=row + 1,
            col=col + 1,
        )

    d, h, w = X.shape[1], X.shape[2], X.shape[3]
    scene_layout = dict(
        aspectmode="cube",
        xaxis=dict(range=[0, w], title="x"),
        yaxis=dict(range=[0, h], title="y"),
        zaxis=dict(range=[0, d], title="z"),
    )
    layout_scenes = {f"scene{k+1}" if k else "scene": scene_layout for k in range(n_plots)}
    fig.update_layout(
        title=f"{n_plots} zufällige 3D-Samples",
        height=400 * n_rows,
        width=1200,
        margin=dict(l=0, r=0, t=60, b=0),
        **layout_scenes,
    )
    return fig


fig = plot_random_volumes_plotly(
    X, y, n=10, seed=42, class_names=encoder.categories_[0]
)
fig.write_html(target_dir / "4_geometric_3d_samples.html")
#fig.write_image(target_dir / "4_geometric_3d_samples.png", scale=2)
fig.show(renderer="notebook")

# %% [markdown]
# <a id="kapitel-11"></a>
# # 11. Fazit und weiterführende Schritte
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ## Der Weg, den wir gegangen sind
#
# Das Notebook hat eine vollständige XAI-Pipeline im Kleinen durchlaufen:
#
# 1. **Daten mit bekannter Ground Truth erzeugen** — 600 Volumen aus drei Klassen,
#    bei denen wir voxelgenau wissen, wo das Objekt liegt.
# 2. **Ein Modell trainieren** — ein 3D-CNN mit rund 46.000 Parametern, klein
#    genug, um es zu durchschauen.
# 3. **Erklärungen berechnen** — LRP mit einer schichtabhängigen Regelstrategie,
#    ein Erklärer pro Klasse.
# 4. **Erklärungen experimentell prüfen** — mit konstruierten Hybrid-Objekten,
#    für die wir eine überprüfbare Erwartung formulieren konnten.
#
# ## Was wir über das Modell gelernt haben
#
# | Befund | Belegt durch |
# |---|---|
# | Das Modell reagiert auf die Objektregion, nicht auf leeren Raum | Relevanz erscheint nur in Schnitten, die Objekt enthalten (7.1, 9.1) |
# | Es hat einen deutlichen **Würfel-Bias** | `square` gewinnt alle sechs Hybrid-Fälle mit 0.59 bis 0.86 (8.1) |
# | Es ist **positionsabhängig**, obwohl es ein CNN ist | Vertauschen der Hälften ändert die Vorhersage (8.1) |
# | Es erkennt `noise` vermutlich über einen **Intensitäts-Shortcut** | `noise: 0.00` bei allen unbekannten Eingaben (8.1) |
#
# Keiner dieser vier Befunde wäre aus einer Genauigkeitszahl ersichtlich gewesen.
# Das ist der Kern der Botschaft: **Metriken sagen, wie gut ein Modell ist;
# Erklärungen sagen, warum — und ob man ihm trauen darf.**
#
# ## Was wir über die Methode gelernt haben
#
# Genauso wichtig sind die Grenzen, die sichtbar wurden:
#
# - **Architektur bedingte Artefakte.** Zero-Padding und vier Pooling-Stufen bis
#   auf $1^3$ erzeugen einen Randeffekt, der die Farbskala dominiert. Wer solche
#   Artefakte nicht erkennt, interpretiert Rechenrauschen als Modellverhalten.
# - **Begrenzte Klassensensitivität.** Die Erklärungen für `circle`, `noise` und
#   `square` sind sich sehr ähnlich, weil alle drei Klassen aus denselben 32
#   Merkmalen lesen. Kontrastive Differenzbilder liefern deshalb hier fast nichts.
# - **Darstellungsentscheidungen verändern die Aussage.** Normalisierung,
#   Colormap und Wertebereich sind keine Kosmetik — dieselben Zahlen können
#   überzeugend oder nichtssagend aussehen. Jede Heatmap sollte mit der Angabe
#   ihrer Normalisierung präsentiert werden.
#
# ## Bekannte Schwächen dieses Notebooks
#
# Zwei Punkte wurden bewusst nicht korrigiert, damit der Code unverändert bleibt,
# und sollten in eigenen Projekten anders gemacht werden:
#
# - **Trainings- und Testmenge sind identisch** (`X[:300]` für beide, siehe 2.2).
#   Alle Validierungszahlen sind damit ohne Aussagekraft.
# - **Nur die letzte Figur einer Schleife wird gespeichert** (siehe 7.1); die
#   Lösung zeigt Abschnitt 9.
#
# ## Wie es weitergeht
#
# Naheliegende Experimente, um das Gelernte zu vertiefen:
#
# - **An der LRP-Strategie drehen.** Ersetzen Sie in der Strategie aus Abschnitt 5
#   $\alpha=2,\beta=1$ durch $\alpha=1,\beta=0$ (nur positive Beiträge) und
#   vergleichen Sie die Karten. Die Beschränkung $\alpha = \beta + 1$ ist dabei
#   einzuhalten.
# - **Weniger aggressiv poolen.** Lassen Sie eine `MaxPooling3D`-Stufe weg, sodass
#   vor dem `Flatten` ein $2^3$-Raster statt einer einzigen Position übrig bleibt.
#   Erwartung: räumlich schärfere und klassenspezifischere Erklärungen.
# - **Die Shortcut-Hypothese testen.** Skalieren Sie das Rauschen so, dass seine
#   mittlere Intensität der von Würfeln und Kugeln entspricht. Bleibt die
#   Rauschklasse dann noch perfekt trennbar?
# - **Eine Zwischenschicht erklären.** Setzen Sie `layer` auf einen kleineren Wert
#   als 19 und untersuchen Sie, was das Netz auf mittlerer Abstraktionsebene
#   repräsentiert.
# - **Mit anderen Methoden vergleichen.** Berechnen Sie für dieselben Bilder eine
#   einfache Gradientenkarte. Wenn zwei unabhängige Methoden dieselbe Region
#   hervorheben, steigt das Vertrauen in beide erheblich.
#
# ## Weiterführende Notebooks in diesem Repository
#
# Mit dem hier gewonnenen Verständnis lassen sich die realistischeren Beispiele
# gut einordnen — dort fehlt allerdings die Ground Truth, die dieses Notebook so
# wertvoll macht:
#
# - `Train_and_explain_3D_mnist_model` — handgeschriebene Ziffern als Volumen,
#   derselbe Aufbau mit echten Daten.
# - `Explain_2D_VGG_predictions` — LRP auf einem großen, vortrainierten Netz für
#   natürliche Bilder.
# - `Explain_brain_age_predictions` — der Anwendungsfall aus der Medizin:
#   Altersvorhersage aus MRT-Aufnahmen. Hier ist Interpretierbarkeit kein
#   akademisches Anliegen mehr, sondern Voraussetzung für klinische Akzeptanz.
#
# ## Literatur
#
# - Bach, S. et al. (2015): *On Pixel-Wise Explanations for Non-Linear Classifier
#   Decisions by Layer-Wise Relevance Propagation.* PLOS ONE 10(7). — Die
#   Originalarbeit zu LRP.
# - Montavon, G. et al. (2019): *Layer-Wise Relevance Propagation: An Overview.*
#   In: Explainable AI, Springer LNCS 11700. — Praktische Empfehlungen zur
#   Regelwahl pro Schicht, die Grundlage der `LRPStrategy` dieses Notebooks.
# - Lapuschkin, S. et al. (2019): *Unmasking Clever Hans Predictors and Assessing
#   What Machines Really Learn.* Nature Communications 10. — Woher der Begriff
#   Clever Hans stammt und warum XAI kein Luxus ist.
# - Adebayo, J. et al. (2018): *Sanity Checks for Saliency Maps.* NeurIPS. — Warum
#   man Erklärungsmethoden selbst validieren muss.
