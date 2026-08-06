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
# # Synthetische Gehirne, 3D-CNNs und LRP
#
# ### Ein kommentiertes Notebook für Deep-Learning-Einsteiger
#
# **Worum geht es hier?**
#
# Dieses Notebook baut ein komplettes Experiment in vier Schritten:
#
# 1. Es **erzeugt** einen künstlichen 3D-Datensatz: kugelförmige "Gehirne", in die
#    Tunnel gebohrt werden. Die Tunnelbreite ist bekannt und dient als Zielwert.
# 2. Es **trainiert** ein 3D-CNN (Convolutional Neural Network), das aus dem
#    Volumen die Tunnelbreite vorhersagen soll — eine *Regression*, keine
#    Klassifikation.
# 3. Es **erklärt** die Vorhersagen mit *Layer-wise Relevance Propagation* (LRP):
#    Welche Voxel haben die Vorhersage nach oben, welche nach unten getrieben?
# 4. Es **prüft** die Erklärungen mit kausalen Eingriffen (mehr Tunnel bohren,
#    Tunnel verbreitern) und schaut in die interne Repräsentation des Netzes.
#
# **Warum synthetische Daten? Der größere Kontext.**
#
# Das eigentliche Ziel des Projekts (siehe die Schwester-Notebooks
# `Explain_brain_age_predictions`) ist *Brain-Age-Prediction*: Aus einem echten
# MRT-Volumen soll ein Netz das Alter einer Person schätzen. Dort stellt sich
# sofort die Vertrauensfrage — schaut das Netz auf plausible anatomische
# Strukturen, oder auf Artefakte des Scanners?
#
# Das Problem: bei echten MRT-Daten kennen wir die "richtige" Erklärung nicht.
# Wir können also nie sagen, ob eine schöne Heatmap tatsächlich korrekt ist.
# Deshalb der Trick dieses Notebooks: Wir bauen ein Spielzeugproblem, bei dem wir
# die Ground Truth der Erklärung **per Konstruktion kennen** — der Zielwert hängt
# ausschließlich von der Tunnelbreite ab. Eine gute Erklärungsmethode *muss* also
# die Tunnelränder markieren und nicht etwa den Bildhintergrund. Damit wird die
# XAI-Methode selbst validierbar. Dieses Vorgehen heißt in der Literatur
# *sanity check* bzw. *ground-truth-based evaluation of attribution methods*.
#
# > **Wichtiger Hinweis zu diesem Notebook-Durchlauf:** Beide Modelle wurden hier
# > mit `epochs=2` trainiert (im Original: 500). Das ist eine Debug-Einstellung,
# > damit der Durchlauf Minuten statt Stunden dauert. Die Modelle sind deshalb
# > **massiv untertrainiert**, und alle Heatmaps zeigen ein Netz, das die Aufgabe
# > noch kaum gelöst hat. Für die didaktische Frage "*wie liest man solche
# > Bilder?*" ist das kein Problem — man muss die Ergebnisse nur ehrlich als
# > "Zwischenstand" lesen. An den entsprechenden Stellen unten steht jeweils
# > dabei, was man bei einem austrainierten Modell erwarten würde.
#
# ---
#
# ## Inhaltsverzeichnis
#
# | # | Abschnitt | Thema |
# |---|-----------|-------|
# | 1 | [Datengenerierung: synthetische Gehirne mit Tunneln](#sec1) | Daten |
# | 2 | [Zwischenstopp: der Kernel lebt noch](#sec2) | Daten |
# | 3 | [3D-Visualisierung: Volume Rendering](#sec3) | Daten |
# | 4 | [3D-Visualisierung: Isosurface](#sec4) | Daten |
# | 5 | [Aufteilung in Trainings-, Validierungs- und Testdaten](#sec5) | Daten |
# | 6 | [Noch ein Zwischenstopp](#sec6) | Daten |
# | 7 | [Modell 1: 3D-CNN mit Global Average Pooling](#sec7) | Modell |
# | 8 | [Training und Lernkurven](#sec8) | Modell |
# | 9 | [Wie gut sagt das Modell vorher?](#sec9) | Modell |
# | 10 | [LRP: Grundidee und erste Heatmaps](#sec10) | XAI |
# | 11 | [Kausaler Test 1: immer mehr Tunnel bohren](#sec11) | XAI |
# | 12 | [Kausaler Test 2: breite vs. schmale Tunnel](#sec12) | XAI |
# | 13 | [LRP über das gesamte Volumen](#sec13) | XAI |
# | 14 | [Der Bottleneck als Repräsentation](#sec14) | XAI |
# | 15 | [Kontrastive Erklärungen mit RestructuredLRP](#sec15) | XAI |
# | 16 | [Neuronenweise LRP im Bottleneck](#sec16) | XAI |
# | 17 | [Wie redundant sind die 32 Features?](#sec17) | XAI |
# | 18 | [Modell 2: ohne GAP, mit fixiertem Bias](#sec18) | Modell |
# | 19 | [Training von Modell 2](#sec19) | Modell |
# | 20 | [LRP für Modell 2](#sec20) | XAI |
# | 21 | [Neuronenweise LRP für Modell 2](#sec21) | XAI |
# | 22 | [Korrelationen in Modell 2 und tote ReLUs](#sec22) | XAI |
# | 23 | [Reproduzierbarkeitscheck](#sec23) | XAI |
# | 24 | [Referenz-Encodings für Modell 2](#sec24) | XAI |
# | 25 | [Fazit und Ausblick](#sec25) | Zusammenfassung |
#
# ---
#
# ### Vokabelliste für den Einstieg
#
# | Begriff | Bedeutung in diesem Notebook |
# |---------|------------------------------|
# | **Voxel** | ein Bildpunkt in 3D (das 3D-Äquivalent zum Pixel) |
# | **Volumen** | ein Datenpunkt: ein $32\times32\times32$-Würfel mit einem Grauwert pro Voxel |
# | **Regression** | das Netz gibt eine *Zahl* aus (hier: Tunnelbreite 1–10), keine Klasse |
# | **Conv3D** | Faltungsschicht, die mit einem $3\times3\times3$-Filter über das Volumen fährt |
# | **Pooling** | Verkleinern der Auflösung (hier: von $32^3$ auf $16^3$ auf $8^3$ auf $4^3$) |
# | **Bottleneck** | die schmalste Schicht vor der Ausgabe (hier: 32 Neuronen) — die "Zusammenfassung" des Volumens |
# | **Relevanz / Heatmap** | pro Voxel eine Zahl: wie stark hat dieser Voxel die Vorhersage beeinflusst |
# | **LRP** | Layer-wise Relevance Propagation: das Erklärungsverfahren dieses Repos |

# %% [markdown]
# <a id="sec1"></a>
# ## 1. Datengenerierung: synthetische Gehirne mit Tunneln
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Zuerst: die technische Vorbereitung
#
# Der obere Teil der Zelle hat noch nichts mit Daten zu tun, sondern richtet die
# Umgebung ein. Drei Dinge passieren dort:
#
# 1. **`find_repo_root()`** sucht vom aktuellen Arbeitsverzeichnis aus nach oben,
#    bis eine `pyproject.toml` oder ein `explainability`-Ordner auftaucht, und legt
#    diesen Pfad in `sys.path`. Das ist der Grund, warum `from explainability import
#    LRP` funktioniert, egal aus welchem Verzeichnis das Notebook gestartet wurde.
# 2. **`find_notebook_name()`** ermittelt den Namen des laufenden Notebooks. Das ist
#    überraschend fummelig, weil Jupyter diese Information nicht standardisiert
#    bereitstellt: Die Funktion probiert der Reihe nach `ipynbname`, dann
#    Umgebungsvariablen (`JPY_SESSION_NAME`, `__session__`), dann `__file__` (falls
#    als Skript ausgeführt) und schließlich die Kommandozeilen der Elternprozesse
#    über `/proc` (für `nbconvert`/`quarto`). Wer den Automatismus umgehen will,
#    setzt die Umgebungsvariable `NOTEBOOK_NAME`.
# 3. **`target_dir`** wird daraus gebaut: `output/notebooks/<notebook-name>`. Dort
#    landen später die trainierten Modelle (siehe Abschnitt 8). So schreibt jedes
#    Notebook in seinen eigenen Ausgabeordner und Ergebnisse verschiedener
#    Experimente vermischen sich nicht.
#
# `np.random.seed(42)` fixiert den Zufallsgenerator. Damit ist der ganze Datensatz
# **reproduzierbar** — bei jedem Ausführen entstehen dieselben Gehirne. Für
# Experimente ist das unverzichtbar, weil man sonst nicht unterscheiden kann, ob ein
# Unterschied vom geänderten Code oder vom Zufall kommt.
#
# ### Was passiert in dieser Zelle?
#
# Die Zelle erzeugt den kompletten Datensatz: `N = 1000` Volumen der Größe
# $32\times32\times32\times1$ (die letzte `1` ist der Farbkanal, analog zu einem
# Graustufenbild). Der Ablauf pro Volumen:
#
# **Schritt 1 — die Kugel ("Gehirn").** `create_brain` legt eine Vollkugel in den
# Würfel. Mittelpunkt $c$ und Radius $r$ werden leicht zufällig gewählt:
#
# $$c \sim \mathcal{U}\{14,\dots,17\}^3, \qquad r \sim \mathcal{U}\{10,\dots,13\}$$
#
# Ein Voxel an Position $p$ gehört zum Gewebe, wenn $\lVert p - c\rVert_2 \le r$.
# Alle Gewebe-Voxel bekommen einen **zufälligen** Grauwert
# $\mathcal{U}(0.25,\,1)$, alles außerhalb bleibt $0$. Dieses Rauschen ist
# Absicht: es imitiert die Textur echter MRT-Bilder und verhindert, dass das Netz
# eine triviale Abkürzung über konstante Intensitäten findet.
#
# **Schritt 2 — die Tunnel.** `drill` bohrt `NUM_TUNNELS = 6` Gänge. Jeder Tunnel
# startet an einem zufälligen Punkt der Kugeloberfläche und läuft grob Richtung
# Mittelpunkt. Die Laufrichtung wird bei jedem Schritt mit
# $\mathcal{N}(\mu,\,|\mu|/3)$ verrauscht — daraus entstehen die gekrümmten,
# unregelmäßigen Gänge statt gerader Bohrlöcher. An jeder Position wird eine
# kleine Kugel ("pocket") ausgeschnitten, also auf $0$ gesetzt, mit Radius
#
# $$\rho \sim \mathcal{U}\big(\lfloor w/2 \rfloor,\, 1\big)$$
#
# wobei $w$ das Label des Volumens ist. **Das ist der entscheidende Punkt:** je
# größer das Label $w$, desto größer die ausgeschnittenen Kugeln, desto breiter
# die Tunnel. Da das entfernte Volumen ungefähr mit $\rho^3$ skaliert, wächst der
# Gewebeverlust stark überproportional mit dem Label.
#
# **Schritt 3 — die Labels.** `y = np.random.randint(1, MAX_RADIUS + 1, N)` zieht
# für jedes Volumen ein Label aus $\{1,\dots,10\}$ gleichverteilt. Das Netz soll
# später genau dieses $w$ zurückrechnen.
#
# ### Was ist die "Aufgabe" für das Netz?
#
# Gesucht ist die Funktion $f: \mathbb{R}^{32\times32\times32} \to \mathbb{R}$ mit
# $f(X) \approx w$. Wichtig ist, was *nicht* zählt: Position der Tunnel, Anzahl
# der Tunnel (immer 6), Kugelgröße, Grauwertrauschen. Das sind alles
# **Störfaktoren** (nuisance variables), gegen die das Netz invariant sein muss.
# Nur die *lokale Breite* der Gänge trägt Information. Genau diese Trennung macht
# das Problem zu einem guten Prüfstand für Erklärungsmethoden.
#
# ### Die Grafik: 10 Zeilen × 8 Spalten
#
# Am Ende der Zelle wird ein Gitter geplottet. Die Logik ist wichtig, um das Bild
# zu lesen:
#
# * **Zeile $i$** = das *erste* Volumen im Datensatz mit Label $y = i$, also
#   Zeile 1 = Tunnelbreite 1, Zeile 10 = Tunnelbreite 10.
# * **Spalte $j$** = ein 2D-Schnitt (Slice) durch dasselbe Volumen, nämlich die
#   Ebenen $z = 12 \dots 19$ — grob die Mitte des Würfels.
#
# Man schaut also **nicht** auf 80 verschiedene Gehirne, sondern auf 10 Gehirne in
# je 8 Schnittbildern. Das ist die übliche Darstellung in der Bildgebung: ein 3D-
# Volumen wird als Stapel von 2D-Schichten gezeigt.
#
# ### Interpretation
#
# * **Zeile 1 (Breite 1):** nahezu vollständige, gesprenkelte Scheiben. Die Tunnel
#   sind so schmal, dass sie in einem Schnitt oft nur als einzelne schwarze Punkte
#   oder feine Kratzer erscheinen (im Bild z. B. das kleine Kreuz in Spalte 6).
# * **Zeilen 2–5:** deutlich erkennbare schwarze Löcher und Kanäle, die von außen
#   nach innen laufen. Da ein Tunnel dreidimensional gekrümmt ist, sieht man
#   denselben Gang in aufeinanderfolgenden Schnitten wandern — er "bewegt" sich
#   von Spalte zu Spalte.
# * **Zeilen 6–10:** die Kugel wird regelrecht ausgehöhlt. In Zeile 6 und Zeile 10
#   bleiben nur noch Sicheln und Fragmente übrig; in Zeile 10 ist von der Scheibe
#   fast nichts mehr da.
# * **Störvariation sichtbar machen:** Vergleicht man Zeile 5 und Zeile 8, so ist
#   der Gesamtdurchmesser der Scheibe unterschiedlich, obwohl die Labels
#   verschieden sind — der Kugelradius variiert unabhängig vom Label. Wer nur die
#   sichtbare Gewebefläche zählt, kann sich also täuschen. Das Netz muss lernen,
#   *Loch-Breite* von *Kugelgröße* zu unterscheiden.
#
# ### Der größere Kontext
#
# Dieser Datensatz ist ein bewusst vereinfachtes Analogon zur Atrophie im
# alternden Gehirn: mit dem Alter weiten sich Sulci und Ventrikel, es geht
# Gewebe verloren. "Tunnelbreite" spielt hier die Rolle von "Alter". Wenn eine
# XAI-Methode auf diesem Spielzeugproblem versagt, braucht man sie auf echten
# MRT-Daten gar nicht erst zu probieren.

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

import ipynbname
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from typing import Any

from explainability import LRP, LRPStrategy


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

np.random.seed(42)

IMAGE_SIZE = 32
NUM_TUNNELS = 6
MAX_RADIUS = 10
#N = 10
N = 1000
#N = 100

def key(x: Any):
    if isinstance(x, tuple):
        return f'{int(x[0][0])}-{int(x[1][0])}-{int(x[2][0])}'
    else:
        return f'{int(x[0])}-{int(x[1])}-{int(x[2])}'

def drill(brain: np.ndarray, surface: np.ndarray, center: np.ndarray, width: float, 
          inside_keys: set, idx: np.ndarray) -> np.ndarray:
    current_idx = np.random.choice(np.arange(len(surface)))
    current = surface[current_idx]
    direction = center - current
    direction = direction / np.sum(np.abs(direction))
    current_idx = tuple(np.expand_dims(current+direction, -1).astype(int))
    
    while key(current_idx) in inside_keys:
        vertex_radius = np.random.uniform(width // 2, 1)
        vertex_distances = euclidean_distances(idx, np.asarray(current_idx).reshape(1, 3))[:,0]
        pocket = vertex_distances <= vertex_radius
        brain[tuple(idx[pocket].T)] = 0

        next = current + direction
        direction = next - current
        direction[0] = np.random.normal(direction[0], np.abs(direction[0] / 3))
        direction[1] = np.random.normal(direction[0], np.abs(direction[1] / 3))
        direction[2] = np.random.normal(direction[0], np.abs(direction[2] / 3))
        direction = direction / np.sum(np.abs(direction))
        current = next
        current_idx = tuple(np.expand_dims(current, -1).astype(int))
        
    return brain

def create_brain(size: int, width: int, num_tunnels: int = 1):
    brain = np.zeros((size, size, size, 1))
    
    center = np.random.randint(7 * size//16, 9*size//16, 3)
    radius = np.random.randint(size//2-6, size//2-2)
    
    idx = np.asarray(np.meshgrid(*[np.arange(size) for _ in range(3)])).T.reshape(-1, 3)
    distances = euclidean_distances(idx, center.reshape(1, -1))[:,0]
    inside = distances <= radius
    surface = np.isclose(distances, radius, atol=1e-1)
    surface = idx[surface]
    
    brain[tuple(idx[inside].T)] = np.random.uniform(0.25, 1, (len(idx[inside]), 1))
    brain[tuple(idx[surface].T)] = np.random.uniform(0.25, 1, (len(idx[surface]), 1))
    
    inside_keys = set([key(x) for x in idx[inside]]) | set([key(x) for x in surface])
    
    for _ in range(num_tunnels):
        drill(brain, surface, center, width, inside_keys, idx)
    
    return brain

X = []
y = np.random.randint(1, MAX_RADIUS + 1, N)
print("y: ", y)

for i in range(len(y)):
    X.append(create_brain(IMAGE_SIZE, width=y[i], num_tunnels=NUM_TUNNELS))
    print(f'{i+1}/{N}')

    
fig, ax = plt.subplots(10, 8, figsize=(15, 15))

for i in range(1, MAX_RADIUS + 1):
    print("i: ", i)
    idx = np.where(y == i)[0][0]
    #idx = np.where(y == i)[0]
    print("idx: ", idx)
    
    for j in range(8):
        ax[i-1][j].imshow(X[idx][12+j], cmap='Greys_r')
        ax[i-1][j].axis('off')
        
plt.show()

# %% [markdown]
# <a id="sec2"></a>
# ## 2. Zwischenstopp: der Kernel lebt noch
#
# [↑ Inhaltsverzeichnis](#toc)
#
# `1+1` ist keine inhaltliche Berechnung, sondern ein in der Praxis sehr üblicher
# **Marker**. Die Zelle davor läuft je nach `N` mehrere Minuten. Eine triviale
# Zelle danach dient als Kontrollpunkt:
#
# * Erscheint die Ausgabe `2`, ist der Kernel noch am Leben und die teure Zelle
#   vollständig durchgelaufen.
# * Sie zerlegt das Notebook in Blöcke, die man einzeln erneut ausführen kann,
#   ohne die Datengenerierung zu wiederholen (die Daten liegen im Kernel-Speicher).
#
# Beim Aufräumen eines Notebooks würde man solche Zellen löschen — hier bleiben
# sie, weil der Python-Code laut Aufgabenstellung unverändert bleibt.

# %%
1+1

# %% [markdown]
# <a id="sec3"></a>
# ## 3. 3D-Visualisierung: Volume Rendering
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert in dieser Zelle?
#
# Die Schnittbilder oben zeigen immer nur eine Ebene. Um zu verstehen, dass die
# Tunnel tatsächlich zusammenhängende 3D-Kanäle sind, wird hier ein einzelnes
# Gehirn (`width=5`, 6 Tunnel) neu erzeugt und mit Plotly als drehbares Volumen
# dargestellt.
#
# Technisch werden dazu die Voxel in eine lange Liste von Koordinaten-Wert-Tupeln
# überführt: `np.mgrid` erzeugt für jeden der $32^3 = 32\,768$ Voxel seine
# $(x,y,z)$-Position, `vol.flatten()` liefert den zugehörigen Grauwert. Plotly
# rendert daraus halbtransparente Isoflächen.
#
# Die wichtigen Parameter:
#
# * `isomin=0.25` — die Schwelle. Gewebe hat per Konstruktion Werte $\ge 0.25$,
#   Tunnel und Außenraum haben exakt $0$. Alles unterhalb der Schwelle wird also
#   ausgeblendet, und die Tunnel werden zu *sichtbaren Hohlräumen*.
# * `opacity=0.15` — starke Transparenz, damit man in die Kugel hineinschauen kann.
# * `surface_count=12` — Anzahl der gerenderten Schalen; mehr Schalen wirken
#   "wolkiger", weniger wirken kantiger.
#
# Der Kommentar im Code (`nicht y/x/z — y sind die Regressions-Labels!`) weist auf
# eine echte Falle hin: die Variable `y` enthält im Notebook die **Labels**. Würde
# man sie hier als Achsen-Variable wiederverwenden, wären die Zielwerte
# überschrieben und das anschließende Training würde stillschweigend Unsinn lernen.
#
# ### Die Grafik
#
# Zu sehen ist eine graue, milchig-transparente Kugel. Weil das Rendering
# interaktiv ist (Maus ziehen = drehen, Scrollen = zoomen), lohnt es sich, sie zu
# kippen: dann erkennt man die Tunnel als dunkle Einbuchtungen, die von der
# Oberfläche ins Innere ziehen — genau die Struktur, deren *Breite* das Netz
# später schätzen soll. Diese Ansicht ist reine Qualitätskontrolle der Daten und
# geht nicht ins Modell ein.

# %%
import numpy as np
import plotly.graph_objects as go

# Ein synthetisches Gehirn erzeugen (wie im Notebook)
np.random.seed(42)
brain = create_brain(size=32, width=5, num_tunnels=6)
vol = brain[..., 0]  # Shape: (32, 32, 32)

# Voxel-Koordinaten und Werte (nicht y/x/z — y sind die Regressions-Labels!)
grid_z, grid_y, grid_x = np.mgrid[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]

fig = go.Figure(
    data=go.Volume(
        x=grid_x.flatten(),
        y=grid_y.flatten(),
        z=grid_z.flatten(),
        value=vol.flatten(),
        isomin=0.25,          # Schwelle: nur „Gewebe“, Tunnel (0) weg
        isomax=1.0,
        opacity=0.15,         # etwas transparent → Tunnel sichtbar
        surface_count=12,
        colorscale="Greys",
        caps=dict(x_show=False, y_show=False, z_show=False),
    )
)

fig.update_layout(
    title="Synthetisches Gehirn (drehbar)",
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Z",
        aspectmode="data",
    ),
    width=700,
    height=700,
    margin=dict(l=0, r=0, t=40, b=0),
)

fig.show()  # im Notebook: Maus ziehen zum Drehen, Scroll zum Zoomen

# %% [markdown]
# <a id="sec4"></a>
# ## 4. 3D-Visualisierung: Isosurface
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier — und wo ist der Unterschied zur Zelle davor?
#
# Beide Zellen zeigen dasselbe Volumen, aber mit unterschiedlicher Renderstrategie:
#
# | | `go.Volume` (Abschnitt 3) | `go.Isosurface` (hier) |
# |---|---|---|
# | Prinzip | mehrere gestapelte, transparente Schalen | **eine** geschlossene Oberfläche |
# | `surface_count` | 12 | 1 |
# | `opacity` | 0.15 | 0.7 |
# | Eindruck | wolkig, Blick ins Innere | feste Oberfläche, klare Kanten |
#
# Mathematisch ist eine Isofläche die Punktmenge
#
# $$\{p \in \mathbb{R}^3 \;:\; V(p) = \tau\}$$
#
# also alle Punkte mit demselben Wert $\tau$ (hier $\tau = 0.25$, die
# Gewebeschwelle). Das ist dasselbe Prinzip wie Höhenlinien auf einer Landkarte,
# nur eine Dimension höher. `caps=dict(...=False)` schaltet die Deckflächen an den
# Rändern des Würfels ab — sonst würde die Kugel dort "zugeklebt" aussehen, wo sie
# den Bildrand berührt.
#
# ### Die Grafik
#
# Eine kompakte graue Kugel mit einer erkennbar rauen Oberfläche. Die Rauheit ist
# kein Rendering-Artefakt, sondern Folge des Grauwertrauschens: einzelne
# Randvoxel liegen knapp über oder unter der Schwelle. Die Tunnelöffnungen
# erscheinen als dunkle Krater bzw. Einschnitte. Diese Darstellung ist die
# anschaulichste Antwort auf die Frage "*wie sieht ein Datenpunkt eigentlich
# aus?*" — ein Blick, den man sich vor jedem Training nehmen sollte.

# %%
fig = go.Figure(
    data=go.Isosurface(
        x=grid_x.flatten(),
        y=grid_y.flatten(),
        z=grid_z.flatten(),
        value=vol.flatten(),
        isomin=0.25,
        isomax=1.0,
        surface_count=1,
        colorscale="Greys",
        opacity=0.7,
        caps=dict(x_show=False, y_show=False, z_show=False),
    )
)
fig.update_layout(scene_aspectmode="data", width=700, height=700)
fig.show()

# %% [markdown]
# <a id="sec5"></a>
# ## 5. Aufteilung in Trainings-, Validierungs- und Testdaten
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert in dieser Zelle?
#
# Aus der Python-Liste `X` wird ein NumPy-Array mit der endgültigen Form
# $(1000, 32, 32, 32, 1)$, die Labels werden zu $(1000, 1)$ umgeformt (Keras
# erwartet für eine Regression eine Spalte, nicht einen flachen Vektor). Danach
# wird sequenziell in drei Blöcke geschnitten:
#
# $$\underbrace{60\,\%}_{\text{train} = 600} \;\big|\;
#   \underbrace{20\,\%}_{\text{val} = 200} \;\big|\;
#   \underbrace{20\,\%}_{\text{test} = 200}$$
#
# ### Warum drei Mengen und nicht zwei?
#
# Das ist eines der wichtigsten Konzepte im Deep Learning:
#
# * **Training** — daraus lernt das Netz seine Gewichte.
# * **Validierung** — daraus lernt das Netz *nichts*, aber *wir* treffen damit
#   Entscheidungen: Wann stoppen wir (`EarlyStopping`)? Wann senken wir die
#   Lernrate (`ReduceLROnPlateau`)? Welche Architektur nehmen wir? Weil wir diese
#   Menge für Entscheidungen benutzen, ist auch sie am Ende "kontaminiert".
# * **Test** — bleibt bis zum Schluss unangetastet und liefert die einzige
#   ehrliche Schätzung der Generalisierungsfähigkeit.
#
# Hier wird ohne Shuffle geschnitten, was normalerweise gefährlich ist (bei
# sortierten Daten landen ganze Klassen in nur einem Split). In diesem Fall ist es
# unkritisch, weil die Volumen bereits in völlig zufälliger Reihenfolge erzeugt
# wurden — die Labels sind i.i.d. gleichverteilt gezogen.
#
# ### Die beiden `assert`-Anweisungen
#
# Sie sind kein Beiwerk, sondern Schutz vor genau dem Fehler, der oben im Kommentar
# erwähnt wurde: Wenn `y` zwischenzeitlich von einer Plot-Zelle überschrieben
# wurde (etwa durch `np.mgrid`), dann passen `X` und `y` nicht mehr zusammen. Ohne
# Assertion würde das Training einfach *falsche* Paare lernen und man würde sich
# über schlechte Ergebnisse wundern. Solche "Guard Rails" in Notebooks sind
# ausgesprochen empfehlenswert, weil Zellen in beliebiger Reihenfolge ausführbar
# sind und der Zustand des Kernels leicht inkonsistent wird.
#
# ### Was fehlt hier bewusst?
#
# Eine Normalisierung der Eingabedaten. Üblicherweise skaliert man Bilddaten auf
# Mittelwert $0$ / Standardabweichung $1$. Hier liegen alle Werte bereits per
# Konstruktion in $[0, 1]$, deshalb ist der Schritt überflüssig. Für LRP ist das
# sogar günstig: der Wert $0$ bedeutet eindeutig "kein Gewebe" und ist damit ein
# natürlicher *Referenzpunkt* ("Baseline"), auf den sich Relevanzwerte beziehen.
#
# ### Ausgabe
#
# `Split: train=600, val=200, test=200` — das ist die einzige Ausgabe der Zelle.

# %%
from plotly.figure_factory import create_distplot

X = np.asarray(X)
if X.ndim == 4:
    X = X[..., np.newaxis]
assert X.shape[1:] == (IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1), (
    f"X.shape={X.shape}, erwartet (N, {IMAGE_SIZE}, {IMAGE_SIZE}, {IMAGE_SIZE}, 1)"
)
y = np.asarray(y).reshape((-1,))
assert len(X) == len(y), (
    f"X/y desynchron: len(X)={len(X)}, len(y)={len(y)} — "
    "wurde y überschrieben (z.B. durch np.mgrid)?"
)
y = y.reshape((-1, 1))
train_X = X[:int(0.6*len(X))]
train_y = y[:int(0.6*len(X))]

val_X = X[int(0.6*len(X)):int(0.8*len(X))]
val_y = y[int(0.6*len(X)):int(0.8*len(X))]

test_X = X[int(0.8*len(X)):]
test_y = y[int(0.8*len(X)):]
print(f"Split: train={len(train_X)}, val={len(val_X)}, test={len(test_X)}")

# %% [markdown]
# <a id="sec6"></a>
# ## 6. Noch ein Zwischenstopp
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Wieder ein Kontrollpunkt wie in Abschnitt 2. Er trennt den Datenteil des
# Notebooks vom Modellteil: alles oberhalb betrifft die Daten, alles unterhalb das
# neuronale Netz.

# %%
1+1

# %% [markdown]
# <a id="sec7"></a>
# ## 7. Modell 1: 3D-CNN mit Global Average Pooling
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Erst die Technik: GPU und Zahlenformat
#
# `set_memory_growth(gpu, True)` verhindert, dass TensorFlow gleich beim Start den
# gesamten GPU-Speicher reserviert. Bei 3D-Daten ist das wichtig, denn ein Batch
# aus 32 Volumen mit 32 Kanälen belegt in der ersten Schicht bereits
# $32 \cdot 32^3 \cdot 32 \cdot 4\,\text{Byte} \approx 134\,\text{MB}$ — 3D-Netze
# sind speicherhungrig.
#
# `mixed_precision.set_global_policy("float32")` ist hier **keine Formalie**.
# Normalerweise beschleunigt `mixed_float16` das Training deutlich. Die
# LRP-Implementierung dieses Repos baut aber ein zweites Modell aus den Gewichten
# des ersten und rechnet die Relevanz rückwärts; dabei kollidieren `float16`- und
# `float32`-Tensoren. Deshalb wird global auf `float32` festgelegt. Merksatz:
# *Erklärbarkeit stellt eigene Anforderungen an die Numerik.*
#
# ### Die Architektur
#
# Der Aufbau folgt dem klassischen CNN-Muster "*Auflösung runter, Kanäle rauf*":
#
# ```
# Input (32,32,32,1)
#   ├─ 3× Block:  Conv3D(3×3×3) → BatchNorm → ReLU → MaxPool(2×2×2)
#   │             Kanäle 32 → 64 → 128,   Auflösung 32³ → 16³ → 8³ → 4³
#   ├─ Conv3D(1×1×1, 64) → BatchNorm → ReLU      (Kanalmischung / Bottleneck)
#   ├─ GlobalAveragePooling3D                    → Vektor der Länge 64
#   ├─ Dense(32) → ReLU → Dropout(0.5)           → Bottleneck-Repräsentation
#   └─ Dense(1)                                  → die Vorhersage
# ```
#
# Insgesamt **289 089 Parameter**, davon 288 513 trainierbar (der Rest sind die
# gleitenden Mittelwerte der BatchNorm-Schichten, die nicht per Gradient gelernt
# werden).
#
# ### Die Bausteine im Einzelnen — und warum sie hier stehen
#
# **`Conv3D(3,3,3)`** — der Filter fährt über den Würfel und berechnet an jeder
# Position eine gewichtete Summe der $3^3 = 27$ Nachbarvoxel:
#
# $$z_{d,h,w}^{(k)} = \sum_{i,j,l,c} W^{(k)}_{i,j,l,c}\, a_{d+i,\,h+j,\,w+l,\,c} + b^{(k)}$$
#
# Entscheidend ist die **Gewichtsteilung**: derselbe Filter wird überall
# angewendet. Genau das brauchen wir, weil die Tunnel an zufälligen Stellen
# liegen — ein Kantendetektor muss überall gleich funktionieren.
#
# **`BatchNormalization`** — normalisiert die Aktivierungen innerhalb eines Batches
# auf Mittelwert $0$, Varianz $1$ und skaliert sie dann mit gelernten Parametern
# $\gamma, \beta$:
#
# $$\hat{a} = \gamma \cdot \frac{a - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$
#
# Das stabilisiert und beschleunigt das Training erheblich. Für LRP ist wichtig:
# BatchNorm ist zur Inferenzzeit eine *affine* Operation und kann daher in die
# vorhergehende Conv-Schicht "hineingerechnet" werden — genau das macht die
# Funktion `fuse_batchnorm` im Paket `explainability`. Deshalb zählt BatchNorm
# später auch **nicht** als eigene LRP-Schicht.
#
# **`Activation('relu')`** — $\mathrm{ReLU}(x) = \max(0,x)$ bringt die
# Nichtlinearität. Ohne sie wäre das ganze Netz eine einzige lineare Abbildung.
# Beachte, dass die Aktivierung hier *bewusst getrennt* von der Conv-Schicht
# steht (`activation=None` in `Conv3D`). Das ist eine Voraussetzung für LRP, das
# Zugriff auf die *linearen* Vor-Aktivierungen braucht.
#
# **`MaxPooling3D(2,2,2)`** — halbiert jede Raumdimension und behält pro
# $2^3$-Block das Maximum. Das reduziert Rechenaufwand und macht das Netz robust
# gegen kleine Verschiebungen. Nebeneffekt, der später in den Heatmaps sichtbar
# wird: durch die Schrittweite 2 entstehen **Gitterartefakte**.
#
# **Rezeptives Feld.** Nach den drei Blöcken "sieht" ein einzelnes Neuron einen
# Bereich von etwa $22^3$ Voxeln (Rechnung: $3 \to 4 \to 8 \to 10 \to 18 \to 22$).
# Bei einem Würfel von $32^3$ ist das schon ein großer Teil des Volumens — jedes
# Neuron der letzten Feature-Map hat also globalen Kontext.
#
# **`Conv3D(1,1,1)`** — mischt nur Kanäle, ohne räumliche Nachbarn einzubeziehen.
# Das ist eine billige Dimensionsreduktion von 128 auf 64 Kanäle
# (8 256 statt 221 312 Parameter).
#
# **`GlobalAveragePooling3D`** — der konzeptionell interessanteste Baustein:
#
# $$g_c = \frac{1}{4^3}\sum_{d,h,w} a_{d,h,w,c}$$
#
# Aus jedem der 64 Kanäle wird *ein* Zahlenwert, der räumliche Mittelwert. Damit
# ist das Netz **vollständig translationsinvariant**: es ist ihm gleichgültig,
# *wo* ein Tunnel liegt, nur *wie viel* Tunnelmerkmal insgesamt vorhanden ist.
# Für unsere Aufgabe ist das exakt die richtige Induktionsannahme (bias), denn die
# Tunnelposition ist ein reiner Störfaktor. Der Preis: Ortsinformation wird
# verworfen, was Heatmaps unschärfer macht. Genau dieser Kompromiss wird in
# Abschnitt 18 mit Modell 2 gegenteilig gelöst — dort ohne GAP.
#
# **`Dense(32)` (der Bottleneck)** — 32 Zahlen, die das ganze Volumen
# zusammenfassen. Diese Schicht wird später einzeln untersucht: Was kodiert jedes
# der 32 Neuronen? Wie redundant sind sie? (Abschnitte 16 und 17)
#
# **`Dropout(0.5)`** — schaltet im Training die Hälfte der Neuronen zufällig ab,
# eine Regularisierung gegen Overfitting. Zur Inferenzzeit inaktiv, für LRP also
# ein Durchlauf-Baustein.
#
# **`Dense(1)` ohne Aktivierung** — bei Regression darf der Ausgang *nicht*
# beschränkt werden (kein Sigmoid, kein Softmax), er muss beliebige reelle Werte
# annehmen können.
#
# **`l2(1e-3)`** — Gewichtsstrafe $\lambda\lVert W\rVert_2^2$, die zur Loss addiert
# wird und große Gewichte bestraft. Das hält die Funktion glatt und, angenehmer
# Nebeneffekt, macht auch die LRP-Erklärungen stabiler.
#
# ### Die Ausgabe: `model.summary()`
#
# Die Tabelle ist der wichtigste Sanity-Check jeder Architektur. Man liest sie von
# oben nach unten und prüft zwei Dinge:
#
# 1. **Die Formen** verkleinern sich wie geplant:
#    $(32,32,32,1) \to (16,16,16,32) \to (8,8,8,64) \to (4,4,4,128) \to (64) \to (32) \to (1)$.
# 2. **Wo sitzen die Parameter?** Hier: 221 312 der 289 089 Parameter (drei
#    Viertel!) stecken in der dritten Conv-Schicht, weil
#    $128 \cdot 64 \cdot 3^3 + 128 = 221\,312$. Der Ausgabeteil kostet
#    fast nichts (2 080 + 33). Das ist typisch für Netze mit GAP und ein
#    Gegenbeispiel zu Modell 2 später.

# %%
import numpy as np
import tensorflow as tf

# GPU Safe-Init (float32: LRP verträgt mixed_float16 nicht)
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {len(gpus)} — {[g.name for g in gpus]}")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import mixed_precision
# float32: mixed_float16 bricht LRP (Dtype-Konflikt) und MaxPool3D auf CPU
mixed_precision.set_global_policy("float32")

from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Conv3D, Dense, Dropout, Input, \
                                    GlobalAveragePooling3D, MaxPooling3D
from tensorflow.keras.regularizers import l2

# deterministic
np.random.seed(42)
tf.random.set_seed(42)

# IMAGE_SIZE muss zu den erzeugten Daten passen (oben: 32)
regularizer = l2(1e-3)
depths = [32, 64, 128, 256, 256, 64]
activation='relu'
dropout=0.5

inputs = Input((IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
x = inputs

for i in range(3):
    x = Conv3D(depths[i], (3, 3, 3), padding='SAME',
               activation=None, kernel_regularizer=regularizer,
               bias_regularizer=regularizer)(x)

    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPooling3D((2, 2, 2))(x)

x = Conv3D(depths[-1], (1, 1, 1), padding='SAME', activation=None,
           kernel_regularizer=regularizer)(x)

x = BatchNormalization()(x)
x = Activation(activation)(x)
x = GlobalAveragePooling3D()(x)

x = Dense(32, activation=None)(x)
x = Activation(activation)(x)
x = Dropout(dropout)(x)
x = Dense(1, activation=None)(x)

model = Model(inputs, x)

model.summary()

# %% [markdown]
# <a id="sec8"></a>
# ## 8. Training und Lernkurven
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was wird berechnet?
#
# `model.compile(loss='mse', optimizer=Adam(1e-3), metrics=['mae'])` legt fest,
# *was* minimiert wird und *wie*.
#
# **Die Verlustfunktion (MSE, mittlerer quadratischer Fehler):**
#
# $$\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)^2$$
#
# Quadrieren bestraft große Abweichungen überproportional. Zusätzlich zeigt Keras
# den **MAE** (mittlerer absoluter Fehler) an:
#
# $$\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}\left|y_i - \hat{y}_i\right|$$
#
# Der MAE ist die interpretierbarere Zahl: "*das Modell liegt im Schnitt um X
# Einheiten Tunnelbreite daneben*".
#
# **Der Optimierer (Adam, Lernrate $10^{-3}$)** passt die Gewichte in Richtung des
# negativen Gradienten an und adaptiert dabei die Schrittweite pro Parameter.
#
# ### Die entscheidende Referenzgröße: was ist ein *guter* Loss?
#
# Diese Frage stellen Einsteiger zu selten. Ein Loss von "17" sagt nichts, solange
# man keine Baseline hat. Die naive Baseline lautet: *immer den Mittelwert
# vorhersagen*. Deren MSE ist genau die Varianz der Zielwerte. Für eine
# Gleichverteilung auf $\{1,\dots,10\}$ gilt
#
# $$\mathbb{E}[y] = 5.5, \qquad
#   \operatorname{Var}(y) = \frac{10^2 - 1}{12} = 8.25$$
#
# **Jedes sinnvolle Modell muss also einen MSE deutlich unter 8.25 erreichen.**
# Der entsprechende MAE der Baseline liegt bei 2.5.
#
# ### Die Callbacks
#
# * **`ReduceLROnPlateau(monitor="loss", factor=0.1, patience=10)`** — stagniert
#   der Trainingsverlust 10 Epochen lang, wird die Lernrate mit $0.1$ multipliziert
#   (bis minimal $10^{-5}$). Typisches Bild: die Lernkurve macht bei jeder
#   Reduktion einen sichtbaren Sprung nach unten.
# * **`EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)`** —
#   verbessert sich der *Validierungs*verlust 50 Epochen lang nicht, wird
#   abgebrochen und auf die besten Gewichte zurückgesetzt. Das ist der
#   Standardschutz gegen Overfitting.
#
# Beachte die unterschiedlichen Monitore: Lernrate reagiert auf das Training,
# Abbruch auf die Validierung. Bei `epochs=2` greift keiner der beiden Callbacks.
#
# ### Trainieren oder laden? Der Caching-Zweig
#
# Die Zelle trainiert nicht unbedingt. Sie sucht erst im Verzeichnis
# `model_1_gap` nach einer `.keras`-Datei:
#
# * **Datei gefunden** → `load_model(...)` lädt die Gewichte, `history = None`,
#   es wird *nicht* trainiert.
# * **Keine Datei** → es wird trainiert und das Ergebnis mit `model.save(...)`
#   abgelegt.
#
# Das ist bei teuren Modellen sehr praktisch, hat aber zwei Konsequenzen, die man
# kennen muss:
#
# 1. **Ohne Training gibt es keine `history`** und damit keine Lernkurve. Die Zelle
#    gibt dann nur `Keine Lernkurven: das Modell wurde geladen, nicht trainiert.`
#    aus. Der `if history is None`-Zweig fängt das ab, damit die Zelle nicht mit
#    einem Fehler abbricht.
# 2. **Ein gespeichertes Modell ist "unsichtbarer Zustand".** Ändert man die
#    Architektur oder die Epochenzahl, wird trotzdem das alte Modell geladen,
#    solange die Datei existiert. Wer neu trainieren will, muss die `.keras`-Datei
#    löschen. Das ist eine klassische Stolperfalle bei Caching in Notebooks —
#    Ergebnisse und Code passen dann nicht mehr zusammen.
#
# Die unten diskutierten Zahlen stammen aus einem Durchlauf, in dem tatsächlich
# trainiert wurde.
#
# ### Die Grafik: Lernkurven
#
# Geplottet werden Trainings- und Validierungsverlust pro Epoche. Die tatsächlichen
# Werte dieses Durchlaufs:
#
# | Epoche | Training loss | Validation loss |
# |--------|--------------|-----------------|
# | 1 | 30.80 | 36.48 |
# | 2 | 17.45 | 31.92 |
#
# ### Interpretation
#
# * Beide Kurven **fallen** — das Netz lernt überhaupt etwas, die Lernrate ist
#   nicht katastrophal falsch gewählt.
# * Beide Werte liegen aber **weit über der Baseline von 8.25**. Nach zwei Epochen
#   hat das Modell noch nicht einmal gelernt, den Mittelwert vorherzusagen. Der
#   Grund ist einfach: der Bias der Ausgabeschicht startet bei $0$, und ihn von $0$
#   auf $\approx 5.5$ zu schieben, dauert viele Schritte (600 Trainingsbeispiele
#   ergeben nur 19 Gradientenschritte pro Epoche).
# * Der Validierungsverlust liegt über dem Trainingsverlust. Bei zwei Epochen ist
#   das noch **kein** Overfitting-Signal: der Trainingswert ist ein *Durchschnitt
#   über die Epoche* (also über zunehmend bessere Gewichte), der Validierungswert
#   wird erst am Epochenende gemessen. Zusätzlich sind Dropout und BatchNorm im
#   Trainings- und Inferenzmodus unterschiedlich aktiv.
# * **Mit zwei Punkten kann man keine Lernkurve beurteilen.** Der eigentliche Zweck
#   dieses Plots — erkennen, ob und wann Training und Validierung auseinanderlaufen
#   (die klassische Overfitting-Schere) — erfordert die ursprünglichen 500 Epochen.
#
# ### Der größere Kontext
#
# Die Lernkurve ist das erste Diagnosewerkzeug, das man liest, und sie beantwortet
# drei Fragen: Lernt das Modell überhaupt (fällt der Trainingsverlust)?
# Generalisiert es (folgt der Validierungsverlust)? Ist es fertig (flachen beide
# ab)? Erst wenn diese drei Fragen positiv beantwortet sind, ergibt es Sinn, über
# Erklärungen nachzudenken — **die Erklärung eines schlechten Modells erklärt
# nichts über die Daten, sondern nur über das schlechte Modell.**

# %%
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss='mse', optimizer=Adam(1e-3), metrics=['mae'])

callbacks = [
    ReduceLROnPlateau(
        monitor="loss",
        factor=0.1,
        patience=10,
        min_lr=1e-5
    ),
    EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=50,
        restore_best_weights=True
    )
]

MODEL_DIR = target_dir / "model_1_gap"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "brain_regression_gap.keras"

existing_model_path = next(iter(sorted(MODEL_DIR.glob("*.keras"))), None)

if existing_model_path is not None:
    model = load_model(existing_model_path)
    history = None
    print(f"Model geladen von: {existing_model_path}")
else:
    history = model.fit(train_X, train_y, 
                        validation_data=(val_X, val_y), 
                        batch_size=32,
                        #epochs=500,
                        epochs=2,
                        callbacks=callbacks)

    model.save(MODEL_PATH)
    print(f"Model gespeichert unter: {MODEL_PATH}")


if history is None:
    print("Keine Lernkurven: das Modell wurde geladen, nicht trainiert.")
else:
    traces = [
        go.Scatter(
            x=np.arange(len(history.history['loss'])),
            y=history.history['loss'],
            name='Training loss'
        ),
        go.Scatter(
            x=np.arange(len(history.history['loss'])),
            y=history.history['val_loss'],
            name='Validation loss'
        )
    ]

    iplot(go.Figure(traces))

# %% [markdown]
# <a id="sec9"></a>
# ## 9. Wie gut sagt das Modell vorher? Streudiagramme
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was wird berechnet?
#
# Für alle drei Datenmengen werden die Vorhersagen $\hat{y}$ berechnet und gegen
# die wahren Werte $y$ aufgetragen: links Training, Mitte Validierung, rechts Test.
# Die zusätzliche Linie von $(0,0)$ nach $(11,11)$ ist die **Identität** $\hat y = y$.
#
# Dieses "Predicted vs. Observed"-Diagramm ist bei Regression aussagekräftiger als
# jede einzelne Kennzahl, weil es die *Struktur* der Fehler zeigt. Man liest es so:
#
# | Muster | Bedeutung |
# |--------|-----------|
# | Punkte liegen eng auf der Diagonale | ideal |
# | Punkte parallel *unter* der Diagonale | systematische Unterschätzung (Bias) |
# | Punkte bilden eine flache Wolke | Modell sagt fast immer denselben Wert (kein Signal genutzt) |
# | Steigung $<1$, an den Rändern eingeknickt | "Regression zur Mitte" — der Klassiker bei Altersschätzung |
# | Streuung wächst nach rechts | heteroskedastische Fehler |
#
# ### Die Grafik in diesem Durchlauf
#
# Die Zahlen aus den geplotteten Daten:
#
# | Menge | $n$ | Bereich $y$ | Bereich $\hat{y}$ | Korrelation | MAE |
# |-------|-----|-------------|-------------------|-------------|-----|
# | Training | 600 | 1 – 10 | 0.26 – 1.00 | 0.45 | 4.91 |
# | Validierung | 200 | 1 – 10 | 0.27 – 0.92 | 0.49 | 4.84 |
# | Test | 200 | 1 – 10 | 0.26 – 0.98 | 0.51 | 4.91 |
#
# ### Interpretation
#
# * **Alle Punkte kleben als flaches Band am unteren Bildrand.** Die Diagonale
#   verläuft steil darüber und die Punktwolke berührt sie fast nirgends. Das ist
#   der visuelle Fingerabdruck eines Modells mit **massivem
#   Unterschätzungs-Bias**: die Vorhersagen bewegen sich in $[0.26, 1.0]$, während
#   die Zielwerte von 1 bis 10 reichen. Der MAE von ~4.9 ist fast so groß wie der
#   Mittelwert der Zielgröße — schlechter als die naive Mittelwert-Baseline (2.5).
# * **Aber es gibt schon Signal:** Die Korrelation zwischen $y$ und $\hat y$ liegt
#   bei $0.45$–$0.51$. Das Modell ordnet die Volumen also bereits merklich
#   *richtig ein*, es hat nur noch nicht gelernt, in welchem Wertebereich seine
#   Antwort liegen muss. Man kann das trennen: die *Rangordnung* ist teilweise
#   gelernt, die *Kalibrierung* fehlt völlig.
# * **Kein Overfitting:** Training (0.45) und Test (0.51) liegen gleichauf, der
#   Testwert ist sogar minimal besser. Nach zwei Epochen konnte das Netz die
#   Trainingsdaten noch nicht auswendig lernen.
# * **Konsequenz für alles Weitere:** Alle folgenden LRP-Heatmaps stammen von genau
#   diesem Modell. Sie zeigen daher nicht "*wie löst ein gutes Netz die Aufgabe*",
#   sondern "*worauf schaut ein Netz nach zwei Epochen*". Man sollte sie als
#   Demonstration des *Verfahrens* lesen, nicht als inhaltliche Aussage über die
#   Daten. Für inhaltliche Schlüsse müsste `epochs=500` gesetzt werden.
#
# ### Der größere Kontext
#
# Genau dieses Diagramm ist in der Brain-Age-Literatur der Standard: aufgetragen
# werden chronologisches Alter gegen geschätztes Alter, und die Differenz
# ($\text{brain age gap} = \hat{y} - y$) ist der eigentlich interessante Biomarker.
# Dort ist der oben genannte "Regression zur Mitte"-Effekt ein bekanntes und viel
# diskutiertes Problem, für das es eigene Korrekturverfahren gibt.

# %%
from plotly.subplots import make_subplots


train_predictions = model.predict(train_X)
val_predictions = model.predict(val_X)
test_predictions = model.predict(test_X)

fig = make_subplots(1, 3)

fig.add_trace(
    go.Scatter(
        x=train_y.squeeze(),
        y=train_predictions.squeeze(),
        mode='markers',
        showlegend=False
    )
, row=1, col=1)
fig.add_trace(
    go.Scatter(
        x=[0, 11],
        y=[0, 11],
        mode='lines',
        showlegend=False
    )
, row=1, col=1)

fig.add_trace(
    go.Scatter(
        x=val_y.squeeze(),
        y=val_predictions.squeeze(),
        mode='markers',
        showlegend=False
    )
, row=1, col=2)
fig.add_trace(
    go.Scatter(
        x=[0, 11],
        y=[0, 11],
        mode='lines',
        showlegend=False
    )
, row=1, col=2)

fig.add_trace(
    go.Scatter(
        x=test_y.squeeze(),
        y=test_predictions.squeeze(),
        mode='markers',
        showlegend=False
    )
, row=1, col=3)
fig.add_trace(
    go.Scatter(
        x=[0, 11],
        y=[0, 11],
        mode='lines',
        showlegend=False
    )
, row=1, col=3)

# %% [markdown]
# <a id="sec10"></a>
# ## 10. LRP: Grundidee und erste Heatmaps
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Ab hier beginnt der eigentliche Kern des Notebooks. Das Modell steht, jetzt geht
# es um die Frage: **warum sagt es, was es sagt?**
#
# ### Die Idee von Layer-wise Relevance Propagation
#
# LRP nimmt die Vorhersage $f(x)$ und verteilt sie schichtweise rückwärts auf die
# Eingabe, bis jeder Voxel einen Anteil bekommt. Der Anteil heißt **Relevanz**
# $R_i$. Die zentrale Eigenschaft ist die **Erhaltung** (conservation):
#
# $$\sum_{i \in \text{Schicht } l} R_i^{(l)}
#   \;=\; \sum_{j \in \text{Schicht } l+1} R_j^{(l+1)}
#   \;=\; \dots \;=\; f(x)$$
#
# Die Vorhersage wird also wie ein Geldbetrag rückwärts aufgeteilt: Jede Schicht
# gibt ihren Betrag an die Neuronen weiter, die dazu beigetragen haben. Am Ende
# gilt $\sum_i R_i \approx f(x)$, wobei $i$ über alle $32^3$ Voxel läuft.
#
# Der Unterschied zu Gradienten-Verfahren (Saliency, Grad-CAM) ist konzeptionell:
# Ein Gradient beantwortet "*was würde passieren, wenn ich diesen Voxel minimal
# ändere*" (Sensitivität), LRP beantwortet "*wie viel von der Vorhersage geht auf
# diesen Voxel zurück*" (Beitrag). Für Regression ist Letzteres meist die Frage,
# die man eigentlich stellen wollte.
#
# ### Die Regeln — und was `LRPStrategy` hier konfiguriert
#
# Die Weitergabe der Relevanz von Schicht zu Schicht folgt einer *Regel*. Drei
# davon werden hier benutzt.
#
# **1. Die $\varepsilon$-Regel** (Basisregel mit Stabilisierung):
#
# $$R_i = \sum_j \frac{a_i w_{ij}}{\varepsilon\,\mathrm{sign}(z_j) + z_j}\, R_j,
#   \qquad z_j = \sum_{i'} a_{i'} w_{i'j} + b_j$$
#
# Jedes Eingangsneuron $i$ erhält den Anteil seines Beitrags $a_i w_{ij}$ am
# Gesamtinput $z_j$. Das $\varepsilon$ im Nenner verhindert Division durch (fast)
# Null und dämpft dabei schwaches Rauschen — größeres $\varepsilon$ ergibt
# dünnere, aber stabilere Heatmaps. Hier: `{'epsilon': 0.25}`.
#
# **2. Die $\alpha\beta$-Regel** (getrennte Behandlung von Zustimmung und Widerspruch):
#
# $$R_i = \sum_j \left(
#     \alpha \frac{(a_i w_{ij})^+}{\sum_{i'}(a_{i'} w_{i'j})^+}
#   - \beta  \frac{(a_i w_{ij})^-}{\sum_{i'}(a_{i'} w_{i'j})^-}
#   \right) R_j, \qquad \alpha - \beta = 1$$
#
# Positive und negative Beiträge werden getrennt normiert und dann gewichtet
# zusammengeführt. Mit `{'alpha': 2, 'beta': 1}` werden anregende Beiträge doppelt
# gewichtet und hemmende einfach abgezogen. Das ist die in der Praxis bewährte
# Einstellung für tiefe Conv-Netze; sie erzeugt kontrastreiche, weniger verrauschte
# Karten als die reine $\varepsilon$-Regel. Die Bedingung $\alpha = \beta + 1$
# garantiert die Erhaltung und wird von der Bibliothek per `assert` erzwungen.
#
# **3. Die `flat`-Regel** (für die *erste* Schicht):
#
# Hier werden Aktivierungen und Gewichte durch Einsen ersetzt ($a \equiv 1$,
# $w \equiv 1$), die Relevanz also *gleichmäßig* auf das rezeptive Feld verteilt.
# Der Grund: in der Eingabeschicht sind die "Aktivierungen" rohe Voxelintensitäten,
# und die üblichen Regeln erzeugen dort sehr punktuelle, rauschige Muster. Die
# `flat`-Regel glättet das zu zusammenhängenden Regionen — sie macht Heatmaps
# *lesbarer*, um den Preis der räumlichen Präzision. Diesen Effekt sieht man in
# Abschnitt 13 sehr deutlich.
#
# **Die Zuordnung Strategie → Schicht.** `LRPStrategy(layers=[...])` erwartet
# genau einen Eintrag pro *Standard-LRP-Schicht*, in Reihenfolge von der Eingabe
# zur Ausgabe. Nur Conv- und Dense-Schichten zählen; Pooling, BatchNorm,
# Aktivierung und Dropout werden anders behandelt (BatchNorm wird sogar in die
# Conv-Schicht hineingerechnet). Für Modell 1 ergibt das sechs Einträge:
#
# | # | Schicht | Regel | Warum |
# |---|---------|-------|-------|
# | 1 | Conv3D(32) | `flat` | Eingabeschicht → glatte, lesbare Karten |
# | 2 | Conv3D(64) | $\alpha=2,\beta=1$ | Standard für Conv-Blöcke |
# | 3 | Conv3D(128) | $\alpha=2,\beta=1$ | " |
# | 4 | Conv3D(64) 1×1×1 | $\alpha=2,\beta=1$ | " |
# | 5 | Dense(32) | $\alpha=2,\beta=1$ | " |
# | 6 | Dense(1) | $\varepsilon=0.25$ | Ausgabeschicht, nur stabilisieren |
#
# ### Was macht der Rest der Zelle?
#
# `clone_model` + `set_weights(float32)` erzeugt eine saubere `float32`-Kopie des
# Modells — nochmals die Numerik-Vorsichtsmaßnahme von oben.
#
# `LayerwiseRelevancePropagator(lrp_model, layer=-1, idx=0, strategy=strategy)`
# baut daraus ein **neues Keras-Modell**, dessen *Ausgabe die Heatmap ist*. Das ist
# der elegante Kern dieses Repos: die Erklärung ist selbst ein Modell, das man mit
# `.predict()` aufruft und auf der GPU laufen lassen kann. `layer=-1` heißt "erkläre
# die letzte Schicht", `idx=0` heißt "erkläre deren nulltes Ausgabeneuron" — bei
# Regression gibt es nur dieses eine.
#
# Die Schleife sucht für jede Tunnelbreite $1..10$ das erste passende Testvolumen
# und erklärt es. Anschließend wird normiert:
#
# $$\tilde{R} = \frac{R}{\max_i |R_i|} \in [-1, 1]$$
#
# Das ist reine Darstellungsnormierung; sie macht Bilder vergleichbar, geht aber
# auf Kosten der absoluten Skala (ein Volumen mit sehr schwacher Relevanz sieht
# danach genauso "stark" aus wie eins mit starker).
#
# ### Die Grafiken: 10 Bilder, je 2 × 8 Felder
#
# Es entsteht **ein Bild pro Tunnelbreite**, also zehn Bilder von oben (Breite 1)
# nach unten (Breite 10). In jedem Bild gilt:
#
# * **obere Reihe:** die Schnittbilder $z = 12\dots19$ in Graustufen (das Eingabevolumen),
# * **untere Reihe:** die zugehörige Relevanz derselben Schnitte, Farbskala
#   `seismic` mit fixierten Grenzen $[-1, +1]$.
#
# **Die Farbskala ist der Schlüssel zum Lesen:**
#
# | Farbe | Relevanz | Bedeutung |
# |-------|----------|-----------|
# | **rot** | $R > 0$ | dieser Voxel hat die vorhergesagte Tunnelbreite **erhöht** |
# | **weiß** | $R \approx 0$ | ohne Einfluss |
# | **blau** | $R < 0$ | dieser Voxel hat die Vorhersage **gesenkt** |
#
# Weil `clim=(-1,1)` fest gesetzt ist, sind Farben zwischen den zehn Bildern
# vergleichbar (jeweils relativ zum Maximum des eigenen Volumens).
#
# ### Interpretation
#
# **Das Grundmuster (Breite 1–5):** In der Mitte liegt ein **blauer Klecks**, der
# ziemlich genau die Gewebekugel abdeckt, und darum herum ein **roter Ring**, der
# den leeren Außenraum und die Kugelkontur einnimmt.
#
# Das ist inhaltlich stimmig, und zwar mit dem *richtigen Vorzeichen*: Gewebe ist
# Evidenz *gegen* breite Tunnel (viel Gewebe = wenig ausgehöhlt = kleines Label),
# also negative Relevanz; leerer Raum ist Evidenz *für* breite Tunnel, also positive
# Relevanz. Das Netz hat nach zwei Epochen zwar noch keine kalibrierte Skala, aber
# offenbar bereits die grundsätzliche Richtung "*Gewebe runter, Leere rauf*" gelernt.
#
# **Die Entwicklung über die Breiten:** Von Bild 1 zu Bild 10 schrumpft der blaue
# Anteil und der rote Anteil wächst und wird intensiver. Beim größten Label
# (letztes Bild) ist vom Gewebe nur noch ein Fragment übrig, entsprechend gibt es
# nur noch kleine blaue Inseln in einem überwiegend roten Feld. Da
# $\sum_i R_i \approx f(x)$ gilt, bedeutet mehr Rot und weniger Blau *rechnerisch*
# eine höhere Vorhersage. Die Heatmaps bilden also genau die **monotone Beziehung**
# ab, die wir wollen — der Mechanismus stimmt, nur die Skala ist noch falsch.
#
# **Das Gittermuster im Hintergrund:** Auffällig ist ein regelmäßiges rotes
# Schachbrett-/Punktraster außerhalb der Kugel. Das ist **kein** Befund über die
# Daten, sondern ein **Artefakt**, und es lohnt sich, seine Herkunft zu kennen:
#
# 1. Das dreifache `MaxPooling3D` mit Schrittweite 2 bildet je $2\times2\times2$
#    Voxel auf eine Zelle ab; beim Rückweg wird Relevanz auf dieses Gitter
#    zurückverteilt und erzeugt eine Periodizität von $2^3 = 8$ Voxeln.
# 2. `padding='SAME'` lässt Filter über den Bildrand ragen, wodurch Randvoxel
#    systematisch anders behandelt werden.
# 3. Die `flat`-Regel in der ersten Schicht verteilt Relevanz *unabhängig von der
#    Aktivierung* — deshalb bekommen auch Voxel mit Wert $0$ Relevanz zugewiesen.
#
# Solche Checkerboard-Artefakte in Attributionskarten sind gut dokumentiert. Wichtig
# ist die Konsequenz für die Praxis: **Man darf Struktur in einer Heatmap nicht
# automatisch als inhaltlichen Befund lesen.** Ein Teil des Musters kommt von der
# Architektur und der gewählten LRP-Regel, nicht von den Daten. Genau deshalb folgen
# als Nächstes zwei *kausale* Gegenproben.
#
# **Was noch fehlt:** Was wir gerne sehen würden — scharfe rote Konturen *entlang
# der Tunnelwände*, denn dort steckt die Information über die Breite — ist hier nur
# angedeutet. Die Karten sind grobkörnig und großflächig. Zwei Ursachen: das
# untertrainierte Modell und das Global Average Pooling, das Ortsinformation
# verwirft. Modell 2 in Abschnitt 18 greift genau diesen zweiten Punkt an.

# %%
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import clone_model

from explainability import LayerwiseRelevancePropagator, LRPStrategy


# LRP braucht float32 (auch wenn Training mit mixed_float16 lief)
mixed_precision.set_global_policy("float32")
lrp_model = clone_model(model)
lrp_model.set_weights([np.asarray(w, dtype=np.float32) for w in model.get_weights()])

# 6 StandardLRP-Layer: 4× Conv3D + Dense(32) + Dense(1); Pooling zählt nicht
strategy = LRPStrategy(
    layers=[
        {'flat': True},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'epsilon': 0.25},
    ]
)

explainer = LayerwiseRelevancePropagator(lrp_model, layer=-1, idx=0, strategy=strategy)

assert len(test_X) == len(test_y), (
    f"test_X/test_y desynchron: {len(test_X)} vs {len(test_y)} — Split-Zelle neu ausführen"
)
labels = np.asarray(test_y).ravel()
for radius in range(1, MAX_RADIUS + 1):
    matches = np.flatnonzero(labels == radius)
    if matches.size == 0:
        print(f"Skip radius {radius}: kein Sample in test_y")
        continue
    idx = int(matches[0])
    sample = np.asarray(test_X[idx:idx + 1], dtype=np.float32)

    fig, ax = plt.subplots(2, 8, figsize=(15, 3))
    explanations = explainer.predict(sample, verbose=0)
    explanations = explanations / np.amax(np.abs(explanations))

    for j in range(8):
        ax[0][j].imshow(test_X[idx, 12 + j], cmap='Greys_r')
        ax[0][j].axis('off')
        ax[1][j].imshow(explanations[0, 12 + j], cmap='seismic', clim=(-1, 1))
        ax[1][j].axis('off')

plt.show()

# %% [markdown]
# <a id="sec11"></a>
# ## 11. Kausaler Test 1: immer mehr Tunnel bohren
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Die Idee hinter diesem Experiment
#
# Heatmaps sind *korrelativ*: sie zeigen, wo das Netz Relevanz verortet, aber nicht,
# ob eine Änderung dort die Vorhersage tatsächlich verändert. Der stärkere Test ist
# ein **kausaler Eingriff**: Man verändert die Eingabe gezielt und beobachtet die
# Reaktion des Modells. In der XAI-Literatur heißt diese Familie von Tests
# *perturbation analysis* bzw. *pixel flipping*.
#
# Der Vorteil des synthetischen Datensatzes: wir können den Eingriff *exakt so*
# durchführen, wie die Daten generiert wurden, und kennen daher die richtige Antwort.
#
# ### Was passiert in dieser Zelle?
#
# Es wird **ein** Gehirn gebaut, diesmal mit fixem Mittelpunkt (genau in der Mitte)
# und fixem Radius, damit keine Störvariation dazukommt. Dann wird in einer Schleife
# $2 \cdot 6 = 12$-mal ein Tunnel gebohrt — **immer mit derselben Breite
# `width=5`** — und nach *jedem* Bohrvorgang die Vorhersage abgefragt:
#
# $$p_k = f\big(X_k\big), \qquad k = 1,\dots,12$$
#
# wobei $X_k$ das Volumen mit $k$ Tunneln ist. Wichtig: die Tunnel werden
# *kumulativ* in dasselbe Volumen gebohrt (`drill` verändert `brain` in place).
#
# ### Was wäre das *korrekte* Verhalten?
#
# Hier liegt der didaktische Kern. Das Label hängt nur von der **Breite** ab, nicht
# von der **Anzahl** der Tunnel. Die Breite bleibt konstant 5. Ein perfektes Modell
# müsste also eine **waagerechte Linie bei 5** liefern — daher die grüne
# gestrichelte Referenzlinie im Plot. Jede Steigung ist eine Verletzung der
# gewünschten *Invarianz gegenüber der Tunnelanzahl*.
#
# Das ist genau die Sorte Fehler, die man in der Praxis fürchtet: das Modell nutzt
# eine mit dem Ziel korrelierte, aber falsche Größe (hier: den gesamten
# Gewebeverlust) als Abkürzung — ein **Shortcut** bzw. Confounder. In den
# Trainingsdaten war die Anzahl immer 6, weshalb das Modell nie lernen musste, die
# beiden Effekte zu trennen.
#
# ### Die Grafiken
#
# **Erstens** eine Reihe von 8 Schnittbildern des *fertigen* Volumens (nach allen
# 12 Bohrungen): eine kreisförmige Scheibe, in die von allen Seiten schwarze
# Buchten und Löcher hineinragen. Gut zu sehen ist, wie stark die Kugel nach 12
# Tunneln bereits durchlöchert ist, obwohl jeder einzelne Tunnel nur Breite 5 hat.
#
# **Zweitens** der Verlauf der Vorhersage über die Tunnelanzahl. Die gemessenen
# Werte:
#
# | Tunnel $k$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
# |---|---|---|---|---|---|---|---|---|---|---|---|---|
# | $p_k$ | 0.160 | 0.160 | 0.163 | 0.167 | 0.168 | 0.172 | 0.172 | 0.180 | 0.185 | 0.186 | 0.189 | 0.189 |
#
# ### Interpretation
#
# * **Die Richtung ist richtig, die Größenordnung nicht.** Die Kurve steigt
#   monoton: mehr entferntes Gewebe → höhere Vorhersage. Das Modell hat die
#   Grundlogik "*weniger Gewebe bedeutet größeres Label*" verinnerlicht.
# * **Die Kurve liegt aber bei ~0.16 bis 0.19, während die Referenzlinie bei 5
#   liegt.** Im Plot ist die blaue Messkurve deshalb praktisch platt am unteren
#   Rand, weit unter der grünen Linie. Das ist derselbe Kalibrierungsfehler wie im
#   Streudiagramm: der Effekt ist um mehr als einen Faktor 25 zu klein.
# * **Der Anstieg ist eine Invarianzverletzung.** Ein austrainiertes Modell sollte
#   hier flach bei 5 verlaufen. Der Anstieg von 0.160 auf 0.189 (+18 %) zeigt, dass
#   die Anzahl der Tunnel mitgemessen wird. Ob das bei 500 Epochen verschwindet, ist
#   eine offene und lohnende Frage — ein sauberer Weg wäre, `NUM_TUNNELS` in den
#   Trainingsdaten zu variieren, damit das Modell die Anzahl als irrelevant lernen
#   *kann*.
# * **Ein Darstellungsdetail:** Die x-Werte der Messkurve werden mit
#   `np.arange(1, 41)` erzeugt, es gibt aber nur 12 Messwerte. Plotly zeichnet
#   entsprechend nur 12 Punkte — inhaltlich harmlos, aber die x-Achse suggeriert
#   einen längeren Verlauf, als tatsächlich gemessen wurde.
#
# ### Der größere Kontext
#
# Dieser Test ist das synthetische Gegenstück zu einer Frage, die in der
# medizinischen Bildgebung ständig auftaucht: Reagiert mein Alters-Schätzer auf
# *Atrophie* oder auf *Kopfgröße*, *Scanner-Modell*, *Bewegungsartefakte*? Da man
# in echten Daten nicht kontrolliert eingreifen kann, ist das synthetische Setup
# hier die einzige Möglichkeit, diese Frage sauber zu beantworten.

# %%
from plotly.colors import DEFAULT_PLOTLY_COLORS


np.random.seed(42)

brain = np.zeros((IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))

center = np.asarray([IMAGE_SIZE // 2 for _ in range(3)])
radius = IMAGE_SIZE // 2-2

idx = np.asarray(np.meshgrid(*[np.arange(IMAGE_SIZE) for _ in range(3)])).T.reshape(-1, 3)
distances = euclidean_distances(idx, center.reshape(1, -1))[:,0]
inside = distances <= radius
surface = np.isclose(distances, radius, atol=1e-1)
surface = idx[surface]

brain[tuple(idx[inside].T)] = np.random.uniform(0.25, 1, (len(idx[inside]), 1))
brain[tuple(idx[surface].T)] = np.random.uniform(0.25, 1, (len(idx[surface]), 1))

inside_keys = set([key(x) for x in idx[inside]])

predictions = []

for _ in range(1, 2 * NUM_TUNNELS + 1):
    brain = drill(brain, surface, center, 5, inside_keys, idx)
    predictions.append(model.predict(np.expand_dims(brain, 0))[0,0])
    
fig, ax = plt.subplots(1, 8, figsize=(15, 2))

for i in range(8):
    ax[i].imshow(brain[12+i], cmap='Greys_r')
    ax[i].axis('off')

plt.show()
    
traces = [
    go.Scatter(
        x=np.arange(1, 41),
        y=predictions,
        mode='markers+lines',
        showlegend=False,
        marker={
            'color': DEFAULT_PLOTLY_COLORS[0]
        },
        line={
            'color': DEFAULT_PLOTLY_COLORS[0]
        }
    ),
    go.Scatter(
        x=[1, 2*NUM_TUNNELS],
        y=[5, 5],
        mode='lines',
        showlegend=False,
        line={
            'color': DEFAULT_PLOTLY_COLORS[2],
            'dash': 'dash'
        }
    )
]

layout = go.Layout(
    title={
        'x': 0.5,
        'text': 'Prediction as a function of number of tunnels'
    },
    xaxis={
        'title': 'Number of tunnels'
    },
    yaxis={
        'title': 'Prediction'
    }
)

iplot(go.Figure(traces, layout))

# %% [markdown]
# <a id="sec12"></a>
# ## 12. Kausaler Test 2: abwechselnd breite und schmale Tunnel
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was wird hier variiert?
#
# Der Aufbau ist derselbe wie in Abschnitt 11, mit einem entscheidenden Unterschied:
# jetzt bleibt die *Anzahl* der Schritte klein (6), aber die **Breite wechselt**:
#
# $$w_i = 2 + 6\,(i \bmod 2) \;\Longrightarrow\;
#   w = (8,\,2,\,8,\,2,\,8,\,2)$$
#
# Also: Schritte 1, 3, 5 bohren **breite** Tunnel ($w=8$), Schritte 2, 4, 6 bohren
# **schmale** ($w=2$). Die Marker im Plot sind entsprechend abwechselnd gefärbt.
#
# Damit trennt dieses Experiment die beiden Effekte, die in Abschnitt 11 noch
# vermischt waren: In *jedem* Schritt kommt ein Tunnel hinzu (Anzahl-Effekt), aber
# nur in jedem *zweiten* ist er breit (Breiten-Effekt). Wenn das Modell wirklich die
# Breite misst, dann müssen die **Sprünge nach breiten Bohrungen deutlich größer
# sein** als nach schmalen — im Idealfall eine Treppe mit Stufen nach oben bei
# 1, 3, 5 und Plateaus bei 2, 4, 6.
#
# ### Die Grafiken
#
# **Erstens** wieder 8 Schnittbilder des fertigen Volumens. Hier sieht man den
# Größenunterschied direkt im Bild: einige Einschnitte sind breite, weit
# aufgerissene Buchten (die $w=8$-Bohrungen), andere nur feine Kratzer und kleine
# Punkte ($w=2$).
#
# **Zweitens** der Verlauf der Vorhersage. Die gemessenen Werte samt Zuwachs:
#
# | Schritt | Breite | Vorhersage | Zuwachs $\Delta$ |
# |---------|--------|-----------|------------------|
# | 1 | **8** | 0.1847 | — |
# | 2 | 2 | 0.1850 | **+0.0003** |
# | 3 | **8** | 0.2384 | **+0.0534** |
# | 4 | 2 | 0.2437 | +0.0053 |
# | 5 | **8** | 0.2546 | **+0.0109** |
# | 6 | 2 | 0.2584 | +0.0038 |
#
# ### Interpretation
#
# * **Das ist das ermutigendste Ergebnis des ganzen Notebooks.** Vergleicht man die
#   Zuwächse, so bringt Schritt 3 (breit) mit $+0.053$ etwa **das Zehnfache** von
#   Schritt 4 (schmal, $+0.005$), und Schritt 5 (breit, $+0.011$) rund das Dreifache
#   von Schritt 6 (schmal, $+0.004$). Über alle Schritte gemittelt liefern breite
#   Bohrungen $\approx +0.021$ pro Schritt, schmale $\approx +0.003$.
# * **Damit reagiert das Modell tatsächlich auf die Zielgröße**, und nicht nur auf
#   "irgendwie weniger Gewebe". Genau das wollte man nachweisen: die *Breite* hat
#   einen eigenen, deutlich stärkeren Effekt als die bloße Anzahl. Nach nur zwei
#   Epochen ist das mehr, als man erwarten würde.
# * **Die Treppe ist aber unsauber.** Zwischen Schritt 1 und 2 ist praktisch kein
#   Unterschied, und die Kurve fällt nie ab, sondern steigt durchgehend. Ein
#   sauberer Nachweis der Invarianz gegenüber der Anzahl würde flache Plateaus bei
#   den schmalen Schritten erfordern. Zudem bleibt der absolute Wert bei ~0.26 statt
#   im Bereich $2$–$8$.
# * **Eine wichtige methodische Einschränkung:** Da kumulativ gebohrt wird, enthält
#   das Volumen am Ende *sowohl* breite *als auch* schmale Tunnel gleichzeitig. Der
#   "wahre" Zielwert dieses Mischvolumens ist gar nicht wohldefiniert — die
#   Referenzlinie bei 5 ist deshalb eher als Mittelwert-Orientierung zu verstehen
#   ($(8+2)/2 = 5$) als als harte Ground Truth. Ein noch strengerer Test würde für
#   jede Breite ein *frisches* Volumen bohren.
#
# ### Der größere Kontext
#
# Die beiden Tests der Abschnitte 11 und 12 zusammen sind ein Muster, das sich auf
# jedes Projekt übertragen lässt:
#
# 1. **Verändere die kausal relevante Größe** → das Modell soll stark reagieren.
# 2. **Verändere eine irrelevante Größe** → das Modell soll *nicht* reagieren.
#
# Erst wenn beide Bedingungen erfüllt sind, hat man Grund, den Heatmaps zu
# vertrauen. In diesem Durchlauf ist Bedingung 1 ansatzweise erfüllt, Bedingung 2
# noch nicht.

# %%
np.random.seed(42)

brain = np.zeros((IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))

center = np.asarray([IMAGE_SIZE // 2 for _ in range(3)])
radius = IMAGE_SIZE // 2-2

idx = np.asarray(np.meshgrid(*[np.arange(IMAGE_SIZE) for _ in range(3)])).T.reshape(-1, 3)
distances = euclidean_distances(idx, center.reshape(1, -1))[:,0]
inside = distances <= radius
surface = np.isclose(distances, radius, atol=1e-1)
surface = idx[surface]

brain[tuple(idx[inside].T)] = np.random.uniform(0.25, 1, (len(idx[inside]), 1))
brain[tuple(idx[surface].T)] = np.random.uniform(0.25, 1, (len(idx[surface]), 1))

inside_keys = set([key(x) for x in idx[inside]])

predictions = []

for i in range(1, NUM_TUNNELS + 1):
    width = 2 + (6 * (i % 2))
    brain = drill(brain, surface, center, width, inside_keys, idx)
    predictions.append(model.predict(np.expand_dims(brain, 0))[0,0])
    
fig, ax = plt.subplots(1, 8, figsize=(15, 2))

for i in range(8):
    ax[i].imshow(brain[12+i], cmap='Greys_r')
    ax[i].axis('off')

plt.show()

colours = [DEFAULT_PLOTLY_COLORS[(i+1) % 2] for i in range(len(predictions))]
    
traces = [
    go.Scatter(
        x=np.arange(1, NUM_TUNNELS + 1),
        y=predictions,
        mode='markers+lines',
        showlegend=False,
        marker={
            'color': colours
        },
        line={
            'color': DEFAULT_PLOTLY_COLORS[0]
        },
    ),
    go.Scatter(
        x=[1, NUM_TUNNELS],
        y=[5, 5],
        mode='lines',
        showlegend=False,
        line={
            'color': DEFAULT_PLOTLY_COLORS[2],
            'dash': 'dash'
        }
    )
]

layout = go.Layout(
    title={
        'x': 0.5,
        'text': 'Prediction as a function of number of tunnels'
    },
    xaxis={
        'title': 'Number of tunnels'
    },
    yaxis={
        'title': 'Prediction'
    }
)

iplot(go.Figure(traces, layout))

# %% [markdown]
# <a id="sec13"></a>
# ## 13. LRP über das gesamte Volumen
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert in dieser Zelle?
#
# Bisher wurden immer nur die acht mittleren Schnitte ($z = 12\dots19$) gezeigt.
# Jetzt wird das **komplette** Volumen erklärt und alle 32 Schichten geplottet — von
# $z = 0$ (Rand des Würfels, außerhalb der Kugel) bis $z = 31$.
#
# Erklärt wird das Volumen aus Abschnitt 12, also das mit den abwechselnd breiten
# und schmalen Tunneln. Zusätzlich werden zwei Schichtindizes automatisch bestimmt:
#
# * `output_layer` — die letzte Schicht, `Dense(1)`, das Erklärungsziel.
# * `bottleneck_layer` — die `Dense`-Schicht mit 32 Einheiten. Sie wird in den
#   Abschnitten 14 bis 17 gebraucht. Die Suche per `next(...)` statt einer
#   hartcodierten Zahl ist robuster: bei Änderungen der Architektur verschieben sich
#   Indizes sonst stillschweigend.
#
# ### Das Layout der Grafik
#
# Das $8\times8$-Gitter ist **paarweise** zu lesen. Zeilen 1, 3, 5, 7 zeigen
# Graustufen-Schnitte, Zeilen 2, 4, 6, 8 direkt darunter die zugehörige Relevanz:
#
# | Bildzeile | Inhalt | Schichten |
# |-----------|--------|-----------|
# | 1 | Volumen | $z = 0 \dots 7$ |
# | 2 | Relevanz | $z = 0 \dots 7$ |
# | 3 | Volumen | $z = 8 \dots 15$ |
# | 4 | Relevanz | $z = 8 \dots 15$ |
# | 5 | Volumen | $z = 16 \dots 23$ |
# | 6 | Relevanz | $z = 16 \dots 23$ |
# | 7 | Volumen | $z = 24 \dots 31$ |
# | 8 | Relevanz | $z = 24 \dots 31$ |
#
# Man liest also immer *ein Bild und das Bild direkt darunter* als Paar.
#
# ### Interpretation
#
# * **Der Aufbau und Abbau der Kugel** ist in den Graustufenzeilen schön zu sehen:
#   $z=0,1$ sind komplett schwarz (vor der Kugel), $z=2$ zeigt einen einzelnen
#   Punkt, dann wachsen die Scheiben, erreichen um $z=12\dots19$ ihr Maximum und
#   schrumpfen wieder auf einen Punkt bei $z=29$ und Schwarz bei $z=30,31$. Das ist
#   die Geometrie einer Kugel im Schnitt — und ein guter Anlass, sich klarzumachen,
#   dass das Netz *dieses* 3D-Objekt sieht, nicht 32 unabhängige Bilder.
# * **Der wichtigste Befund: Relevanz dort, wo gar keine Daten sind.** Bei
#   $z = 0$ und $z = 1$ ist die Eingabe *exakt null* — und trotzdem zeigt die
#   Relevanzkarte darunter deutliches Rot und einen weißen Kern. Das ist ein
#   sauberer Beleg für den Punkt aus Abschnitt 10:
#
#   Die `flat`-Regel der ersten Schicht setzt $a \equiv 1$ und $w \equiv 1$ und
#   verteilt Relevanz damit **unabhängig vom Voxelwert** über das rezeptive Feld.
#   Zusammen mit `padding='SAME'` und den Biases kann Relevanz in leere Regionen
#   fließen. Für Einsteiger ist das eine zentrale Lektion:
#
#   > Eine Heatmap zeigt nicht nur das Modell, sondern immer auch die gewählte
#   > Erklärungsregel. Was man sieht, ist $\text{Modell} \times \text{Regel}$.
#
#   Wer schärfere, dateninformierte Karten will, ersetzt die `flat`-Regel in der
#   Eingabeschicht durch die $z^\mathcal{B}$-Regel, die den erlaubten Wertebereich
#   der Eingabe berücksichtigt und außerhalb des Datenbereichs keine Relevanz
#   verteilt.
# * **In den mittleren Schichten** ($z=8\dots23$) zeigt sich das bekannte Muster:
#   blass-blaues Inneres (Gewebe senkt die Vorhersage), roter Ring auf und um die
#   Kontur, plus das Gitterartefakt. Die roten Bereiche liegen tendenziell dort, wo
#   die Buchten der breiten Tunnel in die Kugel schneiden — schwach, aber in der
#   richtigen Region.
# * **Symmetrie als Plausibilitätsprüfung:** Die Relevanzkarten am Anfang und am
#   Ende des Stapels sehen sich sehr ähnlich (beide leer, beide rot umrandet). Das
#   ist konsistent, weil die Kugel in $z$ ungefähr symmetrisch liegt — ein
#   nützlicher, kostenloser Sanity-Check: eine Erklärung, die für symmetrische
#   Eingaben stark unsymmetrisch ausfällt, wäre verdächtig.
#
# ### Praktischer Hinweis
#
# Die auskommentierte Zeile `#plt.savefig('standard.png')` verrät den Zweck dieser
# Abbildung: Sie ist die "Standard-LRP"-Referenzabbildung, gegen die in den
# folgenden Abschnitten die *kontrastive* Variante (RestructuredLRP) verglichen
# werden soll.

# %%
# Erstes Modell: Output=Dense(1), Bottleneck=Dense(32) ohne built-in Activation
# (RestructuredLRP verlangt lineare Bottleneck-Dense + optionale Activation/Dropout dazwischen)
output_layer = len(model.layers) - 1
bottleneck_layer = next(
    i for i, l in enumerate(model.layers)
    if isinstance(l, Dense) and int(l.units) == 32
)

explainer = LayerwiseRelevancePropagator(
    model, layer=output_layer, idx=0, strategy=strategy
)
explanations = explainer.predict(np.expand_dims(brain, 0))[0]
explanations = explanations / np.amax(np.abs(explanations))

fig, ax = plt.subplots(8, 8, figsize=(15, 8))

for i in range(0, 8, 2):
    for j in range(8):
        idx = ((i // 2) * 8)+ j
        
        ax[i][j].imshow(brain[idx], cmap='Greys_r')
        ax[i][j].axis('off')
        ax[i+1][j].imshow(explanations[idx], cmap='seismic', clim=(-1, 1))
        ax[i+1][j].axis('off')
        
#plt.savefig('standard.png')

plt.show()

# %% [markdown]
# <a id="sec14"></a>
# ## 14. Der Bottleneck als Repräsentation
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was wird berechnet?
#
# Hier wird das Modell **aufgeschnitten**. Mit
# `Model(model.input, model.layers[bottleneck_layer].output)` entsteht ein neues
# Keras-Modell, das dieselben Gewichte benutzt, aber nach der `Dense(32)`-Schicht
# aufhört. Der Fachbegriff dafür ist **Encoder**: er bildet ein Volumen auf einen
# 32-dimensionalen Vektor ab.
#
# $$z = \mathrm{enc}(X) \in \mathbb{R}^{32}, \qquad
#   \hat{y} = w^\top \mathrm{ReLU}(z) + b$$
#
# Die Interpretation dieser Zerlegung ist wichtig: Das gesamte Netz ist eine
# Kombination aus einem komplizierten, nichtlinearen **Merkmalsextraktor** und einer
# ganz einfachen linearen **Ausgabeschicht** darauf. Alles, was das Modell über ein
# Volumen "weiß", muss durch diesen 32-Zahlen-Engpass. Von $32\,768$ Voxeln auf 32
# Zahlen — eine Kompression um Faktor 1024.
#
# Dann wird eine **Referenzgruppe** gebildet: alle Testvolumen mit Label $y = 5$
# (also mittlere Tunnelbreite), und davon Mittelwert und Streuung pro Merkmal:
#
# $$\bar{z} = \frac{1}{|G|}\sum_{n \in G} z^{(n)}, \qquad
#   \sigma_k = \sqrt{\frac{1}{|G|}\sum_{n \in G}\big(z^{(n)}_k - \bar{z}_k\big)^2}$$
#
# ### Wozu dient das?
#
# $\bar{z}$ ist ein **Prototyp**: "so sieht ein durchschnittliches Gehirn mit
# Tunnelbreite 5 im Merkmalsraum aus". $\sigma$ beschreibt, wie stark einzelne
# Merkmale innerhalb dieser Gruppe schwanken, also was noch "normal" ist.
#
# Diese beiden Größen werden im nächsten Abschnitt gebraucht. Der Gedanke dahinter
# ist der Übergang von einer absoluten zu einer **kontrastiven** Erklärung:
#
# * Standard-LRP beantwortet: "*Warum sagt das Modell diesen Wert?*"
# * Kontrastive Erklärung beantwortet: "*Warum sagt das Modell einen **anderen**
#   Wert als bei einem typischen Gehirn der Referenzgruppe?*"
#
# Die zweite Frage ist in der Medizin fast immer die interessantere — man will nicht
# wissen, was normal ist, sondern was *abweicht*. Es ist auch die Frage, die
# Menschen von Natur aus stellen: Erklärungen sind fast immer kontrastiv
# ("*warum X und nicht Y?*").
#
# ### Ausgabe
#
# Diese Zelle plottet nichts; sie legt nur `mean_encoding` und `encoding_stddev` für
# die folgenden Zellen an. Die sichtbare Ausgabe sind lediglich die
# Fortschrittsbalken von `encoder.predict`.

# %%
encoder = Model(model.input, model.layers[bottleneck_layer].output)
encodings = encoder.predict(test_X)
group_idx = np.where(test_y == 5)[0]
group_encodings = encodings[group_idx]
mean_encoding = np.mean(group_encodings, axis=0)
encoding_stddev = np.std(group_encodings, axis=0)

# %% [markdown]
# <a id="sec15"></a>
# ## 15. Kontrastive Erklärungen mit RestructuredLRP (deaktiviert)
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Warum passiert hier nichts?
#
# Der Code dieser Zelle und der nächsten ist in `"""..."""` eingeschlossen, also ein
# reiner String und damit **wirkungslos**. Die einzige Ausgabe ist der String selbst.
# Der Grund ist technischer Natur: `RestructuredLRP` stellt zusätzliche
# Anforderungen an die Architektur (eine *lineare* Bottleneck-`Dense`-Schicht, davor
# und danach nur Aktivierung/Dropout) und ist mit der aktuellen Keras-Version
# offenbar noch nicht lauffähig. Die Zellen sind als Dokumentation der Absicht
# stehen geblieben.
#
# Für das Verständnis lohnt es sich trotzdem, das Konzept anzusehen — es ist der
# eigentliche methodische Beitrag dieses Repos.
#
# ### Die Idee von RestructuredLRP
#
# Standard-LRP verteilt die *ganze* Vorhersage $f(X)$ auf die Voxel. Wenn aber alle
# Gehirne einen ähnlichen Grundwert erzeugen (etwa "irgendwas um 5"), dann wird die
# Heatmap von diesem gemeinsamen Anteil dominiert, und der interessante,
# individuelle Teil geht unter. RestructuredLRP erklärt deshalb nicht $f(X)$,
# sondern die **Abweichung von einer Referenz**:
#
# $$\Delta = f(X) - f_{\text{ref}}
#   \quad\text{bzw. im Merkmalsraum}\quad
#   \delta_k = z_k - \bar{z}_k$$
#
# Technisch wird der Relevanzfluss am Bottleneck "umgebaut" (daher *restructured*):
# Statt der absoluten Aktivierung $z_k$ wird die Differenz $z_k - \bar{z}_k$ nach
# unten propagiert. Die resultierende Heatmap zeigt dann, welche Voxel dieses
# Gehirn *vom Prototyp unterscheiden*.
#
# Die zweite Zelle geht noch einen Schritt weiter (`threshold=True`) und übergibt
# zusätzlich `encoding_stddev`. Damit werden Merkmale, die innerhalb der normalen
# Streuung der Referenzgruppe liegen, **unterdrückt**. Der Gedanke entspricht einem
# z-Score mit Schwelle:
#
# $$\delta_k^{\text{thresh}} =
#   \begin{cases}
#     z_k - \bar{z}_k & \text{falls } |z_k - \bar{z}_k| > \tau\,\sigma_k \\
#     0 & \text{sonst}
#   \end{cases}$$
#
# So bleibt in der Heatmap nur, was **auffällig** ist — statistisches Rauschen
# innerhalb der Norm wird herausgefiltert.
#
# ### Was die Abbildung gezeigt hätte
#
# Die geplante Abbildung hat vier Zeilen, jeweils für die Schnitte $z=12\dots19$:
#
# | Zeile | Inhalt | Frage |
# |-------|--------|-------|
# | 1 | Volumen (grau) | Was ist zu sehen? |
# | 2 | Standard-LRP | Warum dieser Wert? |
# | 3 | RestructuredLRP | Warum anders als der Prototyp? |
# | 4 | Differenz der Zeilen 3 und 2 | Was fügt die kontrastive Sicht hinzu? |
#
# Die vierte Zeile ist die eigentlich diagnostische: Wo die Differenz null (weiß)
# ist, liefern beide Verfahren dasselbe; wo sie farbig ist, macht die kontrastive
# Sicht einen Unterschied. Erwartung wäre, dass die kontrastive Karte *sparsamer*
# aussieht und stärker auf die Tunnelränder fokussiert, weil der überall gleiche
# "Kugel-Grundanteil" herausgerechnet ist.
#
# ### Der größere Kontext
#
# Kontrastive Erklärungen relativ zu einer gesunden Referenzgruppe sind in der
# medizinischen Bildanalyse das methodische Ideal: Ein Radiologiebericht sagt nicht
# "das ist ein Gehirn", sondern "*hier weicht es von der Norm ab*". Verwandte
# Konzepte, denen man in der Literatur begegnet, sind *Deep Taylor Decomposition*
# mit Wurzelpunkt (root point), *Integrated Gradients* mit Baseline und
# *counterfactual explanations*. Allen gemeinsam ist die Frage: **Erklärung im
# Vergleich zu was?** Genau diese Frage bleibt bei naiver Anwendung von
# Attributionsmethoden meist unbeantwortet.

# %%
"""
from explainability import RestructuredLRP
    
restructured_lrp = RestructuredLRP(
    model, layer=output_layer, idx=0, bottleneck=bottleneck_layer, strategy=strategy
)
restructured_explanations = restructured_lrp.predict([np.expand_dims(brain, 0), 
                                                      np.expand_dims(mean_encoding, 0)])[0]
restructured_explanations = restructured_explanations / np.amax(np.abs(restructured_explanations))

fig, ax = plt.subplots(4, 8, figsize=(15, 4))


for i in range(8):
    ax[0][i].imshow(brain[12+i], cmap='Greys_r')
    ax[0][i].axis('off')
    ax[1][i].imshow(explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[1][i].axis('off')
    ax[2][i].imshow(restructured_explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[2][i].axis('off')
    ax[3][i].imshow(restructured_explanations[12+i] - explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[3][i].axis('off')

plt.show()
"""

# %% [markdown]
# ### 15b. Dieselbe Idee mit Schwellenwert (ebenfalls deaktiviert)
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Diese Zelle ist die Variante mit `threshold=True`. Der einzige Unterschied zur
# vorherigen: es wird zusätzlich `encoding_stddev` übergeben, sodass Merkmale
# innerhalb der normalen Streuung der Referenzgruppe unterdrückt werden — die im
# Abschnitt oben beschriebene z-Score-Schwelle. Erwartetes Ergebnis wäre eine
# deutlich **sparsamere** Heatmap, die nur noch wirklich auffällige Regionen zeigt.
#
# Auch dieser Code ist als String deaktiviert und wird nicht ausgeführt.

# %%
"""
from explainability import RestructuredLRP
    
restructured_lrp = RestructuredLRP(
    model, layer=output_layer, idx=0, bottleneck=bottleneck_layer,
    strategy=strategy, threshold=True
)
restructured_explanations = restructured_lrp.predict([np.expand_dims(brain, 0), 
                                                      np.expand_dims(mean_encoding, 0),
                                                      np.expand_dims(encoding_stddev, 0)])[0]
restructured_explanations = restructured_explanations / np.amax(np.abs(restructured_explanations))

fig, ax = plt.subplots(4, 8, figsize=(15, 4))


for i in range(8):
    ax[0][i].imshow(brain[12+i], cmap='Greys_r')
    ax[0][i].axis('off')
    ax[1][i].imshow(explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[1][i].axis('off')
    ax[2][i].imshow(restructured_explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[2][i].axis('off')
    ax[3][i].imshow(restructured_explanations[12+i] - explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[3][i].axis('off')

plt.show()
"""

# %% [markdown]
# <a id="sec16"></a>
# ## 16. Neuronenweise LRP im Bottleneck
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert in dieser Zelle?
#
# Bisher wurde immer *die Vorhersage* erklärt. Jetzt wird eine Schicht tiefer
# gebohrt: Was kodiert eigentlich **jedes einzelne** der 32 Bottleneck-Neuronen?
#
# Dazu wird das Teilmodell `tmp` gebaut, das bei `Dense(32)` endet, und dann in einer
# Schleife für jedes Neuron $i = 0,\dots,31$ ein eigener Erklärer erzeugt:
#
# ```python
# LayerwiseRelevancePropagator(tmp, layer=len(tmp.layers)-1, idx=i, strategy=strategy)
# ```
#
# Das Argument `idx=i` wählt aus, *welches* Ausgabeneuron erklärt wird. Intern
# maskiert LRP alle anderen Ausgaben auf null und propagiert nur die Aktivierung von
# Neuron $i$ zurück. Das Ergebnis ist eine Heatmap pro Neuron:
#
# $$R^{(i)} = \mathrm{LRP}\big(z_i\big), \qquad
#   \sum_{\text{Voxel } v} R^{(i)}_v \approx z_i$$
#
# Man nennt das *feature visualization by attribution* — man fragt nicht "*warum
# diese Vorhersage*", sondern "*wann feuert dieses Neuron*".
#
# **Warum hat die Strategie hier nur fünf statt sechs Einträge?** Weil das
# Teilmodell eine Schicht kürzer ist: 4 Conv-Schichten + `Dense(32)` = 5
# Standard-LRP-Schichten. Die `Dense(1)`-Ausgabeschicht und damit ihr
# $\varepsilon$-Eintrag fehlen. Die Bibliothek prüft das mit einem `assert` — eine
# falsche Anzahl führt sofort zum Fehler und nicht zu stillschweigend falschen
# Ergebnissen.
#
# **Die Abbruchbedingung `if np.sum(explanations) == 0: continue`** überspringt
# Neuronen ohne jede Relevanz, also "tote" Einheiten. In diesem Durchlauf wurden
# **alle 32 Karten geplottet** — es gibt hier also keine toten Neuronen (das ändert
# sich in Modell 2, siehe Abschnitt 22).
#
# ### Die Grafiken
#
# Zuerst kommt eine Referenzreihe mit den 8 Graustufen-Schnitten des Volumens (das
# Volumen aus Abschnitt 12, mit gemischten Tunnelbreiten). Danach folgen **32
# Reihen**, jeweils eine pro Bottleneck-Neuron, wieder mit den Schnitten
# $z = 12\dots19$ und der Farbskala rot/weiß/blau.
#
# ### Interpretation
#
# * **Der frappierende Befund: fast alle 32 Karten sehen gleich aus.** Sie zeigen
#   dasselbe räumliche Muster — ein Kleeblatt aus Relevanz, das der Gewebeverteilung
#   folgt, mit hellen Aussparungen an den Tunneln. Der einzige echte Unterschied
#   zwischen den Neuronen ist das **Vorzeichen**: manche Karten sind durchgehend blau
#   (z. B. die erste), andere durchgehend rot (z. B. die späteren), und die
#   Intensität variiert.
# * **Das bedeutet massive Redundanz.** Wenn 32 Neuronen im Wesentlichen dieselbe
#   räumliche Funktion mit unterschiedlichem Vorzeichen und Maßstab berechnen, dann
#   ist die effektive Dimension der Repräsentation viel kleiner als 32 —
#   vermutlich nahe **1**: "wie viel Gewebe ist da". Das würde auch erklären, warum
#   das Modell die Aufgabe noch nicht gut löst: eine einzige Zahl "Gewebemenge"
#   verwechselt breite Tunnel mit einer kleinen Kugel.
# * **Es gibt eine architektonische Erklärung dafür**, und die ist lehrreich. Vor
#   `Dense(32)` steht `GlobalAveragePooling3D`. Jedes Bottleneck-Neuron ist damit
#   nur eine Linearkombination von 64 Kanalmittelwerten:
#
#   $$z_i = \sum_{c=1}^{64} W_{ci}\, g_c + b_i,
#     \qquad g_c = \text{Mittelwert von Kanal } c$$
#
#   Und weil LRP linear rückverteilt, ist auch jede Neuronen-Heatmap eine
#   Linearkombination derselben 64 Kanal-Relevanzmuster. Die 32 Karten *können* also
#   gar nicht beliebig verschieden sein — sie leben in einem 64-dimensionalen Raum
#   fester Basismuster. Dass sie sich fast nur im Vorzeichen unterscheiden, heißt,
#   dass die Gewichtsvektoren $W_{\cdot i}$ nach zwei Epochen noch stark
#   kollinear sind.
# * **Konsequenz:** Global Average Pooling ist gut für Translationsinvarianz, aber
#   schlecht für vielfältige, ortsaufgelöste Merkmale. Genau darum wird in
#   Abschnitt 18 ein zweites Modell ohne GAP gebaut.
#
# ### Der größere Kontext
#
# Neuronenweise Analyse ist die Brücke zwischen Attributionsmethoden (die eine
# *einzelne Vorhersage* erklären) und *mechanistischer Interpretierbarkeit* (die
# fragt, welche Konzepte ein Netz intern repräsentiert). Verwandte Verfahren:
# *Network Dissection*, *TCAV* (Testing with Concept Activation Vectors) und
# *Feature Visualization* durch Optimierung der Eingabe. Die praktische Frage
# dahinter ist immer dieselbe: Hat das Netz eine reichhaltige, faktorisierte
# Repräsentation gelernt — oder nur einen einzigen Summenindikator?

# %%
tmp = Model(model.input, model.layers[bottleneck_layer].output)

strategy = LRPStrategy(
    layers=[
        {'flat': True},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
    ]
)


fig, ax = plt.subplots(1, 8, figsize=(15, 3))

for i in range(8):
    ax[i].imshow(brain[12+i], cmap='Greys_r')
    ax[i].axis('off')
    
plt.show()

for i in range(32):
    explainer = LayerwiseRelevancePropagator(
        tmp, layer=len(tmp.layers) - 1, idx=i, strategy=strategy
    )
    explanations = explainer.predict(np.expand_dims(brain, 0))[0]

    if np.sum(explanations) == 0:
        continue
    
    explanations = explanations / np.amax(np.abs(explanations))

    fig, ax = plt.subplots(1, 8, figsize=(15, 8))

    for j in range(8):
        ax[j].imshow(explanations[12+j], cmap='seismic', clim=(-1, 1))
        ax[j].axis('off')

    plt.show()

# %% [markdown]
# <a id="sec17"></a>
# ## 17. Wie redundant sind die 32 Features? Die Korrelationsmatrix
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was wird berechnet?
#
# Der Verdacht aus Abschnitt 16 (die Bottleneck-Neuronen sind redundant) wird jetzt
# quantitativ geprüft. Für alle 600 Trainingsvolumen werden die Encodings berechnet
# und daraus paarweise die **Pearson-Korrelation** zwischen je zwei Neuronen $i$ und
# $j$ über die Volumen hinweg:
#
# $$r_{ij} = \frac{\operatorname{Cov}(z_i, z_j)}{\sigma_{z_i}\,\sigma_{z_j}}
#          = \frac{\sum_n (z^{(n)}_i - \bar{z}_i)(z^{(n)}_j - \bar{z}_j)}
#                 {\sqrt{\sum_n (z^{(n)}_i - \bar{z}_i)^2}\sqrt{\sum_n (z^{(n)}_j - \bar{z}_j)^2}}$$
#
# Das Ergebnis ist eine symmetrische $32\times32$-Matrix mit Werten in $[-1, 1]$ und
# Einsen auf der Diagonale. Sie wird als Bild dargestellt.
#
# ### Wie man die Grafik liest
#
# * Achsen: Neuronenindex 0 bis 31, beide Achsen identisch belegt.
# * Farbskala `viridis` mit `clim=(0, 1)`: **gelb $= r \approx 1$** (perfekt
#   gleichläufig), **grün $\approx 0.7$**, **dunkelblau/violett $= r \le 0$**.
# * Wichtige Einschränkung: Wegen `clim=(0, 1)` werden **alle negativen
#   Korrelationen auf dieselbe dunkle Farbe geklemmt**. Ein Neuronenpaar mit
#   $r = -0.98$ (also perfekt gegenläufig und damit genauso redundant!) sieht
#   identisch aus wie ein Paar mit $r = 0$ (unabhängig). Wer Redundanz messen will,
#   sollte $|r|$ plotten oder die Skala auf $[-1, 1]$ setzen. Das ist beim Lesen
#   dieses Bildes unbedingt zu beachten.
#
# ### Interpretation
#
# * **Große gelbe Blöcke.** Besonders auffällig ist ein zusammenhängender gelber
#   Block etwa bei den Indizes 5–10: dort sind *alle* Paare fast perfekt korreliert.
#   Diese sechs Neuronen tragen praktisch dieselbe Information. Auch außerhalb des
#   Blocks ziehen sich helle Zeilen und Spalten durch das Bild.
# * **Das Schachbrettmuster** entsteht dadurch, dass die Neuronen in *zwei Gruppen*
#   fallen, die untereinander stark positiv korreliert und zwischen den Gruppen
#   antikorreliert sind (dunkel, weil geklemmt). Man sieht hier also im Grunde eine
#   einzige dominante Richtung, an der alle Neuronen mit unterschiedlichem
#   Vorzeichen hängen — exakt das Bild, das die Heatmaps in Abschnitt 16
#   vorhergesagt haben (überall dasselbe Muster, nur rot oder blau).
# * **Nur ein kleiner Bereich ist wirklich diagonal-dominiert:** um die Indizes
#   11–15 ist die Diagonale gelb, die Umgebung aber dunkel. Diese Neuronen tragen
#   individuellere Information.
# * **Fazit:** Die effektive Dimensionalität des Bottlenecks ist weit kleiner als 32.
#   Das ist bei einem 2-Epochen-Modell nicht überraschend — die Gewichte sind kaum
#   von ihrer Initialisierung wegdiversifiziert.
#
# ### Wie würde man das quantifizieren?
#
# Die Heatmap ist qualitativ. Für eine Zahl bietet sich das Spektrum der
# Korrelationsmatrix an. Mit den Eigenwerten $\lambda_1 \ge \dots \ge \lambda_{32}$
# und $p_k = \lambda_k / \sum_j \lambda_j$ ist die *effektive Dimension*
# (Beispiel: Entropie-basiert)
#
# $$d_{\text{eff}} = \exp\left(-\sum_{k} p_k \log p_k\right)$$
#
# Bei perfekter Redundanz gilt $d_{\text{eff}} \to 1$, bei völlig unabhängigen
# Merkmalen $d_{\text{eff}} \to 32$. Alternativ genügt oft schon eine PCA: "*wie
# viele Komponenten erklären 95 % der Varianz?*"
#
# ### Der größere Kontext
#
# Redundanz im Bottleneck hat direkte Folgen für die Erklärbarkeit. Wenn mehrere
# Neuronen dasselbe kodieren, ist die Aussage "*Neuron 7 kodiert Tunnelbreite*"
# nicht identifizierbar — man könnte die Information beliebig zwischen den
# korrelierten Neuronen umverteilen, ohne die Vorhersage zu ändern. Erklärungen auf
# Neuronenebene sind also nur dann sinnvoll, wenn die Repräsentation halbwegs
# **entwirrt** (disentangled) ist. Genau deshalb steht diese Analyse im Notebook: sie
# ist die Voraussetzungsprüfung für alles, was man über einzelne Neuronen behaupten
# möchte.

# %%
train_encodings = encoder.predict(train_X)

correlations = [[np.corrcoef(train_encodings[:,i], train_encodings[:,j])[0,1] \
                 for i in range(32)] for j in range(32)]

plt.figure(figsize=(10, 10))
heatmap = plt.imshow(correlations, clim=(0, 1))
plt.colorbar(heatmap)
plt.show()

# %% [markdown]
# <a id="sec18"></a>
# ## 18. Modell 2: ohne Global Average Pooling, mit fixiertem Bias
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Die Analysen der Abschnitte 13 bis 17 haben zwei Schwächen von Modell 1 aufgedeckt:
#
# 1. Die Heatmaps sind räumlich unscharf, weil Global Average Pooling die
#    Ortsinformation wegmittelt.
# 2. Die Bottleneck-Repräsentation ist stark redundant.
#
# Modell 2 ist die architektonische Antwort darauf. Es ist **kein** besseres Modell
# im Sinne von "genauer", sondern ein **erklärbarkeitsfreundlicheres**.
#
# ### Die beiden entscheidenden Änderungen
#
# **Änderung 1: `Reshape` statt `GlobalAveragePooling3D`.**
#
# ```
# Modell 1:  ... → (4,4,4,128) → GlobalAveragePooling3D → (64)
# Modell 2:  ... → (4,4,4,128) → Reshape → (8192)
# ```
#
# Anstatt jeden Kanal auf seinen Mittelwert zu reduzieren, wird die komplette
# Feature-Map "plattgemacht": $4 \cdot 4 \cdot 4 \cdot 128 = 8192$ Werte gehen
# vollständig in die erste Dense-Schicht. Konsequenzen:
#
# | | Modell 1 (GAP) | Modell 2 (Reshape) |
# |---|---|---|
# | Ortsinformation | verworfen | erhalten |
# | Translationsinvarianz | ja, per Konstruktion | nein, muss gelernt werden |
# | Parameter | 289 089 | **1 348 865** |
# | Datenbedarf | niedriger | höher |
# | Heatmap-Schärfe | gering | potenziell hoch |
#
# Allein die Schicht `Dense(128)` nach dem Reshape kostet
# $8192 \cdot 128 + 128 = 1\,048\,704$ Parameter — drei Viertel des gesamten
# Modells. Das ist der klassische Preis, den man für erhaltene Ortsinformation
# zahlt, und der Grund, warum moderne Architekturen normalerweise GAP verwenden.
# Hier nehmen wir ihn bewusst in Kauf, weil LRP nur dann lokalisieren kann, wenn
# Ortsinformation die letzten Schichten überhaupt erreicht.
#
# **Änderung 2: der auf 5 fixierte Ausgabe-Bias.**
#
# ```python
# Dense(1, activation=None, bias_initializer=Constant([5.]),
#       bias_constraint=MinMaxNorm(min_value=5.0, max_value=5.0))
# ```
#
# Der Bias der Ausgabeschicht wird auf $5$ initialisiert **und** durch einen
# Constraint dort festgenagelt (Minimum = Maximum = 5, also unveränderlich). Das
# Modell berechnet damit
#
# $$f(X) = 5 + w^\top h(X)$$
#
# Das ist ein hübscher und lehrreicher Trick mit zwei Effekten:
#
# * **Trainingstechnisch:** $5$ liegt sehr nahe am Mittelwert der Zielverteilung
#   ($\mathbb{E}[y] = 5.5$). Das Modell startet also bereits bei der
#   Mittelwert-Baseline und muss nur noch die *Abweichung* davon lernen. Das erklärt
#   den in Abschnitt 19 sichtbaren, viel besseren Startverlust. Genau dieses
#   Vorgehen (Bias auf den Zielmittelwert initialisieren) ist bei Regression
#   generell empfehlenswert.
# * **Erklärungstechnisch:** Der konstante Anteil $5$ trägt keine Relevanz, denn LRP
#   verteilt nur, was durch die Gewichte fließt. Alles, was in der Heatmap
#   erscheint, gehört damit zur Abweichung $w^\top h(X)$ vom Mittelwert. Das Modell
#   ist also **per Konstruktion kontrastiv** — die Heatmaps sind ein Stück weit das,
#   was RestructuredLRP nachträglich erreichen wollte.
#
# ### Weitere Unterschiede im Detail
#
# * Nur noch **drei** Conv-Blöcke; die $1\times1\times1$-Conv-Schicht fehlt.
# * Dafür ein tieferer Kopf: `Dense(128) → BN → ReLU → Dense(128) → BN → ReLU →
#   Dense(32) → ReLU → Dense(1)`. BatchNorm auch zwischen den Dense-Schichten, weil
#   ein so tiefer, voll verbundener Kopf sonst schlecht konvergiert.
# * `Dropout(0.5)` sitzt jetzt *vor* dem ersten `Dense`, also auf 8192
#   Aktivierungen. Das ist sinnvoll, weil genau dort die Overfitting-Gefahr durch die
#   Million Parameter am größten ist.
# * Der Code-Kommentar `Kein BN nach Pooling — fuse_batchnorm kann nur
#   Conv/Dense→BN fusionieren` dokumentiert eine **Einschränkung der
#   LRP-Bibliothek**, die die Architektur mitbestimmt: BatchNorm darf nur direkt
#   hinter einer Conv- oder Dense-Schicht stehen, damit sie beim Erklären
#   hineingerechnet werden kann. Ein gutes Beispiel dafür, dass Erklärbarkeit
#   *Designentscheidungen erzwingt* und nicht nachträglich aufgesetzt werden kann.
#
# ### Die Ausgabe: `model.summary()`
#
# **1 348 865 Parameter**, also 4.7-mal so viele wie Modell 1. Man beachte in der
# Tabelle die Verschiebung des Schwerpunkts: Bei Modell 1 saßen drei Viertel der
# Parameter in den Conv-Schichten (Merkmalsextraktion), bei Modell 2 sitzen drei
# Viertel im voll verbundenen Kopf (Klassifikation/Regression aus fertigen
# Merkmalen). Die Formen zeigen den Schnitt sauber:
# $(4,4,4,128) \to (8192) \to (128) \to (128) \to (32) \to (1)$.
#
# > **Achtung, eine Notebook-Falle:** Die Variable `model` wird hier
# > **überschrieben**. Alles, was oben mit `model` gemacht wurde (Encoder,
# > Erklärer), bezieht sich ab dieser Zelle auf das *neue* Netz. Führt man die
# > oberen Zellen jetzt noch einmal aus, erklärt man ungewollt Modell 2. Das ist
# > eine der häufigsten Fehlerquellen beim Arbeiten mit Notebooks.

# %%
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import GlobalMaxPooling3D, Reshape


np.random.seed(42)
tf.random.set_seed(42)

regularizer = l2(1e-3)
depths = [32, 64, 128, 256, 256, 64]
activation='relu'
dropout=0.5

inputs = Input((IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
x = inputs

for i in range(3):
    x = Conv3D(depths[i], (3, 3, 3), padding='SAME',
               activation=None, kernel_regularizer=regularizer,
               bias_regularizer=regularizer)(x)

    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPooling3D((2, 2, 2))(x)

# Kein BN nach Pooling — fuse_batchnorm kann nur Conv/Dense→BN fusionieren
x = Activation(activation)(x)
x = Reshape((-1,))(x)
x = Dropout(dropout)(x)

x = Dense(128, activation=None)(x)
x = BatchNormalization()(x)
x = Activation(activation)(x)
x = Dense(128, activation=None)(x)
x = BatchNormalization()(x)
x = Activation(activation)(x)

x = Dense(32, activation=None)(x)
x = Activation('relu')(x)

x = Dense(1, activation=None, bias_initializer=Constant([5.]), 
          bias_constraint=MinMaxNorm(min_value=5.0, max_value=5.0))(x)

model = Model(inputs, x)

model.summary()

# %% [markdown]
# <a id="sec19"></a>
# ## 19. Training von Modell 2
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Identisches Trainingsrezept wie in Abschnitt 8: MSE-Verlust, Adam mit
# $10^{-3}$, dieselben Callbacks, dieselben Daten, `epochs=2`. Auch die Seeds sind
# gleich gesetzt. Das ist gute Experimentierpraxis: Wenn man eine Architektur
# vergleichen will, muss **alles andere gleich bleiben**, sonst weiß man nicht,
# woran ein Unterschied liegt.
#
# Auch hier gibt es den Caching-Zweig aus Abschnitt 8, diesmal mit einem eigenen
# Verzeichnis für Modell 2. Existiert dort schon eine `.keras`-Datei, wird geladen
# statt trainiert, und es erscheint keine Lernkurve.
#
# ### Die Grafik: Lernkurven im direkten Vergleich
#
# | Epoche | Modell 1 Train | Modell 1 Val | **Modell 2 Train** | **Modell 2 Val** |
# |--------|---------------|--------------|--------------------|-------------------|
# | 1 | 30.80 | 36.48 | **5.93** | **7.34** |
# | 2 | 17.45 | 31.92 | **3.01** | **6.63** |
#
# Zur Erinnerung die Referenz: Mittelwert-Baseline $= \operatorname{Var}(y) = 8.25$.
#
# ### Interpretation
#
# * **Modell 2 startet dort, wo Modell 1 nach zwei Epochen noch nicht angekommen
#   ist.** Der Trainingsverlust beginnt bei 5.93 statt 30.8 — ein Faktor 5. Der
#   Grund ist nicht die bessere Architektur, sondern der **auf 5 fixierte Bias**: das
#   Modell fängt bereits bei einer sinnvollen Vorhersage an und muss den Mittelwert
#   nicht erst mühsam lernen. Das ist die praktische Lehre dieser Zelle: eine gut
#   gewählte Initialisierung kann viele Epochen Training ersetzen.
# * **Beide Werte liegen jetzt unter der Baseline von 8.25** (Training 3.01,
#   Validierung 6.63). Modell 2 ist damit nach zwei Epochen schon *besser als der
#   triviale Mittelwertschätzer* — Modell 1 war nach zwei Epochen deutlich
#   schlechter. Aus dem Trainingslog liest man einen MAE um 2.0–2.3, also
#   "durchschnittlich 2 Einheiten Tunnelbreite daneben" bei einem Wertebereich von
#   1–10. Brauchbar ist das noch nicht, aber die Größenordnung stimmt endlich.
# * **Die Lücke zwischen Training (3.01) und Validierung (6.63) ist auffällig
#   größer als bei Modell 1.** Das ist genau das erwartete Verhalten eines Modells
#   mit 4.7-mal so vielen Parametern und ohne eingebaute Translationsinvarianz: es
#   passt sich schneller an die Trainingsdaten an. Mit 500 Epochen wäre hier die
#   Overfitting-Frage der zentrale Punkt der Auswertung — und genau dafür sind
#   `Dropout(0.5)`, die L2-Regularisierung und `EarlyStopping` eingebaut.
# * **Vorsicht bei der Schlussfolgerung:** Aus zwei Epochen darf man *nicht*
#   ableiten, dass Modell 2 grundsätzlich besser ist. Ein Großteil des Vorsprungs
#   ist der Bias-Trick, der auch in Modell 1 möglich gewesen wäre. Der faire
#   Vergleich der Architekturen wäre: gleicher Bias-Trick in beiden, 500 Epochen,
#   mehrere Seeds.

# %%
np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss='mse', optimizer=Adam(1e-3), metrics=['mae'])

callbacks = [
    ReduceLROnPlateau(
        monitor="loss",
        factor=0.1,
        patience=10,
        min_lr=1e-5
    ),
    EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=50,
        restore_best_weights=True
    )
]

MODEL_DIR = target_dir / "model_2_no_gap"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "brain_regression_no_gap.keras"

existing_model_path = next(iter(sorted(MODEL_DIR.glob("*.keras"))), None)

if existing_model_path is not None:
    model = load_model(existing_model_path)
    history = None
    print(f"Model geladen von: {existing_model_path}")
else:
    history = model.fit(train_X, train_y, 
                        validation_data=(val_X, val_y), 
                        batch_size=32,
                        epochs=2,
                        #epochs=500,
                        callbacks=callbacks)

    model.save(MODEL_PATH)
    print(f"Model gespeichert unter: {MODEL_PATH}")


if history is None:
    print("Keine Lernkurven: das Modell wurde geladen, nicht trainiert.")
else:
    traces = [
        go.Scatter(
            x=np.arange(len(history.history['loss'])),
            y=history.history['loss'],
            name='Training loss'
        ),
        go.Scatter(
            x=np.arange(len(history.history['loss'])),
            y=history.history['val_loss'],
            name='Validation loss'
        )
    ]

    iplot(go.Figure(traces))

# %% [markdown]
# <a id="sec20"></a>
# ## 20. LRP für Modell 2
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was ist an der Strategie neu?
#
# Die Strategie hat jetzt **sieben** Einträge statt sechs, weil Modell 2 sieben
# Standard-LRP-Schichten hat (3 Conv + 4 Dense):
#
# | # | Schicht | Regel |
# |---|---------|-------|
# | 1 | Conv3D(32) | `flat` |
# | 2 | Conv3D(64) | $\alpha=2,\beta=1$ |
# | 3 | Conv3D(128) | $\alpha=2,\beta=1$ |
# | 4 | Dense(128) | $\alpha=2,\beta=1$ |
# | 5 | Dense(128) | $\alpha=2,\beta=1$ |
# | 6 | Dense(32) | $\varepsilon=0.25$ |
# | 7 | Dense(1) | $\varepsilon=0.25$ |
#
# Das Muster ist typisch und gut zu merken: **`flat` in der Eingabeschicht,
# $\alpha\beta$ im Faltungsteil, $\varepsilon$ im Kopf.** Begründung: Im Kopf sind
# die Aktivierungen abstrakt und wenige, dort will man nur numerisch stabilisieren;
# im Faltungsteil braucht man die kontrastreiche Trennung positiver und negativer
# Beiträge; in der Eingabeschicht will man Lesbarkeit. `Reshape`, `Dropout`,
# `Pooling` und `Activation` zählen nicht mit, `BatchNormalization` wird in die
# vorhergehende Schicht fusioniert.
#
# Erklärt wird wieder dasselbe Volumen wie in Abschnitt 13, sodass die Abbildungen
# direkt vergleichbar sind. Das Layout ist identisch ($8\times8$, abwechselnd
# Volumen- und Relevanzzeilen, alle 32 Schichten).
#
# ### Interpretation im Vergleich zu Modell 1
#
# * **Der Vorzeichenwechsel ist der auffälligste Unterschied.** Bei Modell 1 war das
#   Gewebe blau (negative Relevanz) mit rotem Rand. Bei Modell 2 ist die Karte
#   **überwiegend rot**, mit hellen bis weißen Aussparungen dort, wo die Tunnel
#   liegen, und nur wenigen schwach blauen Stellen.
#
#   Das ist kein Widerspruch, sondern eine Folge des fixierten Bias. Modell 2
#   berechnet $f(X) = 5 + w^\top h(X)$, und die Relevanz erklärt nur den Term
#   $w^\top h(X)$, also die **Abweichung vom Mittelwert 5**. Da für dieses stark
#   ausgehöhlte Volumen eine Vorhersage *oberhalb* von 5 herauskommt, muss die
#   Relevanzsumme positiv sein — überwiegend Rot ist also rechnerisch erwartbar.
#   Die inhaltliche Frage lautet nicht mehr "rot oder blau", sondern **"wo genau
#   sitzt das Rot"**.
# * **Und dort ist Modell 2 sichtbar besser lokalisiert.** Die roten Bereiche folgen
#   erkennbar den **Rändern der Kugel und den Tunnelwänden**; das Innere des intakten
#   Gewebes bleibt blasser. Bei Modell 1 war die Relevanz eine großflächige,
#   diffuse Wolke über der gesamten Scheibe. Das ist genau der erhoffte Effekt des
#   Verzichts auf Global Average Pooling: Ortsinformation erreicht den Kopf und LRP
#   kann sie zurückverfolgen. Und es ist die inhaltlich *richtige* Region — die
#   Breite eines Tunnels ist eine Eigenschaft seiner Wände.
# * **Die Artefakte bleiben.** Auch hier tragen die leeren Randschichten ($z=0,1$
#   und $z=30,31$) noch Relevanz, und das Punktraster aus den Pooling-Schritten ist
#   weiterhin sichtbar. Das bestätigt die Diagnose aus Abschnitt 13: Diese Artefakte
#   stammen aus der `flat`-Regel und der Pooling-Struktur, **nicht** aus der
#   spezifischen Architektur des Kopfes — sie überleben den Architekturwechsel.
# * **Ehrliche Einordnung:** Auch dieses Modell ist nach zwei Epochen weit von
#   "austrainiert" entfernt. Dass die Lokalisierung trotzdem plausibler wird, ist
#   ein Argument dafür, dass die Architekturwahl — und nicht nur die Trainingsdauer —
#   die Qualität von Erklärungen mitbestimmt.

# %%
from explainability import LayerwiseRelevancePropagator

strategy = LRPStrategy(
    layers=[
        {'flat': True},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'epsilon': 0.25},
        {'epsilon': 0.25}
    ]
)

# Zweites Modell: Output=Dense(1), Bottleneck=Dense(32) (linear, für RestructuredLRP)
output_layer = len(model.layers) - 1
bottleneck_layer = next(
    i for i, l in enumerate(model.layers)
    if isinstance(l, Dense) and int(l.units) == 32
)

explainer = LayerwiseRelevancePropagator(
    model, layer=output_layer, idx=0, strategy=strategy
)
explanations = explainer.predict(np.expand_dims(brain, 0))[0]
explanations = explanations / np.amax(np.abs(explanations))

fig, ax = plt.subplots(8, 8, figsize=(15, 8))

for i in range(0, 8, 2):
    for j in range(8):
        idx = ((i // 2) * 8)+ j
        
        ax[i][j].imshow(brain[idx], cmap='Greys_r')
        ax[i][j].axis('off')
        ax[i+1][j].imshow(explanations[idx], cmap='seismic', clim=(-1, 1))
        ax[i+1][j].axis('off')

plt.show()

# %% [markdown]
# <a id="sec21"></a>
# ## 21. Neuronenweise LRP für Modell 2
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Dieselbe Analyse wie in Abschnitt 16, jetzt für Modell 2: für jedes der 32
# Bottleneck-Neuronen eine eigene Heatmap. Die Strategie hat sechs Einträge, weil
# das Teilmodell bis `Dense(32)` sechs Standard-LRP-Schichten enthält (3 Conv +
# `Dense(128)` + `Dense(128)` + `Dense(32)`); der Eintrag für die
# `Dense(1)`-Ausgabeschicht fällt weg.
#
# Der Sinn der Wiederholung: **Hat der Architekturwechsel die Redundanz
# aufgelöst?** Bei Modell 1 waren alle 32 Karten praktisch identisch, weil Global
# Average Pooling jedes Bottleneck-Neuron auf eine Linearkombination von nur 64
# Kanalmittelwerten reduzierte. Ohne GAP steht dem Kopf jetzt der volle
# $8192$-dimensionale Merkmalsvektor zur Verfügung — die Neuronen *könnten* also
# ortsspezifisch werden.
#
# ### Interpretation
#
# * **Die räumliche Struktur ist deutlich anders als bei Modell 1.** Statt der
#   diffusen Kleeblatt-Wolke zeigen die Karten nun einen **Ring entlang der
#   Kugelkontur** mit hellen Kernen im Inneren und kräftigen Punkten dort, wo Tunnel
#   die Oberfläche durchbrechen. Die Relevanz sitzt also an *Kanten*, was für die
#   Aufgabe die richtige Struktur ist.
# * **Die Redundanz ist aber nicht verschwunden.** Auch hier gleichen sich die 32
#   Karten im Muster stark und unterscheiden sich vor allem im **Vorzeichen** —
#   einige Neuronen liefern rote Ringe, andere fast pixelgleiche blaue Ringe.
#   Zwei Neuronen, deren Heatmaps sich nur im Vorzeichen unterscheiden, sind
#   informationstheoretisch dasselbe Merkmal.
# * **Warum bleibt die Redundanz?** Vermutlich schlicht wegen der zwei Epochen: Die
#   Gewichte sind noch nahe der Zufallsinitialisierung, und die Gradienten für alle
#   32 Neuronen zeigen anfangs in ähnliche Richtungen (alle bekommen ihr Signal
#   über denselben skalaren Fehler der Ausgabeschicht). Ausdifferenzierung braucht
#   Training. Ein Gegentest wäre, dieselbe Analyse nach 500 Epochen zu wiederholen
#   und die Vielfalt der Karten zu vergleichen.
# * **Auch hier keine übersprungene Karte:** Das `if np.sum(explanations) == 0`
#   greift nicht, es werden alle 32 Karten gezeichnet. Auf der *linearen*
#   `Dense(32)`-Schicht gibt es also keine toten Einheiten. Das ist ein wichtiger
#   Kontrast zur nächsten Zelle, die die *ReLU*-Ausgabe derselben Schicht
#   betrachtet — und dort sehr wohl tote Neuronen findet.

# %%
tmp = Model(model.input, model.layers[bottleneck_layer].output)

strategy = LRPStrategy(
    layers=[
        {'flat': True},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'epsilon': 0.25}
    ]
)

fig, ax = plt.subplots(1, 8, figsize=(15, 3))

for i in range(8):
    ax[i].imshow(brain[12+i], cmap='Greys_r')
    ax[i].axis('off')
    
plt.show()

for i in range(32):
    explainer = LayerwiseRelevancePropagator(tmp, layer=len(tmp.layers)-1, idx=i, strategy=strategy)
    explanations = explainer.predict(np.expand_dims(brain, 0))[0]

    if np.sum(explanations) == 0:
        continue
    
    explanations = explanations / np.amax(np.abs(explanations))

    fig, ax = plt.subplots(1, 8, figsize=(15, 8))

    for j in range(8):
        ax[j].imshow(explanations[12+j], cmap='seismic', clim=(-1, 1))
        ax[j].axis('off')

    plt.show()

# %% [markdown]
# <a id="sec22"></a>
# ## 22. Korrelationen in Modell 2 — und tote ReLU-Neuronen
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Ein kleiner, aber wichtiger Unterschied zu Abschnitt 17
#
# Der Encoder wird hier über `model.layers[-2].output` gebaut. Das ist **nicht**
# die `Dense(32)`-Schicht, sondern die `Activation('relu')` direkt danach:
#
# ```
# ... → Dense(32) → Activation('relu') → Dense(1)
#                   └── layers[-2], hier verwendet
# ```
#
# In Abschnitt 17 wurde die *lineare* Ausgabe betrachtet, hier die **Ausgabe nach
# ReLU**. Der Unterschied ist keine Kleinigkeit: ReLU setzt alle negativen Werte auf
# null. Ein Neuron, dessen Vor-Aktivierung für *alle* Eingaben negativ ist, gibt
# damit konstant $0$ aus — ein **totes Neuron** (*dead ReLU*).
#
# ### Was daraus für die Korrelation folgt
#
# Die Pearson-Korrelation ist für ein konstantes Merkmal **nicht definiert**, weil
# im Nenner die Standardabweichung steht:
#
# $$r_{ij} = \frac{\operatorname{Cov}(z_i, z_j)}{\sigma_{z_i}\,\sigma_{z_j}},
#   \qquad \sigma_{z_i} = 0 \;\Rightarrow\; \frac{0}{0} = \text{NaN}$$
#
# Genau das erzeugt die Warnung, die diese Zelle ausgibt:
# `RuntimeWarning: invalid value encountered in divide`. Diese Warnung ist also
# **kein Rauschen, sondern ein Befund** — sie ist der Beweis, dass es tote Neuronen
# gibt.
#
# ### Wie man die Grafik liest
#
# Die Darstellung ist identisch zu Abschnitt 17 ($32\times32$, `clim=(0,1)`, gelb
# $= r \approx 1$, dunkel $= r \le 0$), mit einem neuen Element:
#
# * **Weiße Zeilen und Spalten** = `NaN`. Matplotlib zeichnet `NaN`-Werte in der
#   Hintergrundfarbe. Jede weiße Zeile/Spalte ist ein totes Neuron. Im Bild sind
#   mehrere davon zu erkennen (unter anderem in der Gegend der Indizes 5, 9, 19 und
#   25) — das sind gut 10 bis 15 % der Bottleneck-Kapazität, die schlicht nichts tut.
#
# ### Interpretation
#
# * **Befund 1: mehrere tote Neuronen.** Dass nach zwei Epochen einige ReLU-Einheiten
#   dauerhaft aus sind, ist normal. Problematisch wird es, wenn sie es bleiben: ein
#   totes ReLU-Neuron hat überall Gradient null und kann sich nicht mehr erholen. Die
#   Gegenmittel sind bekannt: kleinere Lernrate, `LeakyReLU`/`ELU` statt `ReLU`,
#   bessere Initialisierung oder BatchNorm direkt vor der Aktivierung.
# * **Befund 2: die Redundanz ist geringer als bei Modell 1.** Das Bild wirkt
#   insgesamt **dunkler und feinkörniger**. Wo bei Modell 1 ein großer
#   durchgehend gelber Block bei den Indizes 5–10 lag, findet man hier viele mittlere
#   Werte (grün, blau) und deutlich sichtbare Diagonalen innerhalb kleinerer
#   Blöcke. Es gibt Gruppen korrelierter Neuronen (etwa um 11–17 und 20–23), aber
#   keine einzige dominierende Richtung mehr. Die Repräsentation von Modell 2 ist
#   also **reichhaltiger** — konsistent damit, dass der Kopf 8192 statt 64
#   Eingangswerte sieht.
# * **Achtung, zwei Effekte überlagern sich:** Ein Teil der dunklen Felder kommt vom
#   ReLU selbst. Wenn zwei Neuronen für viele Volumen *beide* null sind, sinkt die
#   Kovarianz systematisch. Der Vergleich mit Abschnitt 17 ist also nicht ganz
#   fair — dort wurde vor der Aktivierung gemessen, hier danach. Für einen exakten
#   Architekturvergleich müsste man in beiden Fällen die gleiche Stelle wählen.
# * **Und wie immer gilt** die Einschränkung von `clim=(0,1)`: negative Korrelationen
#   sind von Unkorreliertheit visuell nicht zu unterscheiden.

# %%
encoder = Model(model.input, model.layers[-2].output)

train_encodings = encoder.predict(train_X)

correlations = [[np.corrcoef(train_encodings[:,i], train_encodings[:,j])[0,1] \
                 for i in range(32)] for j in range(32)]

plt.figure(figsize=(10, 10))
heatmap = plt.imshow(correlations, clim=(0, 1))
plt.colorbar(heatmap)
plt.show()

# %% [markdown]
# <a id="sec23"></a>
# ## 23. Reproduzierbarkeitscheck: dieselbe Erklärung erneut
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Diese Zelle baut die 7-schichtige Strategie erneut auf (identisch zu Abschnitt 20),
# erzeugt einen neuen Erklärer für dieselbe Ausgabeschicht, erklärt dasselbe Volumen
# und plottet dieselbe $8\times8$-Abbildung. Inhaltlich ist es eine **Wiederholung**
# von Abschnitt 20 — die erzeugte Grafik ist byteweise identisch mit der dortigen.
#
# ### Warum ist das trotzdem nützlich?
#
# Weil die Gleichheit selbst eine Information ist, und zwar in drei Punkten:
#
# 1. **LRP ist deterministisch.** Anders als etwa Occlusion- oder
#    Sampling-Verfahren (LIME, SHAP mit Stichproben) enthält LRP keine
#    Zufallskomponente. Zweimal derselbe Aufruf ergibt exakt dasselbe Ergebnis. Bei
#    stochastischen Erklärungsverfahren wäre hier ein Unterschied zu sehen — und
#    genau das wäre ein wichtiger Warnhinweis, denn eine Erklärung, die bei jedem
#    Aufruf anders aussieht, ist schwer zu verantworten.
# 2. **Das Erzeugen eines Erklärers hat keine Nebenwirkungen** auf das erklärte
#    Modell. `LayerwiseRelevancePropagator` baut intern ein *neues* Modell (inklusive
#    `remove_activation` und `fuse_batchnorm`), lässt das Original aber unangetastet.
#    Sonst würde die zweite Erklärung von der ersten abweichen. Das ist bei
#    Bibliotheken, die Modelle umbauen, keine Selbstverständlichkeit und der Test
#    daher berechtigt.
# 3. **Dropout ist im Inferenzmodus tatsächlich aus.** Wäre es aktiv, würden die
#    beiden Durchläufe unterschiedliche Karten liefern.
#
# ### Interpretation
#
# Die Abbildung ist dieselbe wie in Abschnitt 20 und damit auch dieselbe Aussage:
# überwiegend positive Relevanz, konzentriert an Kugelkontur und Tunnelrändern,
# weiterhin Randartefakte in den leeren Schichten. Wer das Notebook aufräumt, könnte
# diese Zelle löschen; als expliziter Determinismus-Nachweis hat sie aber ihren Wert.

# %%
strategy = LRPStrategy(
    layers=[
        {'flat': True},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'epsilon': 0.25},
        {'epsilon': 0.25}
    ]
)

explainer = LayerwiseRelevancePropagator(model, layer=len(model.layers) - 1, idx=0, strategy=strategy)
explanations = explainer.predict(np.expand_dims(brain, 0))[0]
explanations = explanations / np.amax(np.abs(explanations))

fig, ax = plt.subplots(8, 8, figsize=(15, 8))

for i in range(0, 8, 2):
    for j in range(8):
        idx = ((i // 2) * 8)+ j
        
        ax[i][j].imshow(brain[idx], cmap='Greys_r')
        ax[i][j].axis('off')
        ax[i+1][j].imshow(explanations[idx], cmap='seismic', clim=(-1, 1))
        ax[i+1][j].axis('off')

plt.show()

# %% [markdown]
# ### 23b. RestructuredLRP für Modell 2 (deaktiviert)
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Wie in Abschnitt 15, nur für Modell 2: der Code ist als String deaktiviert und
# wird nicht ausgeführt. Geplant war die kontrastive Erklärung relativ zum
# Prototyp `mean_encoding` der Gruppe mit Label 5, dargestellt als vierzeilige
# Abbildung (Volumen / Standard-LRP / RestructuredLRP / Differenz).
#
# Bei Modell 2 wäre der Vergleich besonders interessant, weil dieses Modell durch
# den auf 5 fixierten Bias schon *implizit* kontrastiv rechnet
# ($f(X) = 5 + w^\top h(X)$). Die Frage wäre also: Bringt eine explizit
# kontrastive Erklärung noch zusätzliche Information, wenn das Modell bereits
# gegen den Mittelwert arbeitet? Die vierte Zeile der Abbildung (die Differenz)
# hätte genau darauf geantwortet — wäre sie überall nahe weiß, wären beide
# Ansätze äquivalent.

# %%
"""
from explainability import RestructuredLRP
    
restructured_lrp = RestructuredLRP(
    model, layer=output_layer, idx=0, bottleneck=bottleneck_layer, strategy=strategy
)
restructured_explanations = restructured_lrp.predict([np.expand_dims(brain, 0), 
                                                      np.expand_dims(mean_encoding, 0)])[0]
restructured_explanations = restructured_explanations / np.amax(np.abs(restructured_explanations))

fig, ax = plt.subplots(4, 8, figsize=(15, 4))


for i in range(8):
    ax[0][i].imshow(brain[12+i], cmap='Greys_r')
    ax[0][i].axis('off')
    ax[1][i].imshow(explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[1][i].axis('off')
    ax[2][i].imshow(restructured_explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[2][i].axis('off')
    ax[3][i].imshow(restructured_explanations[12+i] - explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[3][i].axis('off')

plt.show()
"""

# %% [markdown]
# <a id="sec24"></a>
# ## 24. Referenz-Encodings für Modell 2
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Exakt dieselbe Rechnung wie in Abschnitt 14, jetzt aber mit den Gewichten von
# Modell 2: Encoder bis zur `Dense(32)`-Schicht bauen, alle Testvolumen kodieren,
# die Gruppe mit Label $y = 5$ auswählen und deren Mittelwert und Streuung im
# Merkmalsraum bestimmen.
#
# $$\bar{z} = \frac{1}{|G|}\sum_{n \in G} z^{(n)},
#   \qquad
#   \sigma_k = \operatorname{std}_{n \in G}\big(z^{(n)}_k\big)$$
#
# Diese Wiederholung ist nötig, weil `model` in Abschnitt 18 überschrieben wurde:
# `mean_encoding` und `encoding_stddev` aus Abschnitt 14 gehören noch zu Modell 1
# und wären für Modell 2 wertlos. Ein gutes Beispiel dafür, wie schnell in
# Notebooks Variablen und Modelle auseinanderlaufen — und warum man solche
# abgeleiteten Größen immer direkt nach dem zugehörigen Modell neu berechnen
# sollte.
#
# Hier wird `bottleneck_layer` verwendet, also die **lineare** `Dense(32)`-Ausgabe
# (und nicht wie in Abschnitt 22 die ReLU-Ausgabe). Für RestructuredLRP ist das
# Voraussetzung: die Bibliothek verlangt eine lineare Bottleneck-Schicht, damit die
# Differenz $z - \bar{z}$ mathematisch sauber weiterpropagiert werden kann.
#
# ### Ausgabe
#
# Nur die Fortschrittsbalken von `encoder.predict`. Die Zelle bereitet die Eingaben
# für die letzte (deaktivierte) Zelle vor.

# %%
encoder = Model(model.input, model.layers[bottleneck_layer].output)
encodings = encoder.predict(test_X)
group_idx = np.where(test_y == 5)[0]
group_encodings = encodings[group_idx]
mean_encoding = np.mean(group_encodings, axis=0)
encoding_stddev = np.std(group_encodings, axis=0)

# %% [markdown]
# ### 24b. RestructuredLRP mit Schwellenwert für Modell 2 (deaktiviert)
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Die letzte inhaltliche Zelle, ebenfalls deaktiviert. Sie ist die
# Schwellenwert-Variante für Modell 2. Ein Detail lohnt die Aufmerksamkeit:
# übergeben wird `encoding_stddev * 8`, also eine **achtfache** Streuung als
# Schwelle. In Abschnitt 15b war es die einfache Streuung.
#
# Der Faktor ist ein Regler für die Sparsamkeit der Erklärung:
#
# $$\delta_k \ne 0 \iff |z_k - \bar{z}_k| > \tau\,\sigma_k$$
#
# * $\tau$ klein → viele Merkmale überschreiten die Schwelle → dichte Heatmap.
# * $\tau$ groß → nur extreme Abweichungen bleiben → sehr sparsame Heatmap.
#
# $\tau = 8$ ist ein sehr strenger Wert; unter einer Normalverteilungsannahme wäre
# eine Abweichung von acht Standardabweichungen praktisch unmöglich. Dass er hier
# nötig scheint, deutet darauf hin, dass die Merkmalsstreuung innerhalb der
# Referenzgruppe klein ist im Verhältnis zu den Unterschieden zwischen Gruppen —
# oder einfach, dass der Wert experimentell nachjustiert wurde. In der Praxis ist
# genau das der Punkt, an dem eine Erklärungsmethode selbst Hyperparameter bekommt,
# und man sollte ehrlich benennen, dass die "Sparsamkeit" einer Heatmap damit zu
# einer **Entscheidung des Anwenders** wird und nicht mehr allein eine Eigenschaft
# des Modells ist.

# %%
"""
from explainability import RestructuredLRP
    
restructured_lrp = RestructuredLRP(
    model, layer=output_layer, idx=0, bottleneck=bottleneck_layer,
    strategy=strategy, threshold=True
)
restructured_explanations = restructured_lrp.predict([np.expand_dims(brain, 0), 
                                                      np.expand_dims(mean_encoding, 0),
                                                      np.expand_dims(encoding_stddev * 8, 0)])[0]
restructured_explanations = restructured_explanations / np.amax(np.abs(restructured_explanations))

fig, ax = plt.subplots(4, 8, figsize=(15, 4))


for i in range(8):
    ax[0][i].imshow(brain[12+i], cmap='Greys_r')
    ax[0][i].axis('off')
    ax[1][i].imshow(explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[1][i].axis('off')
    ax[2][i].imshow(restructured_explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[2][i].axis('off')
    ax[3][i].imshow(restructured_explanations[12+i] - explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[3][i].axis('off')

plt.show()
"""

# %% [markdown]
# <a id="sec25"></a>
# ## 25. Fazit und Ausblick
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was dieses Notebook demonstriert
#
# Der eigentliche Wert liegt nicht in einem guten Modell, sondern in einem
# **Prüfverfahren für Erklärungen**. Fünf Bausteine, die man auf jedes eigene
# Projekt übertragen kann:
#
# | # | Baustein | Abschnitt | Leitfrage |
# |---|----------|-----------|-----------|
# | 1 | Synthetische Daten mit bekannter Ground Truth | 1 | Kenne ich die richtige Erklärung? |
# | 2 | Modellgüte *vor* der Erklärung prüfen | 8, 9 | Ist das Modell überhaupt erklärungswürdig? |
# | 3 | Heatmaps mit Vorzeichen lesen | 10, 13 | Wo ist Evidenz *für* und *gegen* die Vorhersage? |
# | 4 | Kausale Eingriffe | 11, 12 | Reagiert das Modell auf das Richtige und *nicht* auf das Falsche? |
# | 5 | Repräsentation analysieren | 14, 16, 17, 21, 22 | Ist die interne Kodierung überhaupt interpretierbar? |
#
# ### Die wichtigsten Einsichten im Rückblick
#
# **Zur Erklärbarkeit:**
#
# * Eine Heatmap zeigt immer *Modell × Erklärungsregel*, nie das Modell allein.
#   Der deutlichste Beleg: In Abschnitt 13 tragen komplett leere Bildschichten
#   Relevanz — ein Effekt der `flat`-Regel, nicht der Daten.
# * Architektur bestimmt Erklärbarkeit mit. Global Average Pooling erzwingt
#   Translationsinvarianz (gut für die Aufgabe), zerstört aber Ortsinformation
#   (schlecht für Heatmaps). Modell 2 zeigt in Abschnitt 20 sichtbar besser
#   lokalisierte Relevanz an Kugelkontur und Tunnelwänden.
# * Vorzeichen sind Information: Bei Modell 1 war Gewebe blau (senkt die
#   Vorhersage) und Leerraum rot (erhöht sie) — das ist die inhaltlich korrekte
#   Logik. Bei Modell 2 verschiebt der fixierte Bias die Referenz und damit das
#   Gesamtvorzeichen, ohne dass sich der Mechanismus geändert hätte.
# * Kausale Tests sind stärker als Heatmaps. Abschnitt 12 hat gezeigt, dass breite
#   Bohrungen etwa zehnmal so große Vorhersagesprünge erzeugen wie schmale — der
#   überzeugendste Beleg dieses Notebooks dafür, dass das Modell die *richtige*
#   Größe misst.
# * Erklärungen auf Neuronenebene brauchen eine entwirrte Repräsentation. Solange
#   32 Bottleneck-Neuronen fast dieselbe Karte liefern (Abschnitte 16 und 21), ist
#   die Aussage "Neuron $k$ kodiert $X$" nicht identifizierbar.
#
# **Zum Deep Learning allgemein:**
#
# * Loss-Werte brauchen eine Baseline. $\operatorname{Var}(y) = 8.25$ ist hier die
#   Messlatte; Modell 1 lag nach zwei Epochen mit 17.5 darüber, Modell 2 mit 3.0
#   darunter.
# * Eine gute Initialisierung ersetzt viel Training. Der auf 5 fixierte
#   Ausgabe-Bias von Modell 2 senkt den Startverlust um Faktor 5.
# * Warnungen sind Befunde. Das `RuntimeWarning` in Abschnitt 22 hat tote
#   ReLU-Neuronen aufgedeckt.
# * Notebook-Zustand ist gefährlich. `model` wird überschrieben, `y` kann durch eine
#   Plot-Zelle zerstört werden — die `assert`-Anweisungen im Notebook sind kein
#   Zierrat.
#
# ### Wie man von hier weiterarbeitet
#
# In der Reihenfolge des Nutzens:
#
# 1. **`epochs=500` setzen** (in den Abschnitten 8 und 19). Das ist der einzige
#    Schritt, der aus einer Methodendemonstration eine inhaltliche Aussage macht.
#    Alle Interpretationen oben stehen unter dem Vorbehalt zweier Trainingsepochen.
# 2. **Das Streudiagramm erneut prüfen** (Abschnitt 9). Erst wenn die Punkte
#    ungefähr auf der Diagonale liegen, sind die Heatmaps inhaltlich belastbar.
# 3. **Die kausalen Tests wiederholen** (Abschnitte 11 und 12). Wird die Kurve über
#    die Tunnelanzahl endlich flach (Invarianz) und die Treppe über die Breiten
#    endlich sauber?
# 4. **`NUM_TUNNELS` in den Trainingsdaten variieren.** Solange immer genau 6 Tunnel
#    gebohrt werden, *kann* das Modell nicht lernen, die Anzahl zu ignorieren.
# 5. **Die `flat`-Regel durch die $z^\mathcal{B}$-Regel ersetzen**, um die
#    Relevanz-Leckage in leere Bildbereiche zu beseitigen.
# 6. **Redundanz quantifizieren** statt nur zu betrachten: PCA oder effektive
#    Dimension des Bottlenecks, und die Korrelationsmatrix mit $|r|$ bzw.
#    `clim=(-1,1)` plotten, damit Antikorrelationen sichtbar werden.
# 7. **`RestructuredLRP` wieder zum Laufen bringen** — die vier deaktivierten Zellen
#    enthalten den methodisch interessantesten Teil des Repos.
#
# ### Der Bogen zurück zum Anfang
#
# Ziel war nie die Tunnelbreite. Ziel war die Frage, ob man einer Heatmap glauben
# darf. Auf diesem synthetischen Datensatz kann man das prüfen, weil wir die
# Antwort kennen. Erst wenn eine Erklärungsmethode hier alle fünf Prüfungen
# besteht, hat man ein Recht darauf, sie auf echte MRT-Daten anzuwenden — wo
# niemand die richtige Antwort kennt und eine plausibel aussehende, aber falsche
# Heatmap echten Schaden anrichten kann.

# %%

# %%
