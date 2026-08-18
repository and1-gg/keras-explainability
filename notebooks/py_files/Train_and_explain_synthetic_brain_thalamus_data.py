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
# # Findet LRP den Thalamus?
#
# ### Synthetische Gehirndaten mit bekannter Ground Truth als Prüfstand für XAI
#
# ---
#
# ## Die Frage, um die es geht
#
# Wir haben ein neuronales Netz, das aus einem 3D-Bild des Gehirns eine Zahl vorhersagt, und
# ein Erklärungsverfahren (**LRP**), das uns sagt, welche Bildpunkte dafür wichtig waren. Das
# Verfahren liefert immer ein buntes Bild. Die entscheidende Frage lautet:
#
# > **Sind das die *richtigen* Bildpunkte?**
#
# Bei echten Hirndaten kann man diese Frage nicht beantworten, weil niemand die richtige Antwort
# kennt. Dieses Notebook baut deshalb Daten, bei denen die richtige Antwort **per Konstruktion**
# feststeht: Der vorhergesagte Zahlenwert ist das **Volumen des Thalamus** — und nichts sonst im
# Bild trägt Information darüber. Eine korrekte Erklärung *muss* also auf den Thalamus zeigen.
#
# Damit wird nicht das Gehirn untersucht, sondern **das Erklärungsverfahren selbst**. In der
# Literatur heißt dieses Vorgehen *sanity check* bzw. *ground-truth-based evaluation of
# attribution methods*.
#
# ## Für wen ist dieses Notebook geschrieben?
#
# Für Leser ohne Vorkenntnisse in Deep Learning, in XAI/LRP, in Neuroanatomie und in FreeSurfer.
# Alle vier Themen werden an der Stelle erklärt, an der sie zum ersten Mal gebraucht werden.
# Wer schon weiß, was ein 3D-CNN ist, kann die entsprechenden Kästen überspringen.
#
# ## Warum gerade der Thalamus?
#
# Der **Thalamus** ist ein etwa walnussgroßer, paariger Kern tief in der Mitte des Gehirns —
# einer links, einer rechts, direkt oberhalb des Hirnstamms. Er ist die zentrale Umschaltstelle
# für fast alle Sinnesinformationen auf ihrem Weg zur Hirnrinde. Für dieses Notebook ist er aus
# vier praktischen Gründen ein ideales Ziel:
#
# | Eigenschaft | Warum das hier hilft |
# |---|---|
# | **kompakt und klar abgegrenzt** | Die Maske ist eindeutig — ein Voxel ist drinnen oder draußen |
# | **klein** (ca. 1 % des Hirnvolumens) | Zufälliges Treffen ist unwahrscheinlich, der Test ist scharf |
# | **tief innen gelegen** | Ein Verfahren, das nur Bildränder markiert, fällt sofort auf |
# | **klinisch relevant** | Thalamusatrophie ist ein etablierter Marker, u. a. bei Multipler Sklerose und Demenzen |
#
# ## Der Ablauf in einem Bild
#
# ```text
#              ┌──────────────────────────────────────────────┐
#              │  Zwei Datensätze mit bekannter Ground Truth  │
#              └──────────────────────────────────────────────┘
#                              │
#      ┌───────────────────────┴────────────────────────┐
#      ▼                                                ▼
#  A) echte MRT-Bilder,                       B) vollsynthetische Phantome
#     alles ausser Thalamus                      4 Kompartimente, Zielwert
#     wird permutiert (Rauschen)                  = Thalamusvolumen
#      │                                                │
#      ▼                                                ▼
#  vortrainiertes Brain-Age-SFCN               eigenes 3D-CNN, selbst trainiert
#  (Gewichte aus output/pyment/models)         (Zielwert nachweislich = Thalamus)
#      │                                                │
#      └───────────────────┬────────────────────────────┘
#                          ▼
#                 LRP-Relevanzkarte je Voxel
#                          │
#                          ▼
#         Vergleich mit der Thalamusmaske (Ground Truth)
#         → Relevanzmasse, Dichte-Verhältnis, Top-k-Präzision,
#           Pointing Game, Distanzprofil, Regelvergleich
# ```
#
# ## Die Kurzfassung des Ergebnisses
#
# Damit man weiß, worauf es hinausläuft (Details in den Abschnitten 12–15):
#
# 1. **Das mitgelieferte vortrainierte Brain-Age-Modell ist für diesen Test unbrauchbar.** Es gibt
#    für ein echtes Gehirn, für reines Rauschen und für ein komplett leeres Bild praktisch
#    dieselbe Zahl aus. Abschnitt 7 zeigt das mit drei Zeilen Code. Wer diesen Test überspringt,
#    interpretiert hinterher Heatmaps eines Modells, das die Bilder nie angeschaut hat.
# 2. **Der „Shuffle-Test" kann trügen.** Auf permutierten Bildern landet die Relevanz dieses
#    *konstanten* Modells trotzdem bevorzugt im Thalamus (Faktor ≈ 4 über Zufall, Pointing Game
#    bestanden). Der Grund ist rein bildstatistisch. Ein bestandener Shuffle-Test ist also **kein**
#    Beweis dafür, dass ein Modell die Struktur benutzt.
# 3. **Beim selbst trainierten Modell funktioniert LRP — aber nur teilweise.** Es konzentriert
#    ≈ 9 % der Relevanz auf ein Gebiet, das 1,8 % des Volumens ausmacht (Anreicherung um Faktor
#    ≈ 5,2 gegenüber dem Hirndurchschnitt). Die Lokalisation ist also klar überzufällig, aber weit
#    von den 100 % entfernt, die die Ground Truth hergäbe: 57 % der Relevanz liegt in weißer
#    Substanz, die nichts zur Antwort beiträgt.
# 4. **Die stärkste Anreicherung liegt nicht *im* Thalamus, sondern direkt an seinem Rand.** In der
#    1-mm-Schale außerhalb der Struktur ist die Relevanzdichte 8,4-fach erhöht, innen „nur"
#    5,0-fach; jenseits von 4 mm fällt sie unter den Durchschnitt. Das ist inhaltlich sinnvoll —
#    die Größe einer Struktur liest man an ihrer Grenze ab, nicht in ihrer Mitte — hat aber eine
#    unangenehme Konsequenz für die Bewertung: Wer scharf gegen die Maske prüft, bestraft das
#    Modell für vernünftiges Verhalten.
# 5. **Die Wahl der LRP-Regel entscheidet über Erfolg und Misserfolg** — dieselbe Erklärung
#    desselben Modells erreicht je nach Regel Faktor 5,0 oder Faktor 0,7 (= gar keine
#    Lokalisation), und die Rangfolge der Regeln **wechselt sogar je nach Metrik**. Wer nur eine
#    Regel und nur eine Kennzahl berichtet, weiß nicht, was er misst.
#
# ---
#
# <a id="toc"></a>
# ## Inhaltsverzeichnis
#
# | # | Abschnitt | Thema |
# |---|-----------|-------|
# | 1 | [Setup, Projektpfade und Rechenwerk](#sec-01) | Technik |
# | 2 | [Neuro-Grundlagen: Thalamus, FreeSurfer, Atlas](#sec-02) | Hintergrund |
# | 3 | [IXI-Daten laden und Thalamusmasken](#sec-03) | Daten |
# | 4 | [Kontrolldatensatz A: alles außer dem Thalamus permutieren](#sec-04) | Daten |
# | 5 | [Kontrolldatensatz B: vollsynthetische Phantome](#sec-05) | Daten |
# | 6 | [Woran misst man Erfolg? Die Ground-Truth-Metriken](#sec-06) | Methode |
# | 7 | [Das vortrainierte Modell laden — und prüfen, ob es reagiert](#sec-07) | Modell |
# | 8 | [LRP auf dem vortrainierten Modell: der trügerische Erfolg](#sec-08) | XAI |
# | 9 | [Ein eigenes Modell trainieren, das nur den Thalamus messen kann](#sec-09) | Modell |
# | 10 | [Wie gut sagt das Phantom-Modell vorher?](#sec-10) | Modell |
# | 11 | [LRP auf dem Phantom-Modell: die Heatmaps](#sec-11) | XAI |
# | 12 | [Quantitative Auswertung: Relevanz je Kompartiment](#sec-12) | XAI |
# | 13 | [Wie scharf ist die Lokalisation? Dilatation und Distanzprofil](#sec-13) | XAI |
# | 14 | [Der Regelvergleich: die Regel entscheidet](#sec-14) | XAI |
# | 15 | [Sanity-Check: randomisierte Gewichte](#sec-15) | XAI |
# | 16 | [Übertragung auf echte Daten: das FreeSurfer/FSL-Rezept](#sec-16) | Praxis |
# | 17 | [Fazit, Fallstricke und nächste Schritte](#sec-17) | Zusammenfassung |
#
# ---
#
# ### Vokabelliste für den Einstieg
#
# | Begriff | Bedeutung in diesem Notebook |
# |---------|------------------------------|
# | **Voxel** | ein Bildpunkt in 3D (das 3D-Gegenstück zum Pixel); hier 1 mm × 1 mm × 1 mm |
# | **Volumen** | ein Datenpunkt: ein 3D-Würfel mit einem Grauwert pro Voxel |
# | **NIfTI** (`.nii.gz`) | das Standard-Dateiformat für 3D-Hirnbilder; enthält Bild **und** Lageinformation |
# | **Maske** | 3D-Bild aus Nullen und Einsen: 1 = Voxel gehört zur Struktur |
# | **Segmentierung / Atlas** | 3D-Bild, das jedem Voxel eine *Regionsnummer* gibt (z. B. 10 = linker Thalamus) |
# | **MNI152** | ein Standard-Koordinatensystem; nach Registrierung liegt Voxel $(i,j,k)$ in jedem Bild am (annähernd) gleichen anatomischen Ort |
# | **Registrierung** | ein Bild so verschieben/drehen/skalieren, dass es zu einem Referenzbild passt |
# | **Skull-stripped** | Schädel, Haut und Augen sind entfernt; außerhalb des Gehirns steht exakt 0 |
# | **CNN / Conv3D** | Faltungsnetz; ein Filter (hier 3×3×3) fährt über das Volumen und sucht lokale Muster |
# | **Regression** | das Netz gibt eine *Zahl* aus (hier: ein Volumen), keine Klasse |
# | **GAP** | Global Average Pooling: mittelt eine ganze Merkmalskarte zu einer Zahl |
# | **Relevanz / Heatmap** | pro Voxel eine Zahl: wie stark hat dieser Voxel die Vorhersage beeinflusst |
# | **LRP** | Layer-wise Relevance Propagation, das Erklärungsverfahren dieses Repos |
# | **Ground Truth** | die bekannte richtige Antwort, gegen die man prüft |
#
# **Verwandte Notebooks in diesem Repo:** `Explain_brain_age_predictions` ist die **Referenz für
# die echte IXI-Datenstruktur** (NIfTI-Dataset, Preprocessor, Generator, Ordnerlayout) — an ihr
# orientiert sich Abschnitt 3. `Train_and_explain_synthetic_brain_regression_model` und
# `Train_and_explain_dummy_geometric_data` sind reine Synthese-Notebooks und **kein** Vorbild
# für den IXI-Pfad; die Phantome in Abschnitt 5 dienen nur der kontrollierten LRP-Validierung.
# **Hintergrunddokumente:** [`doc/LRP_Fraunhofer_HHI_Video_Zusammenfassung.md`](../../doc/LRP_Fraunhofer_HHI_Video_Zusammenfassung.md)
# und [`doc/IXI_Datenlayout.md`](../../doc/IXI_Datenlayout.md).

# %% [markdown]
# <a id="sec-01"></a>
# ## 1. Setup, Projektpfade und Rechenwerk
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Vier Dinge, alle organisatorisch:
#
# 1. **Repository-Wurzel finden** und in `sys.path` eintragen. Erst dadurch sind die beiden
#    lokalen Pakete `explainability` (der LRP-Code) und `pyment` (Modelle und Daten-Loader)
#    importierbar, egal aus welchem Ordner der Kernel gestartet wurde.
# 2. **Notebook-Namen ermitteln.** `find_notebook_name()` probiert vier Wege durch (interaktiv,
#    Umgebungsvariable, als Skript, unter `nbconvert`), damit der Ausgabeordner immer gleich
#    heißt — auch wenn das Notebook per `make html-from-single-notebook` gerendert wird.
# 3. **Ausgabeordner anlegen:** `output/notebooks/<notebook-name>/`. Dort landen alle Abbildungen,
#    die Masken-NIfTIs, die Metrik-Tabellen und das trainierte Modell.
# 4. **Rechenwerk festlegen** — der einzige Punkt hier, der inhaltlich wichtig ist.
#
# ### Warum GPU *und* CPU?
#
# LRP baut aus dem Netz ein **zweites** Modell, das den Rückwärtsweg explizit als Schichten
# enthält. Dabei liegen die Aktivierungen des Vorwärtspasses **und** die Relevanzen gleichzeitig
# im Speicher. Für das große Modell auf dem Gitter $167 \times 212 \times 160$ ist die erste
# Faltungsschicht allein
#
# $$167 \cdot 212 \cdot 160 \cdot 32 \text{ Kanäle} \cdot 4 \text{ Byte} \approx 725 \text{ MB},$$
#
# und davon braucht der Rückwärtsweg mehrere Kopien. Auf einer 8-GB-Grafikkarte scheitert das
# (die Fehlermeldung lautet dann kryptisch `Autotuning failed ... No valid config found!`). Auf
# der CPU mit viel Arbeitsspeicher läuft dieselbe Rechnung dagegen in wenigen Sekunden durch.
#
# Deshalb die Aufteilung:
#
# | Teil | Gerät | Grund |
# |---|---|---|
# | Großes SFCN (Abschnitte 7–8) | **CPU** (`tf.device('/CPU:0')`) | zu speicherhungrig für 8 GB VRAM |
# | Kleines Phantom-Modell (Abschnitte 9–15) | **GPU**, falls vorhanden | 64³ passt bequem, Training ist ~50× schneller |
#
# Außerdem wird `mixed_precision` explizit auf `float32` gesetzt: der LRP-Code dieses Repos
# verträgt `mixed_float16` nicht (Dtype-Konflikte beim Rückwärtsweg).
#
# ### Einordnung
#
# Solche Setup-Zellen wirken wie Kleinkram, sind aber der häufigste Grund, warum ein Notebook auf
# einer anderen Maschine nicht läuft. Diese Datei ist über **jupytext** als `py:percent`-Skript
# mit dem `.ipynb` gekoppelt (siehe Kopf der Datei) — dadurch bleibt der Code in Git diff-bar,
# während man trotzdem interaktiv arbeiten kann. Zellgrenzen sind die `# %%`-Marker.

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
import tensorflow as tf


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
    try:
        return ipynbname.name()
    except Exception:
        pass

    for candidate in (os.environ.get("JPY_SESSION_NAME"), globals().get("__session__")):
        if candidate and candidate.endswith(".ipynb"):
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

    raise RuntimeError(
        "Notebook-Name konnte nicht ermittelt werden — bitte NOTEBOOK_NAME setzen."
    )


notebook_name = os.environ.get("NOTEBOOK_NAME") or find_notebook_name()

target_dir = repo_root / "output" / "notebooks" / notebook_name
target_dir.mkdir(parents=True, exist_ok=True)

# --- Rechenwerk -------------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    # ohne memory_growth belegt TF sofort den gesamten VRAM
    tf.config.experimental.set_memory_growth(gpu, True)

# LRP dieses Repos rechnet ausschliesslich in float32
tf.keras.mixed_precision.set_global_policy("float32")

# Das grosse SFCN sprengt 8 GB VRAM, laeuft auf der CPU aber in Sekunden
BIG_MODEL_DEVICE = "/CPU:0"
SMALL_MODEL_DEVICE = "/GPU:0" if gpus else "/CPU:0"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print(f"Notebook-Name       : {notebook_name}")
print(f"Zielordner          : {target_dir}")
print(f"TensorFlow          : {tf.__version__}")
print(f"GPUs erkannt        : {len(gpus)} — {[g.name for g in gpus]}")
print(f"Grosses Modell auf  : {BIG_MODEL_DEVICE}")
print(f"Kleines Modell auf  : {SMALL_MODEL_DEVICE}")

# %% [markdown]
# <a id="sec-02"></a>
# ## 2. Neuro-Grundlagen: Thalamus, FreeSurfer, Atlas
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Dieser Abschnitt enthält keinen Code. Er erklärt die vier Begriffe, ohne die die folgenden
# Zellen nicht verständlich sind. Wer sich damit auskennt, springt direkt zu
# [Abschnitt 3](#sec-03).
#
# ---
#
# ### 2.1 Wie ein Hirnbild auf der Festplatte liegt
#
# Ein MRT des Kopfes ist ein **3D-Array von Grauwerten**. Ein typisches Bild in diesem Projekt
# hat die Form `(167, 212, 160)` — also 167 × 212 × 160 = **5,66 Millionen Voxel**, jeder 1 mm³
# groß. Gespeichert wird das als **NIfTI** (`.nii.gz`), und dieses Format enthält zwei Dinge:
#
# | Bestandteil | Inhalt |
# |---|---|
# | **Daten** | das 3D-Array der Grauwerte |
# | **Affine** (4×4-Matrix) | die Umrechnung von Voxelindex in Millimeter-Weltkoordinaten |
#
# Die Affine ist der Grund, warum man Bilder unterschiedlicher Auflösung überhaupt vergleichen
# kann: sie sagt, *wo im Raum* Voxel $(i,j,k)$ liegt:
#
# $$\begin{pmatrix} x \\ y \\ z \\ 1 \end{pmatrix}
#   = A \cdot \begin{pmatrix} i \\ j \\ k \\ 1 \end{pmatrix}$$
#
# ### 2.2 Warum alle Bilder in denselben Raum müssen (MNI152)
#
# Jeder Kopf ist anders geformt und liegt anders im Scanner. Voxel $(43, 101, 66)$ bedeutet
# deshalb in zwei Rohbildern völlig verschiedene Orte. **Registrierung** löst das: man verschiebt,
# dreht (und skaliert optional) jedes Bild so, dass es zu einer Referenzvorlage passt — hier das
# **MNI152**-Template, der De-facto-Standardraum der Neuroimaging-Welt.
#
# Das Werkzeug dafür heißt `flirt` (aus FSL). Der Parameter `-dof 6` bedeutet *6 Freiheitsgrade*:
# drei Verschiebungen und drei Drehungen, **keine** Skalierung. Damit bleibt die tatsächliche
# Größe der Strukturen erhalten — entscheidend, wenn man nachher **Volumina** messen will.
# Mit `-dof 12` (affin, inkl. Skalierung) würde man genau die Information wegnormieren, die uns
# in diesem Notebook interessiert.
#
# Nach der Registrierung gilt: **Voxel $(i,j,k)$ liegt in jedem Bild am (annähernd) gleichen
# anatomischen Ort.** Erst das erlaubt es, Heatmaps über Personen zu mitteln.
#
# ### 2.3 Was FreeSurfer / FastSurfer macht
#
# **FreeSurfer** ist das seit Jahrzehnten etablierte Softwarepaket zur automatischen Auswertung
# von Hirn-MRTs; **FastSurfer** ist ein Deep-Learning-Nachbau, der in Minuten statt Stunden
# rechnet. Beide liefern u. a. eine **Segmentierung**: ein 3D-Bild in derselben Größe wie das
# MRT, in dem aber nicht Grauwerte, sondern **Regionsnummern** stehen.
#
# Diese Datei heißt `aseg` (*automatic segmentation*) bzw. mit kortikaler Parzellierung
# `aparc.DKTatlas+aseg`. Die Nummern sind global standardisiert:
#
# | Nummer | Struktur |
# |---|---|
# | 0 | Hintergrund |
# | 2 / 41 | weiße Substanz links / rechts |
# | 4 / 43 | Seitenventrikel links / rechts |
# | **10** | **Thalamus links** |
# | 17 / 53 | Hippocampus links / rechts |
# | **49** | **Thalamus rechts** |
#
# Die vollständige Tabelle liegt im Repo unter
# `FastSurferCNN/config/FreeSurferColorLUT.txt` (LUT = *Look-Up Table*).
#
# ### 2.4 Der Kern des Problems: Voxel (43, 101, 66) ist keine Aussage
#
# LRP liefert eine Relevanz **pro Voxel**. Ein Satz wie *„Voxel (43, 101, 66) war wichtig"* ist
# neurowissenschaftlich wertlos — niemand weiß, welche Struktur das ist. Verwertbar wird es erst
# so:
#
# > *„12 % der Relevanz lagen im Thalamus, der 1 % des Hirnvolumens ausmacht."*
#
# Der Übersetzungsschritt dorthin ist **einfacher, als er klingt**, weil beide Bilder auf dem
# gleichen Gitter liegen: man schaut in der Segmentierung an derselben Stelle nach.
#
# ```text
#    Relevanzkarte              Segmentierung (aseg)
#    ┌───┬───┬───┐              ┌───┬───┬───┐
#    │0.1│0.8│0.2│              │ 2 │10 │10 │      Voxel mit aseg == 10
#    ├───┼───┼───┤              ├───┼───┼───┤   →  gehoeren zum linken
#    │0.0│0.9│0.7│              │ 2 │10 │10 │      Thalamus. Deren Relevanz
#    └───┴───┴───┘              └───┴───┴───┘      wird aufsummiert: 0.8+0.2+0.9+0.7
# ```
#
# In NumPy ist das genau eine Zeile:
#
# ```python
# relevanz_thalamus = relevanz[np.isin(aseg, [10, 49])].sum()
# ```
#
# Die einzige echte Hürde ist, dass die Segmentierung meist auf einem **anderen Gitter** liegt
# als das Bild, das ins Netz geht (FastSurfer arbeitet intern mit 256³). Genau das räumt der
# nächste Abschnitt auf.
#
# ### 2.5 Wie zwei Strategien für Testdaten aussehen
#
# Es gibt zwei Wege, sich Daten mit bekannter Ground Truth zu bauen. Dieses Notebook nutzt beide,
# weil sie sich gegenseitig kontrollieren:
#
# | | **A) Permutation echter Bilder** | **B) Vollsynthetische Phantome** |
# |---|---|---|
# | Vorgehen | Alles außer dem Thalamus wird durchmischt | Bild wird von Null an gebaut |
# | Realismus | hoch (echte Anatomie, echte Grauwerte) | niedrig (Ellipsoide) |
# | Wir wissen … | wo Information **nicht** ist | wo Information **ist** |
# | Passendes Modell | jedes vortrainierte Modell | muss selbst trainiert werden |
# | Bildgitter | bleibt exakt im Original-Bildraum | frei wählbar |
# | Abschnitt | [4](#sec-04) und [8](#sec-08) | [5](#sec-05) und [9](#sec-09)–[15](#sec-15) |
#
# Variante A ist der elegantere Test, weil sie im echten Bildraum bleibt und keine
# Trainingsläufe braucht. Sie hat aber eine Lücke: sie liefert nur eine **Negativaussage**
# („hier ist keine Information"), und man weiß nicht, ob das Modell die verbleibende Struktur
# überhaupt benutzt. Abschnitt 8 zeigt, dass genau diese Lücke gefährlich ist.

# %% [markdown]
# <a id="sec-03"></a>
# ## 3. IXI-Daten laden und Thalamusmasken
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Zuerst die **gleiche Datenpipeline** wie in `Explain_brain_age_predictions` — erst danach die
# Thalamusmasken. So bleiben Subjektliste, Normalisierung und Ordnerlayout mit dem Schwester-
# Notebook identisch; wer dort schon gearbeitet hat, findet sich hier sofort zurecht.
#
# #### Schritt A — drei Bausteine aus `pyment` (wie im Brain-Age-Notebook)
#
# | Objekt | Aufgabe |
# |---|---|
# | `NiftiDataset.from_folder(image_folder, target='age')` | verknüpft `cropped/images/*.nii.gz` mit der Spalte `age` aus `cropped/labels.csv` |
# | `NiftiPreprocessor(sigma=255.)` | skaliert die Voxelwerte |
# | `AsyncNiftiGenerator(...)` | liefert Batches à 4 Bilder und lädt die nächsten schon im Hintergrund |
#
# Das Preprocessing besteht aus **genau einer Operation**:
#
# $$X \;\leftarrow\; \frac{X}{\sigma}, \qquad \sigma = 255$$
#
# Die Bilder liegen als 8-Bit-Intensitäten (0–255) vor und landen damit im Bereich $[0, 1]$.
# Es findet **kein** Resampling und keine z-Standardisierung statt — das Cropping auf
# `(167, 212, 160)` ist schon offline geschehen (siehe [`doc/IXI_Datenlayout.md`](../../doc/IXI_Datenlayout.md)).
#
# Erwartetes Ordnerlayout:
#
# ```text
# data/mri/ixi/
# ├── cropped/                         ← Root für NiftiDataset.from_folder
# │   ├── images/<id>.nii.gz           ← 167×212×160, MNI, skull-stripped, Werte 0–255
# │   └── labels.csv                   ← Pflichtspalten: id, age
# └── fastsurfer/<id>/mri/
#     └── aparc.DKTatlas+aseg.deep.mgz ← 256³ Segmentierung (für die Masken)
# ```
#
# Wichtig: `image_folder` zeigt auf **`cropped/`**, nicht auf `cropped/images/`. Die Defaults von
# `from_folder` hängen `images/` und `labels.csv` selbst an — genau wie im Brain-Age-Notebook.
#
# #### Schritt B — Thalamusmaske im Bildraum des Modells
#
# Danach erzeugen wir für jedes Dataset-Subjekt mit vorhandener Segmentierung eine
# **Thalamusmaske** der Form `(167, 212, 160)`. Der Knackpunkt ist ein Gitter-Mismatch:
#
# | Datei | Form | Orientierung |
# |---|---|---|
# | `cropped/images/<id>.nii.gz` (geht ins Netz) | 167 × 212 × 160 | LAS |
# | `fastsurfer/<id>/mri/aparc.DKTatlas+aseg.deep.mgz` | 256 × 256 × 256 | LIA |
#
# „LAS" und „LIA" sind Kürzel für die Achsenrichtungen (**L**eft, **A**nterior, **S**uperior,
# **I**nferior). Die beiden Dateien haben also nicht nur unterschiedliche Größen, sondern auch
# vertauschte und gespiegelte Achsen. Man darf die Arrays deshalb **auf keinen Fall** direkt
# aufeinander legen.
#
# Die Lösung ist eine einzige Funktion:
#
# ```python
# resample_from_to(aseg, image, order=0)
# ```
#
# Sie liest die Affine beider Bilder, rechnet für jedes Ziel-Voxel dessen Weltkoordinate aus,
# schaut nach, welches Quell-Voxel dort liegt, und schreibt dessen Wert hin. `order=0` bedeutet
# **Nearest-Neighbour-Interpolation** — zwingend bei Label-Bildern: Zwischen Thalamus (10) und
# weißer Substanz (2) darf nicht „6" interpoliert werden, denn 6 ist eine ganz andere Struktur.
#
# > **Hinweis zur Richtung:** Das Brain-Age-Notebook mappt Erklärungen per `conform()` *ins*
# > FastSurfer-Gitter (256³), um Relevanz je Atlasregion zu summieren. Hier brauchen wir die
# > Maske *im Modellgitter* `(167, 212, 160)`, weil Shuffle und LRP-Metriken dort leben —
# > deshalb `resample_from_to(aseg → image)`. Gleiches Ordnerlayout, andere Ausrichtungsrichtung.
#
# > **Alternativer Weg (Abschnitt 16):** Wer FSL nutzt, kann dasselbe mit `flirt -applyxfm
# > -interp nearestneighbour` und `fslmaths -thr 10 -uthr 10 -bin` erreichen.
#
# ### Der Plausibilitätscheck, den man nie auslassen darf
#
# Ein Resampling kann still und leise schiefgehen. Drei Prüfungen fangen das ab:
#
# 1. **Volumen:** Ein Thalamus hat pro Seite etwa 6 000–9 000 mm³. Bei 1-mm-Voxeln müssen also
#    ungefähr 6 000–9 000 Voxel pro Seite herauskommen. Zehn Voxel oder 200 000 Voxel wären ein
#    Alarmsignal.
# 2. **Lage:** Der Schwerpunkt muss nahe der Bildmitte liegen (der Thalamus sitzt zentral), und
#    linker und rechter Thalamus müssen auf gegenüberliegenden Seiten liegen.
# 3. **Enthaltensein:** **Jedes** Thalamus-Voxel muss innerhalb des Gehirns liegen, also dort, wo
#    das skull-stripped Bild nicht 0 ist. Ein einziger Treffer außerhalb bedeutet Fehlregistrierung.
#
# ### Ausgabe dieser Zelle
#
# Neben TensorFlow-/Logging-Meldungen erscheint typischerweise:
#
# ```text
# WARNING - Skipping sub-638: Missing labels
# Dataset: 10 Einträge, 9 eindeutige IDs (Alter als Ziel)
# Subjekte mit Bild + Alter + Segmentierung: 9
#      id   age  thal_voxel  hirn_voxel  anteil_%  ausserhalb ...
# sub-395 45.03       17374     1739103      1.00           0 ...
# ...
# ```
#
# Für `sub-638` gibt es zwar ein Bild (und FastSurfer), aber **keine Zeile in `labels.csv`** —
# `NiftiDataset` überspringt es still, genau wie im Brain-Age-Notebook. Umgekehrt steht `sub-554`
# **zweimal** in der CSV; dieses Duplikat steckt in `dataset.ids` / `len(dataset)`, während die
# Maskenliste eindeutige IDs verwendet.
#
# Die ~1 % Thalamusanteil am Hirnvolumen sind das **Zufallsniveau** aller späteren
# Lokalisationsmetriken.
#
# ### Was die Abbildung zeigt
#
# Drei Schnittebenen durch den Schwerpunkt des Thalamus (sagittal / koronal / axial), jeweils
# Grauwertbild und darüber die Maske in Rot. Am aussagekräftigsten ist der **koronale** Schnitt:
# zwei kompakte, spiegelsymmetrische Klumpen tief in der Mitte — genau dort, wo der Thalamus
# anatomisch hingehört.
#
# ### Einordnung
#
# Die Datenstruktur ist bewusst **dieselbe** wie in `Explain_brain_age_predictions`. Was hier
# zusätzlich passiert, ist nur der Atlas-Schritt: Voxel $(i,j,k)$ → FreeSurfer-Label → Aussage
# „liegt im Thalamus". Ohne Maske gibt es keine Ground Truth, ohne Ground Truth keine Bewertung
# der Erklärung.

# %%
import nibabel as nib
import pandas as pd

from nibabel.processing import resample_from_to
from pyment.data import NiftiDataset, AsyncNiftiGenerator
from pyment.data.preprocessors import NiftiPreprocessor

# FreeSurfer-Labelnummern (siehe FastSurferCNN/config/FreeSurferColorLUT.txt)
THALAMUS_LEFT = 10
THALAMUS_RIGHT = 49
THALAMUS_LABELS = (THALAMUS_LEFT, THALAMUS_RIGHT)

# --- Schritt A: IXI wie in Explain_brain_age_predictions -----------------------------
ixi_folder = repo_root / "data" / "mri" / "ixi"
# Wichtig: Root = cropped/, NICHT cropped/images/ — from_folder haengt images/ und labels.csv an
image_folder = ixi_folder / "cropped"
fastsurfer_folder = ixi_folder / "fastsurfer"
ASEG_FILENAME = "aparc.DKTatlas+aseg.deep.mgz"

dataset = None
preprocessor = None
generator = None
subjects: list[str] = []
HAVE_REAL_DATA = False

if (image_folder / "labels.csv").is_file() and (image_folder / "images").is_dir():
    dataset = NiftiDataset.from_folder(str(image_folder), target="age")
    preprocessor = NiftiPreprocessor(sigma=255.0)
    generator = AsyncNiftiGenerator(
        dataset=dataset,
        preprocessor=preprocessor,
        batch_size=4,
        threads=8,
    )
    print(f"Dataset: {len(dataset)} Einträge, "
          f"{len(set(dataset.ids))} eindeutige IDs (Ziel = {dataset.target})")
    print(f"Bildordner : {image_folder / 'images'}")
    print(f"Labels     : {image_folder / 'labels.csv'}")
    print(f"Alter min/max: {float(np.min(dataset.y)):.1f} / {float(np.max(dataset.y)):.1f} Jahre")
else:
    print(f"Kein IXI-Dataset unter {image_folder} "
          f"(erwartet: images/*.nii.gz + labels.csv).")


def subjects_with_aseg() -> list[str]:
    """Eindeutige Dataset-IDs, fuer die auch eine FastSurfer-Segmentierung vorliegt."""
    if dataset is None:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for sid in dataset.ids:
        if sid in seen:
            continue
        seen.add(sid)
        if (fastsurfer_folder / sid / "mri" / ASEG_FILENAME).is_file():
            out.append(sid)
    return out


def path_for_subject(subject: str) -> Path:
    """Bildpfad aus dem Dataset (gleiche Quelle wie der Generator)."""
    for sid, path in zip(dataset.ids, dataset.paths):
        if sid == subject:
            return Path(path)
    raise KeyError(f"{subject} nicht im Dataset")


def age_for_subject(subject: str) -> float:
    for sid, age in zip(dataset.ids, dataset.y):
        if sid == subject:
            return float(age)
    raise KeyError(f"{subject} nicht im Dataset")


def load_subject(subject: str):
    """Bild, Thalamusmaske und Hirnmaske — alle auf dem Gitter des Modell-Bildes."""
    image = nib.load(str(path_for_subject(subject)))
    aseg = nib.load(fastsurfer_folder / subject / "mri" / ASEG_FILENAME)

    # order=0 == nearest neighbour: Labelnummern duerfen nicht interpoliert werden
    segmentation = resample_from_to(aseg, image, order=0).get_fdata()

    volume = image.get_fdata()
    thalamus = np.isin(segmentation, THALAMUS_LABELS)
    brain = volume != 0

    return volume, thalamus, brain, segmentation, image


# --- Schritt B: Thalamusmasken im Modellgitter ---------------------------------------
subjects = subjects_with_aseg()
HAVE_REAL_DATA = len(subjects) > 0
print(f"Subjekte mit Bild + Alter + Segmentierung: {len(subjects)}")

if HAVE_REAL_DATA:
    rows = []
    for subject in subjects:
        volume, thalamus, brain, segmentation, image = load_subject(subject)
        left = np.argwhere(segmentation == THALAMUS_LEFT).mean(axis=0)
        right = np.argwhere(segmentation == THALAMUS_RIGHT).mean(axis=0)
        rows.append({
            "id": subject,
            "age": round(age_for_subject(subject), 2),
            "thal_voxel": int(thalamus.sum()),
            "hirn_voxel": int(brain.sum()),
            "anteil_%": round(100 * thalamus.sum() / brain.sum(), 2),
            "ausserhalb": int((thalamus & ~brain).sum()),
            "schwerpunkt_links": tuple(left.round().astype(int)),
            "schwerpunkt_rechts": tuple(right.round().astype(int)),
        })

    mask_table = pd.DataFrame(rows)
    mask_table.to_csv(target_dir / "01_thalamus_masken.csv", index=False)
    print(mask_table.to_string(index=False))

    # Reihenfolge = Dataset-Reihenfolge (wie Explain), nicht alphabetisch
    reference_subject = subjects[0]
    volume, thalamus, brain, segmentation, image = load_subject(reference_subject)
    nib.save(nib.Nifti1Image(thalamus.astype(np.uint8), image.affine, image.header),
             target_dir / f"01_{reference_subject}_thalamus_mask.nii.gz")

    centre = np.argwhere(thalamus).mean(axis=0).round().astype(int)
    print(f"\nSchnittebenen durch den Thalamus-Schwerpunkt von {reference_subject}: "
          f"{tuple(int(v) for v in centre)}")

    fig, ax = plt.subplots(2, 3, figsize=(13, 9))
    views = [
        ("sagittal", volume[centre[0]], thalamus[centre[0]]),
        ("koronal", volume[:, centre[1]], thalamus[:, centre[1]]),
        ("axial", volume[:, :, centre[2]], thalamus[:, :, centre[2]]),
    ]
    for col, (name, slice_image, slice_mask) in enumerate(views):
        rotate = name != "axial"
        bg = np.rot90(slice_image) if rotate else slice_image
        mk = np.rot90(slice_mask) if rotate else slice_mask
        ax[0][col].imshow(bg, cmap="Greys_r")
        ax[0][col].set_title(f"{name} — MRT")
        ax[1][col].imshow(bg, cmap="Greys_r")
        ax[1][col].imshow(np.ma.masked_where(~mk, mk), cmap="autumn", alpha=0.75)
        ax[1][col].set_title(f"{name} — Thalamusmaske")
        for row in range(2):
            ax[row][col].axis("off")
    fig.suptitle(f"Thalamusmaske aus dem FastSurfer-aseg ({reference_subject})", fontsize=14)
    fig.savefig(target_dir / "01_thalamus_maske.png", bbox_inches="tight", dpi=150)
    plt.show()
else:
    print("Keine IXI-Daten mit Alter + Segmentierung — Abschnitte 3, 4, 7 und 8 übersprungen.")
    print(f"Erwartet: {image_folder}/images/<id>.nii.gz, {image_folder}/labels.csv und "
          f"{fastsurfer_folder}/<id>/mri/{ASEG_FILENAME}")

# %% [markdown]
# <a id="sec-04"></a>
# ## 4. Kontrolldatensatz A: alles außer dem Thalamus permutieren
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Die Idee
#
# Wir nehmen ein echtes MRT und **mischen die Grauwerte aller Voxel außerhalb des Thalamus
# durch**. Formal: sei $\Omega$ die Menge aller Hirnvoxel, $T \subset \Omega$ der Thalamus. Wir
# wählen eine zufällige Permutation $\pi$ auf $\Omega \setminus T$ und setzen
#
# $$X'_v = \begin{cases}
#   X_v & v \in T \quad\text{(Thalamus bleibt unangetastet)}\\
#   X_{\pi(v)} & v \in \Omega \setminus T \quad\text{(alles andere wird durchmischt)}\\
#   0 & v \notin \Omega \quad\text{(Hintergrund bleibt 0)}
# \end{cases}$$
#
# Das hat drei sehr angenehme Eigenschaften:
#
# 1. **Die Bilddimensionen bleiben exakt gleich.** Kein Resampling, kein neues Cropping — das
#    Bild passt unverändert in das Modell.
# 2. **Das Grauwert-Histogramm bleibt exakt gleich.** Eine Permutation verschiebt Werte, sie
#    erzeugt und löscht keine. Ein Modell kann den Unterschied also nicht an der
#    Intensitätsverteilung erkennen, nur an der **räumlichen Struktur**.
# 3. **Die Gehirnform bleibt erhalten**, weil nur Voxel mit $X_v \neq 0$ getauscht werden. Die
#    Silhouette des Kopfes ist unverändert; im Inneren ist alles zerstört.
#
# Damit gilt: **Außerhalb des Thalamus gibt es nachweislich keine anatomische Information mehr.**
# Jede Relevanz, die LRP dort verortet, kann keine echte Struktur widerspiegeln — sie ist entweder
# Rauschen oder ein Artefakt des Verfahrens.
#
# ### Warum das noch keine vollständige Ground Truth ist
#
# Aufgepasst — hier liegt der subtile Punkt: Wir wissen, wo Information **nicht** ist. Wir wissen
# **nicht**, ob das Modell die im Thalamus verbliebene Information überhaupt liest. Ein Modell,
# das schlicht eine Konstante ausgibt, „besteht" diesen Test möglicherweise trotzdem, weil LRP
# seine Relevanz an der lokalen Bildstruktur entlang verteilt. Genau das passiert in
# [Abschnitt 8](#sec-08).
#
# ### Zwei Varianten der Zerstörung
#
# Die Zelle erzeugt zwei Kontrollbilder, weil sie unterschiedliche Fragen beantworten:
#
# | Variante | Vorgehen | Was sie prüft |
# |---|---|---|
# | `shuffled` | Werte außerhalb des Thalamus **permutieren** | Histogramm identisch → Modell kann nur Struktur nutzen |
# | `noise` | Werte außerhalb des Thalamus durch **Gleichverteilung** aus dem Hirn-Wertebereich ersetzen | robuster gegen Sonderfälle, Histogramm ändert sich leicht |
#
# `shuffled` ist die sauberere Variante (das ist die aus der Projektdiskussion), `noise` dient
# als Gegenprobe.
#
# ### Ausgabe dieser Zelle
#
# Die Kontrollzeilen belegen, dass die Konstruktion hält:
#
# ```text
# Thalamus-Voxel unverändert  : True
# Histogramm identisch        : True
# Hintergrund weiterhin 0     : True
# Voxel mit geändertem Wert   : 1,783,466
# (Hirnvoxel minus Thalamus   : 1,823,354)
# ```
#
# Die letzten zwei Zeilen sind **absichtlich** nicht gleich, und der Grund ist ein hübsches
# kleines Detail: Die Bilder haben nur 256 mögliche Grauwerte, aber 1,8 Mio. zu permutierende
# Voxel. Etwa 40 000 von ihnen (2 %) bekommen beim Mischen zufällig denselben Wert, den sie schon
# hatten, und zählen dann nicht als „geändert". Das ist kein Fehler, sondern ein
# Taubenschlag-Effekt — und ein guter Anlass, sich Kontrollzeilen nicht blind, sondern mit einer
# Erwartung anzuschauen.
#
# ### Was die Abbildung zeigt
#
# Links das Original, in der Mitte `shuffled`, rechts `noise` — jeweils der axiale Schnitt durch
# den Thalamus.
#
# Im Original erkennt man die typische Hirnanatomie: helle weiße Substanz innen, dunklere
# Hirnrinde außen, die dunklen Ventrikel. Bei `shuffled` und `noise` ist davon **nichts** übrig:
# ein gleichmäßiges Grieseln in Kopfform. Der **einzige** zusammenhängende, glatte Bereich im
# ganzen Bild ist der Thalamus — er sticht als klar erkennbarer Doppelklumpen hervor.
#
# Dass er so deutlich hervorsticht, ist gleichzeitig die **Achillesferse** dieses Testdesigns.
# Merken Sie sich das Bild; in Abschnitt 8 kommen wir darauf zurück.

# %%
def shuffle_outside(volume: np.ndarray, keep: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permutiert alle Hirnvoxel ausserhalb von `keep`; Histogramm bleibt erhalten."""
    out = volume.copy()
    selection = (~keep) & (volume != 0)
    values = out[selection]
    rng.shuffle(values)
    out[selection] = values
    return out


def noise_outside(volume: np.ndarray, keep: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ersetzt alle Hirnvoxel ausserhalb von `keep` durch Rauschen im Hirn-Wertebereich."""
    out = volume.copy()
    selection = (~keep) & (volume != 0)
    inside = volume[volume != 0]
    out[selection] = rng.uniform(inside.min(), inside.max(), int(selection.sum()))
    return out


if HAVE_REAL_DATA:
    rng = np.random.default_rng(RANDOM_SEED)
    volume, thalamus, brain, segmentation, image = load_subject(reference_subject)

    shuffled = shuffle_outside(volume, thalamus, rng)
    noised = noise_outside(volume, thalamus, rng)

    print(f"Subjekt: {reference_subject}")
    print(f"Thalamus-Voxel unverändert  : "
          f"{np.array_equal(shuffled[thalamus], volume[thalamus])}")
    print(f"Histogramm identisch        : "
          f"{np.array_equal(np.sort(shuffled.ravel()), np.sort(volume.ravel()))}")
    print(f"Hintergrund weiterhin 0     : {np.all(shuffled[~brain] == 0)}")
    print(f"Voxel mit geändertem Wert   : {int((shuffled != volume).sum()):,}")
    print(f"(Hirnvoxel minus Thalamus   : {int(brain.sum() - thalamus.sum()):,})")

    nib.save(nib.Nifti1Image(shuffled, image.affine, image.header),
             target_dir / f"02_{reference_subject}_thalamus_preserved_shuffled.nii.gz")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for col, (name, data) in enumerate([("Original", volume),
                                        ("shuffled (permutiert)", shuffled),
                                        ("noise (ersetzt)", noised)]):
        ax[col].imshow(data[:, :, centre[2]], cmap="Greys_r", vmin=0, vmax=volume.max())
        ax[col].set_title(name)
        ax[col].axis("off")
    fig.suptitle("Kontrolldatensatz A: nur der Thalamus behält seine Struktur "
                 f"(axialer Schnitt, {reference_subject})", fontsize=13)
    fig.savefig(target_dir / "02_shuffle_kontrolle.png", bbox_inches="tight", dpi=150)
    plt.show()

# %% [markdown]
# <a id="sec-05"></a>
# ## 5. Kontrolldatensatz B: vollsynthetische Phantome
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Warum ein zweiter Datensatz?
#
# Datensatz A kann man nur mit einem **fertigen** Modell benutzen — und ein fertiges Modell sagt
# irgendetwas vorher (hier: das Alter), aber nicht das Thalamusvolumen. Um zu prüfen, ob LRP das
# Thalamusvolumen findet, brauchen wir ein Modell, das das Thalamusvolumen **tatsächlich messen
# muss**. Und dafür brauchen wir viele Bilder mit bekanntem Thalamusvolumen.
#
# Deshalb bauen wir die Bilder selbst: ein „Gehirn" aus **vier ineinandergeschachtelten
# Kompartimenten** in einem $64^3$-Würfel. **Das ist bewusst kein IXI-Pfad** — keine NIfTIs,
# kein `labels.csv`, kein `NiftiDataset`. Die IXI-Datenstruktur aus Abschnitt 3 bleibt die des
# Brain-Age-Notebooks; die Phantome sind nur der kontrollierte Prüfstand für Abschnitte 9–15.
#
#
# ```text
#   Querschnitt durch ein Phantom (schematisch)
#
#   ░░░░░░░░░░░░░░░░░░░░░░░░░   ░ Hintergrund      Grauwert 0.00
#   ░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░   ▒ Kortex           Grauwert 0.45
#   ░░▒▒███████████████▒▒░░░░   █ weisse Substanz  Grauwert 0.80
#   ░▒▒████·······██████▒▒░░░   · Ventrikel        Grauwert 0.06
#   ░▒▒█████████████████▒▒░░░   ▓ THALAMUS         Grauwert 0.62
#   ░▒▒███▓▓▓▓█▓▓▓▓█████▒▒░░░
#   ░░▒▒███████████████▒▒░░░░       Zielwert y = Anzahl Thalamus-Voxel / 100
#   ░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░
# ```
#
# ### Der entscheidende Punkt: Konfounder ausschließen
#
# Das Ganze ist nur dann ein gültiger Test, wenn **nichts anderes im Bild** das Thalamusvolumen
# verrät. Sonst kann das Netz die richtige Antwort geben, ohne den Thalamus je angeschaut zu
# haben — und LRP hätte recht, wenn es woanders hinzeigt. Ein solcher Nebenweg heißt in der
# Statistik **Konfounder**, in der XAI-Literatur *shortcut* oder *Clever-Hans-Effekt*.
#
# Deshalb wird **jede** Eigenschaft unabhängig gezogen:
#
# | Eigenschaft | Bereich | Rolle |
# |---|---|---|
# | **Thalamusradius** | 3,0 – 7,0 Voxel | **das Signal** — bestimmt den Zielwert |
# | Kopfradius (3 Achsen) | 24 – 28 Voxel | Störgröße |
# | Kortexdicke | 2,5 – 5,0 Voxel | Störgröße |
# | Ventrikelradius | 3,0 – 6,5 Voxel | Störgröße |
# | Kopfposition | ± 2 Voxel | Störgröße |
# | Globaler Intensitätsfaktor | 0,90 – 1,10 | Störgröße |
# | Rauschstärke | σ = 0,01 – 0,04 | Störgröße |
#
# Und der Zielwert ist genau das, was wir messen wollen — das **Volumen**, nicht der Radius:
#
# $$y \;=\; \frac{1}{100}\,\bigl|\{v : \text{label}(v) = \text{Thalamus}\}\bigr|$$
#
# Der Faktor $1/100$ ist reine Kosmetik, damit die Zahlen im Bereich 2–30 statt 200–3000 liegen;
# das erleichtert dem Netz den Start.
#
# Die Zelle prüft die Unabhängigkeit anschließend mit **Korrelationskoeffizienten**. Ein Wert
# nahe 0 bedeutet „kein linearer Zusammenhang":
#
# $$\rho(y, m) = \frac{\operatorname{Cov}(y, m)}{\sigma_y \, \sigma_m} \in [-1, 1]$$
#
# geprüft für $m \in \{$Hirnvolumen, Ventrikelvolumen, Kortexvolumen, mittlere Intensität$\}$.
#
# > **Ein Fallstrick, in den wir beim Bauen tatsächlich gelaufen sind:** In der ersten Version
# > lagen Ventrikel und Thalamus räumlich so nah, dass ein wachsender Thalamus Ventrikelvoxel
# > überschrieb. Ergebnis: $\rho(y, \text{Ventrikelvolumen}) = -0{,}64$. Das Netz hätte die
# > Aufgabe lösen können, indem es den **Ventrikel** ausmisst — und LRP hätte korrekt auf den
# > Ventrikel gezeigt, während wir „falsch!" gerufen hätten. Erst nach dem Auseinanderrücken der
# > beiden Strukturen sank die Korrelation auf $\approx -0{,}04$. **Konfounder muss man messen,
# > nicht hoffen.**
#
# ### Ausgabe dieser Zelle
#
# ```text
# Phantome              : 600 à 64³
# Zielwert y            : 2.35 .. 29.85   (Mittel 12.76, Std 8.09)
# Thalamusanteil am Hirn: 0.29 % .. 4.94 %
#
# corr(y, Hirnvolumen       ) = -0.035
# corr(y, Ventrikelvolumen  ) = -0.044
# corr(y, Kortexvolumen     ) = -0.023
# corr(y, mittl. Intensität ) = -0.079
#
# Baseline-MAE (Mittelwert vorhersagen) = 6.94
# Baseline-MSE (= Varianz von y)        = 65.49
# ```
#
# Alle vier Korrelationen liegen betragsmäßig unter 0,08 — verglichen mit den −0,64 aus dem
# Fehlversuch oben ist der Datensatz damit konfounderfrei. Der **Thalamusanteil von 0,29 bis
# 4,94 %** umschließt den realistischen Wert von ≈ 1 % aus Abschnitt 3; die Spanne ist absichtlich
# breiter gewählt, damit das Modell überhaupt etwas zu unterscheiden hat.
#
# Die **Baseline** ist die wichtigste Zahl für später: Ein Modell, das stur den Mittelwert
# vorhersagt, erreicht einen mittleren absoluten Fehler (MAE) von 6,94. **Jedes sinnvolle Modell
# muss deutlich darunter liegen** — sonst hat es nichts gelernt, und seine Erklärung ist wertlos.
#
# ### Was die Abbildung zeigt
#
# Fünf Phantome mit aufsteigendem Thalamusvolumen (Zeile 1: Grauwertbild, Zeile 2: die
# Kompartiment-Labels als Farbkarte). In der Labelzeile ist der **Thalamus das helle Paar**, die
# **Ventrikel sind das gelbe Paar**, weiße Substanz ist grün und Kortex blau.
#
# Von links nach rechts wächst das helle Thalamuspaar deutlich sichtbar — von zwei kaum erkennbaren
# Punkten auf zwei große Scheiben. Kopfgröße, Kortexdicke, Ventrikelgröße und Bildhelligkeit
# variieren dabei **unsystematisch**: Im vierten Bild etwa sind die Ventrikel groß, im fünften klein,
# obwohl der Thalamus dort am größten ist. Genau so soll es sein: das Signal ist sichtbar, die
# Störgrößen sind sichtbar, und sie haben nichts miteinander zu tun.
#
# Zwei Details, die für die spätere Interpretation wichtig werden: Der Thalamus ist im Grauwertbild
# nur **wenig dunkler** als die umgebende weiße Substanz (0,62 gegen 0,80) — sein Rand ist also ein
# schwacher Kontrast. Die Ventrikel dagegen sind fast schwarz (0,06) und bilden die **kräftigste
# Kante** im Inneren des Kopfes. Für einen Kantendetektor sind sie damit deutlich auffälliger als
# das eigentliche Ziel.

# %%
PHANTOM_SIZE = 64
N_PHANTOMS = 600

LABEL_BACKGROUND = 0
LABEL_CORTEX = 1
LABEL_WHITE_MATTER = 2
LABEL_VENTRICLE = 3
LABEL_THALAMUS = 4

COMPARTMENTS = {
    LABEL_CORTEX: ("Kortex", 0.45),
    LABEL_WHITE_MATTER: ("weisse Substanz", 0.80),
    LABEL_VENTRICLE: ("Ventrikel", 0.06),
    LABEL_THALAMUS: ("Thalamus", 0.62),
}

# Voxelkoordinaten einmal vorberechnen - macht die Generierung ~10x schneller
_grid = np.stack(np.meshgrid(*[np.arange(PHANTOM_SIZE)] * 3, indexing="ij"),
                 axis=-1).astype(np.float32)


def _ellipsoid(center, radii) -> np.ndarray:
    """Boolesche Maske eines achsenparallelen Ellipsoids: sum((x-c)^2/r^2) <= 1."""
    d = (_grid - np.asarray(center, np.float32)) / np.asarray(radii, np.float32)
    return (d ** 2).sum(-1) <= 1.0


def make_phantom(rng: np.random.Generator):
    """Ein Phantom. Alle Groessen unabhaengig gezogen; Zielwert steckt im Thalamus."""
    label = np.full((PHANTOM_SIZE,) * 3, LABEL_BACKGROUND, np.uint8)
    center = np.array([PHANTOM_SIZE / 2] * 3) + rng.uniform(-2, 2, 3)

    head_radii = rng.uniform(24, 28, 3)
    label[_ellipsoid(center, head_radii)] = LABEL_CORTEX
    label[_ellipsoid(center, head_radii - rng.uniform(2.5, 5.0))] = LABEL_WHITE_MATTER

    # Ventrikel deutlich anterior - darf den Thalamus nicht beruehren (Konfounder!)
    ventricle_radius = rng.uniform(3.0, 6.5)
    for side in (-1, 1):
        label[_ellipsoid(center + [side * 5.0, 12.0, 2.0],
                         [ventricle_radius * 0.7, ventricle_radius,
                          ventricle_radius * 1.3])] = LABEL_VENTRICLE

    # Der Thalamus: paarig, zentral, posterior der Ventrikel. Radius = das Signal.
    thalamus_radius = rng.uniform(3.0, 7.0)
    for side in (-1, 1):
        label[_ellipsoid(center + [side * (thalamus_radius + 0.8), -6.0, 0.0],
                         [thalamus_radius, thalamus_radius * 1.15,
                          thalamus_radius * 0.9])] = LABEL_THALAMUS

    volume = np.zeros((PHANTOM_SIZE,) * 3, np.float32)
    for key, (_, intensity) in COMPARTMENTS.items():
        volume[label == key] = intensity

    volume *= rng.uniform(0.9, 1.1)
    noise = rng.normal(0, rng.uniform(0.01, 0.04), volume.shape).astype(np.float32)
    volume = np.clip(volume + noise * (label != LABEL_BACKGROUND), 0, 1)

    return volume, label


def make_phantom_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = np.zeros((n,) + (PHANTOM_SIZE,) * 3, np.float32)
    L = np.zeros((n,) + (PHANTOM_SIZE,) * 3, np.uint8)
    for i in range(n):
        X[i], L[i] = make_phantom(rng)
    y = (L == LABEL_THALAMUS).sum((1, 2, 3)).astype(np.float32) / 100.0
    return X, L, y


phantom_X, phantom_L, phantom_y = make_phantom_dataset(N_PHANTOMS, RANDOM_SEED)

n_brain = (phantom_L != LABEL_BACKGROUND).sum((1, 2, 3))
n_ventricle = (phantom_L == LABEL_VENTRICLE).sum((1, 2, 3))
n_cortex = (phantom_L == LABEL_CORTEX).sum((1, 2, 3))
share = 100 * (phantom_y * 100) / n_brain

print(f"Phantome              : {N_PHANTOMS} à {PHANTOM_SIZE}³")
print(f"Zielwert y            : {phantom_y.min():.2f} .. {phantom_y.max():.2f}   "
      f"(Mittel {phantom_y.mean():.2f}, Std {phantom_y.std():.2f})")
print(f"Thalamusanteil am Hirn: {share.min():.2f} % .. {share.max():.2f} %")
print()
for name, other in [("Hirnvolumen", n_brain), ("Ventrikelvolumen", n_ventricle),
                    ("Kortexvolumen", n_cortex),
                    ("mittl. Intensität", phantom_X.mean((1, 2, 3)))]:
    print(f"corr(y, {name:18s}) = {np.corrcoef(phantom_y, other)[0, 1]:+.3f}")

phantom_baseline_mae = float(np.abs(phantom_y - phantom_y.mean()).mean())
print(f"\nBaseline-MAE (Mittelwert vorhersagen) = {phantom_baseline_mae:.2f}")
print(f"Baseline-MSE (= Varianz von y)        = {phantom_y.var():.2f}")

order = np.argsort(phantom_y)
examples = order[np.linspace(0, len(order) - 1, 5).astype(int)]
mid = PHANTOM_SIZE // 2

fig, ax = plt.subplots(2, 5, figsize=(15, 6.5))
for col, index in enumerate(examples):
    thal_slice = int(np.argwhere(phantom_L[index] == LABEL_THALAMUS)[:, 2].mean())
    ax[0][col].imshow(phantom_X[index][:, :, thal_slice], cmap="Greys_r", vmin=0, vmax=1)
    ax[0][col].set_title(f"y = {phantom_y[index]:.1f}")
    ax[1][col].imshow(phantom_L[index][:, :, thal_slice], cmap="nipy_spectral",
                      vmin=0, vmax=LABEL_THALAMUS)
    for row in range(2):
        ax[row][col].axis("off")
fig.suptitle("Phantome nach Thalamusvolumen sortiert — oben Bild, unten Kompartiment-Labels "
             "(Thalamus = hellste Farbe)", fontsize=13)
fig.savefig(target_dir / "03_phantome_uebersicht.png", bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# <a id="sec-06"></a>
# ## 6. Woran misst man Erfolg? Die Ground-Truth-Metriken
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Warum man nicht einfach hinschauen darf
#
# Der übliche Umgang mit Heatmaps ist: anschauen und sagen „sieht plausibel aus". Das ist aus drei
# Gründen unzureichend. Erstens sind 3D-Karten mit 5,7 Mio. Voxeln gar nicht als Ganzes
# betrachtbar — man sieht immer nur ein paar Schnitte. Zweitens findet das Auge in jeder
# verrauschten Karte Muster. Drittens lässt sich „sieht plausibel aus" nicht zwischen zwei
# Verfahren vergleichen. Wir brauchen **Zahlen**.
#
# Gegeben sind: eine Relevanzkarte $R_v$, die Zielmaske $T$ (Thalamus) und die Hirnmaske $\Omega$.
# Weil sich positive und negative Relevanz beim Aufsummieren gegenseitig auslöschen würden,
# rechnen die massebezogenen Maße mit dem **positiven Anteil** $R^+_v = \max(R_v, 0)$.
#
# ---
#
# ### Metrik 1: Relevanzmasse im Ziel
#
# $$m \;=\; \frac{\sum_{v \in T} R^+_v}{\sum_{v \in \Omega} R^+_v}$$
#
# *„Welcher Anteil der gesamten positiven Relevanz liegt im Thalamus?"* Wertebereich $[0, 1]$.
#
# Diese Zahl allein ist irreführend: eine große Region sammelt schon aus Versehen viel Relevanz.
# Deshalb braucht sie immer einen Bezugspunkt.
#
# ### Metrik 2: Dichte-Verhältnis (*relevance mass ratio*)
#
# Der Bezugspunkt ist der **Volumenanteil** $s = |T| / |\Omega|$ — der Wert, den man bei völlig
# gleichmäßig verteilter Relevanz erwarten würde:
#
# $$\text{ratio} \;=\; \frac{m}{s}
#   \;=\; \underbrace{\frac{\sum_{v\in T} R^+_v}{|T|}}_{\text{Relevanzdichte im Ziel}}
#     \Bigg/ \underbrace{\frac{\sum_{v\in \Omega} R^+_v}{|\Omega|}}_{\text{Dichte im ganzen Hirn}}$$
#
# Das ist die **Leitmetrik dieses Notebooks**. Sie ist dimensionslos und direkt zu lesen:
#
# | Wert | Bedeutung |
# |---|---|
# | $\text{ratio} = 1$ | keine Lokalisation — Relevanz ist gleichmäßig verschmiert |
# | $\text{ratio} > 1$ | Relevanz ist im Ziel **angereichert** |
# | $\text{ratio} \gg 1$ | starke Lokalisation |
# | $\text{ratio} < 1$ | Relevanz **meidet** das Ziel |
#
# Der theoretische Bestwert ist $1/s$ — bei $s = 1\,\%$ also 100. Das entspräche „100 % der
# Relevanz genau im Thalamus".
#
# ### Metrik 3: Top-$k$-Präzision
#
# Wir nehmen die $k = |T|$ Voxel mit der größten **absoluten** Relevanz und fragen, welcher Anteil
# davon im Thalamus liegt:
#
# $$\text{prec}@k \;=\; \frac{1}{k} \bigl|\, \operatorname{Top}_k(|R|) \cap T \,\bigr|,
#   \qquad k = |T|$$
#
# Der Zufallswert ist $s$ (also ≈ 0,01), der Bestwert 1. Anders als Metrik 2 ist dieses Maß
# **rangbasiert**: es ignoriert, wie groß die Relevanzwerte sind, und schaut nur auf die
# Reihenfolge. Damit ist es robust gegen einzelne Ausreißer, die eine Summe dominieren können.
#
# ### Metrik 4: Pointing Game
#
# Die härteste und gröbste Prüfung — liegt das **allerrelevanteste** Voxel im Ziel?
#
# $$\text{hit} \;=\; \mathbb{1}\bigl[\arg\max_{v \in \Omega} |R_v| \in T\bigr]$$
#
# Pro Bild 0 oder 1, über viele Bilder gemittelt eine Trefferquote. Der Zufallswert ist wieder
# $s$. Das Maß stammt aus der Objektlokalisierung in 2D-Bildern und ist beliebt, weil es so
# einfach ist — aber es hängt an einem einzigen Voxel und ist entsprechend verrauschten
# Schwankungen ausgesetzt.
#
# ### Metrik 5: Dilatierte Maske und Distanzprofil
#
# Die vier Metriken oben behandeln die Thalamusgrenze als messerscharf. Das ist zu streng, und
# zwar aus einem inhaltlichen Grund: **Evidenz für eine Größe muss nicht innerhalb der Struktur
# liegen.** Um zu erkennen, dass der Thalamus groß ist, kann ein Netz genauso gut den **Kontrast
# an seinem Rand** benutzen — und dieser Rand liegt teils schon in der weißen Substanz. Ein
# menschlicher Radiologe würde es genauso machen.
#
# Deshalb messen wir zusätzlich mit einem Toleranzrand. Sei $d(v)$ der euklidische Abstand von
# Voxel $v$ zum nächsten Thalamusvoxel (0 innerhalb):
#
# $$T_r = \{v : d(v) \le r\}, \qquad r \in \{2, 4, 6\}\ \text{Voxel (= mm)}$$
#
# Zusätzlich zerlegen wir die Relevanz in **Abstandsschalen** — das zeigt, ob die Relevanz vom
# Thalamus nach außen abfällt (gute Lokalisation) oder unabhängig vom Abstand verteilt ist:
#
# $$\text{Profil}(a, b) = \frac{\sum_{v:\, a < d(v) \le b} R^+_v}{\sum_{v \in \Omega} R^+_v}$$
#
# ### Warum es immer eine Kontrollbedingung braucht
#
# Keine dieser Zahlen ist für sich interpretierbar. „ratio = 5,2" ist nur dann eine Aussage, wenn
# man weiß, was dieselbe Rechnung liefert, wenn nichts los ist. Dieses Notebook nutzt drei
# Kontrollen:
#
# | Kontrolle | Erwartung, wenn die Methode funktioniert | Abschnitt |
# |---|---|---|
# | **Modell mit Zufallsgewichten** | ratio ≈ 1 (oder darunter) | [15](#sec-15) |
# | **Modell, das den Input ignoriert** | ratio ≈ 1 | [8](#sec-08) |
# | **andere Kompartimente als Ziel** | ratio ≈ 1 für Kortex/WM/Ventrikel | [12](#sec-12) |
#
# ### Ausgabe dieser Zelle
#
# Die Zelle definiert nur Funktionen — und prüft sie an zwei künstlichen Karten mit bekanntem
# Ergebnis, damit ein Vorzeichen- oder Normierungsfehler nicht erst später als „interessanter
# Befund" auffällt:
#
# ```text
# Selbsttest der Metriken (Phantom 0)
#   Volumenanteil des Thalamus s   = 0.0050
#   perfekte Karte  : ratio =   201.91  (theoretisches Maximum 1/s = 201.91)
#   gleichverteilt  : ratio =     1.00  (Sollwert 1.00)
# ```
#
# Beide Sollwerte werden exakt getroffen. Die **perfekte** Karte (Relevanz genau 1 im Thalamus, 0
# überall sonst) erreicht mit 201,91 genau das theoretische Maximum $1/s$ — bei diesem Phantom
# nimmt der Thalamus 0,50 % des Hirnvolumens ein. Die **gleichverteilte** Karte landet exakt auf
# 1,00, dem Neutralwert. Damit ist gezeigt: Die Skala ist richtig kalibriert, und alles, was später
# zwischen 1 und 202 liegt, ist ein echtes Zwischenergebnis und kein Artefakt der Metrik.
#
# ### Einordnung
#
# Solche Maße sind der Standard in der Attributions-Evaluation. Die naheliegende Alternative sind
# **Perturbationsmaße** (*pixel flipping*, *ROAD*): man löscht die als wichtig markierten Voxel
# und schaut, wie stark die Vorhersage einbricht. Diese kommen ohne Ground-Truth-Maske aus, haben
# aber ein eigenes Problem — das Löschen erzeugt Bilder, die es im Training nie gab (*out of
# distribution*), und der Einbruch kann daher auch davon kommen. Weil wir hier eine echte Maske
# haben, ist der maskenbasierte Weg der direktere.

# %%
from scipy import ndimage

DILATION_RADII = (2, 4, 6)
PROFILE_EDGES = (0, 1, 2, 4, 8, 16, np.inf)
PROFILE_LABELS = ["im Thalamus", "0–1", "1–2", "2–4", "4–8", "8–16", "> 16"]


def relevance_metrics(R: np.ndarray, target: np.ndarray, brain: np.ndarray,
                      distance: np.ndarray = None) -> dict:
    """Ground-Truth-Metriken einer Relevanzkarte gegen eine Zielmaske.

    R        Relevanz je Voxel (Vorzeichen erhalten)
    target   boolesche Zielmaske, z.B. der Thalamus
    brain    boolesche Maske des Bereichs, ueber den normiert wird
    distance optional vorberechnete Distanztransformierte zu `target`
    """
    positive = np.maximum(R, 0)
    total = positive[brain].sum()
    n_target, n_brain = int(target.sum()), int(brain.sum())

    mass = positive[target].sum() / total if total > 0 else np.nan
    share = n_target / n_brain

    # Top-k: argpartition ist O(n) statt O(n log n) - bei 5.7 Mio. Voxeln relevant
    scores = np.where(brain, np.abs(R), -np.inf).ravel()
    top_k = np.argpartition(-scores, n_target)[:n_target]

    result = {
        "mass": float(mass),
        "share": float(share),
        "ratio": float(mass / share) if share > 0 else np.nan,
        "precision": float(target.ravel()[top_k].mean()),
        "hit": bool(target.ravel()[int(np.argmax(scores))]),
        "sum_R": float(R[brain].sum()),
        # Ohne positive Relevanz im Hirn sind Masse und Ratio undefiniert (NaN oben).
        # Diese Spalte macht sichtbar, wann genau das passiert.
        "sum_positive": float(total),
    }

    if distance is None:
        return result

    for radius in DILATION_RADII:
        dilated = (distance <= radius) & brain
        dilated_mass = positive[dilated].sum() / total if total > 0 else np.nan
        dilated_share = dilated.sum() / n_brain
        result[f"mass_r{radius}"] = float(dilated_mass)
        result[f"share_r{radius}"] = float(dilated_share)
        result[f"ratio_r{radius}"] = float(dilated_mass / dilated_share)

    for (low, high), label in zip(zip(PROFILE_EDGES[:-1], PROFILE_EDGES[1:]),
                                  PROFILE_LABELS[1:]):
        shell = (distance > low) & (distance <= high) & brain
        result[f"profil_{label}"] = float(positive[shell].sum() / total) if total > 0 else np.nan
    result[f"profil_{PROFILE_LABELS[0]}"] = result["mass"]

    return result


def distance_to(target: np.ndarray) -> np.ndarray:
    """Euklidischer Abstand jedes Voxels zur naechsten True-Position in `target`."""
    return ndimage.distance_transform_edt(~target)


def compartment_table(R: np.ndarray, label: np.ndarray, brain: np.ndarray) -> pd.DataFrame:
    """Relevanzanteil, Volumenanteil und Ratio je Kompartiment."""
    positive = np.maximum(R, 0)
    total = positive[brain].sum()
    rows = []
    for key, (name, _) in COMPARTMENTS.items():
        mask = label == key
        mass = positive[mask].sum() / total if total > 0 else np.nan
        vol = mask.sum() / brain.sum()
        rows.append({"Kompartiment": name, "R-Anteil": mass, "Vol-Anteil": vol,
                     "Ratio": mass / vol if vol > 0 else np.nan})
    return pd.DataFrame(rows)


# Selbsttest der Metriken an einer konstruierten Karte: alle Relevanz genau im Ziel
_demo_target = phantom_L[0] == LABEL_THALAMUS
_demo_brain = phantom_L[0] != LABEL_BACKGROUND
_perfect = relevance_metrics(_demo_target.astype(np.float32), _demo_target, _demo_brain)
_uniform = relevance_metrics(_demo_brain.astype(np.float32), _demo_target, _demo_brain)

print("Selbsttest der Metriken (Phantom 0)")
print(f"  Volumenanteil des Thalamus s   = {_perfect['share']:.4f}")
print(f"  perfekte Karte  : ratio = {_perfect['ratio']:8.2f}  (theoretisches Maximum "
      f"1/s = {1 / _perfect['share']:.2f})")
print(f"  gleichverteilt  : ratio = {_uniform['ratio']:8.2f}  (Sollwert 1.00)")

# %% [markdown]
# <a id="sec-07"></a>
# ## 7. Das vortrainierte Modell laden — und prüfen, ob es reagiert
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was ist das für ein Modell?
#
# `RegressionSFCN` ist das **Simple Fully Convolutional Network** von Peng et al. (2021), das
# Standardmodell für *Brain-Age-Prediction*: aus einem 3D-MRT wird das Alter der Person
# geschätzt. Der Aufbau ist bewusst schlicht — fünf gleich gebaute Blöcke, die das Volumen
# jeweils halbieren:
#
# | Index | Schicht | Ausgabe |
# |---|---|---|
# | 0–1 | Input + Reshape | `(167, 212, 160, 1)` |
# | 2–5 | Block 1: `Conv3D(32)` → BatchNorm → ReLU → `MaxPool3D(2)` | halbe Kantenlänge |
# | 6–21 | Blöcke 2–5, Kanäle 64 → 128 → 256 → 256 | jeweils halbiert |
# | 22–24 | `Conv3D(64, 1×1×1)` → BatchNorm → ReLU | Kanalmischung |
# | **25** | **`GlobalAveragePooling3D`** | **Vektor der Länge 64** |
# | 26–27 | Dropout → `Dense(1)` | eine Zahl |
# | 28–29 | ReLU + Add | Begrenzung auf $[3, 95]$ Jahre |
#
# „Fully convolutional" heißt: bis Index 24 gibt es **keine** vollverbundene Schicht. Erst das
# Global Average Pooling faltet das Restvolumen zu 64 Zahlen zusammen, und eine einzige
# `Dense`-Schicht macht daraus die Vorhersage. Das hält die Parameterzahl klein — bei 5,7 Mio.
# Voxeln pro Fall entscheidend, weil medizinische Datensätze selten mehr als ein paar tausend
# Fälle umfassen.
#
# Die Gewichte werden **direkt aus der Datei im Repo** geladen:
#
# ```text
# output/pyment/models/regression_sfcn_reg_2025_weights.h5
# ```
#
# Wir übergeben den Pfad explizit. `pyment` prüft zuerst mit `os.path.isfile()`, ob das Argument
# eine existierende Datei ist, und lädt sie dann direkt — nur wenn nicht, wird der Download-Pfad
# über `WeightRepository` angestoßen. Mit dem expliziten Pfad ist also garantiert, dass wir genau
# diese Datei benutzen und kein Netzwerkzugriff stattfindet.
#
# ### Der Test, den man nie überspringen darf
#
# Bevor man irgendetwas erklärt, prüft man: **Reagiert das Modell überhaupt auf seine Eingabe?**
# Der Test kostet drei Zeilen. Wir schicken vier grundverschiedene Bilder durch:
#
# | Eingabe | Was ein funktionierendes Modell tun müsste |
# |---|---|
# | echtes Gehirn | ein plausibles Alter ausgeben |
# | Gehirn mit permutierter Umgebung | **anders** antworten (die Anatomie ist zerstört) |
# | reines Rauschen | irgendetwas, aber sicher nicht dasselbe |
# | Nullbild (leer) | offensichtlich etwas anderes |
#
# Zusätzlich schauen wir in den **Bottleneck** — die 64 Zahlen nach dem Global Average Pooling.
# Alles, was das Modell über ein Bild „weiß", muss durch diese 64 Zahlen hindurch; die letzte
# `Dense`-Schicht sieht nichts anderes mehr. Wenn dieser Vektor für ein Gehirn und für ein leeres
# Bild identisch ist, ist die Sache erledigt.
#
# ### Ausgabe dieser Zelle
#
# ```text
# Eingabe                        Vorhersage   Bottleneck-Norm   Abstand zu Gehirn
# echtes Gehirn                      22.168            1.106               0.000
# Gehirn, Umgebung permutiert        22.169            1.106               0.000
# reines Rauschen                    22.169            1.106               0.000
# Nullbild                           22.169            1.106               0.000
#
# Spannweite der Vorhersagen über alle vier Eingaben: 0.001 Jahre
# ```
#
# ### ⚠️ Interpretation: dieser Checkpoint ist funktionslos
#
# **Ein leeres Bild, reines Rauschen und ein echtes Gehirn ergeben dieselbe Zahl** — die
# Spannweite über alle vier Eingaben beträgt **0,001 Jahre**. Auch der 64-dimensionale
# Bottleneck-Vektor ist in allen vier Fällen bis auf die dritte Dezimale identisch (Abstand
# 0,000). Das ist kein Rundungsproblem, sondern die vollständige Diagnose: Die Vorhersage hängt
# nicht von der Eingabe ab. Das Modell gibt eine Konstante zurück.
#
# Woran liegt das? Ein Blick in `pyment/models/utils/weight_repository.py` liefert den Hinweis:
#
# > *„The historical Google Drive links for 'brain-age' are dead (404); 'brain-age' therefore
# > resolves to the current reg-2025 checkpoint, which has the same SFCN topology and **loads by
# > layer order**."*
#
# „Loads by layer order" heißt: Die Gewichte werden der Reihenfolge nach in die Schichten
# geschrieben, **nicht** nach Namen zugeordnet. Passt die Reihenfolge in der Datei nicht exakt zur
# Reihenfolge im Modell, landen z. B. BatchNorm-Parameter an der falschen Stelle. Keras meldet
# dabei keinen Fehler, solange nur die *Formen* stimmen. Das Ergebnis ist ein Netz, dessen
# BatchNorm-Verschiebungen die eigentlichen Bildsignale völlig überdecken — die ReLU-Ausgaben
# hängen dann nur noch an Konstanten.
#
# Das erklärt auch die Beobachtung aus dem Schwester-Notebook `Explain_brain_age_predictions`:
# Dort liegt der mittlere Alters-Fehler bei **19,5 Jahren** statt der publizierten 3–4 Jahre, und
# alle Vorhersagen liegen bei ~22 Jahren. Das ist genau dieselbe Ursache, nur weniger direkt
# sichtbar.
#
# ### Was das für unsere Frage bedeutet
#
# Mit diesem Checkpoint lässt sich **nicht** prüfen, ob LRP das Thalamusvolumen findet. Ein Modell,
# das den Thalamus nicht liest, kann keine Erklärung haben, die auf den Thalamus zeigt. Wir haben
# damit zwei getrennte Aufgaben:
#
# * **Abschnitt 8** wendet LRP trotzdem auf dieses Modell an — als Lehrstück darüber, was die
#   Metriken bei einem Modell liefern, das nichts tut. Das Ergebnis ist überraschend.
# * **Abschnitte 9–15** beantworten die eigentliche Frage mit einem Modell, das wir selbst
#   trainieren und dessen Abhängigkeit vom Thalamus wir belegen können.
#
# ### Einordnung
#
# Der „reagiert das Modell?"-Test gehört an den Anfang jeder XAI-Pipeline, gleich hinter das
# Laden der Gewichte. Er ist trivial und fängt die peinlichsten Fehler ab: falsche Gewichtsdatei,
# stillschweigend fehlgeschlagenes Laden, vertauschte Achsen, falsches Preprocessing. Eine
# Heatmap sieht man das alles **nicht** an — LRP produziert auch für ein kaputtes Modell ein
# hübsches, farbiges, völlig sinnloses Bild.

# %%
from pyment.models import Model as PymentModel, RegressionSFCN

WEIGHTS_PATH = repo_root / "output" / "pyment" / "models" / "regression_sfcn_reg_2025_weights.h5"
print(f"Gewichtsdatei: {WEIGHTS_PATH}")
if WEIGHTS_PATH.is_file():
    print(f"  vorhanden   : ja ({WEIGHTS_PATH.stat().st_size / 1e6:.1f} MB)")
else:
    raise FileNotFoundError(
        f"Gewichtsdatei fehlt: {WEIGHTS_PATH}. Sie wird von "
        "RegressionSFCN(weights='brain-age') bei Bedarf heruntergeladen."
    )

# Explizit auf der CPU: der LRP-Rueckwaertsweg auf 167x212x160 sprengt 8 GB VRAM
with tf.device(BIG_MODEL_DEVICE):
    sfcn = RegressionSFCN(weights=str(WEIGHTS_PATH))
    sfcn_encoder = PymentModel(sfcn.input, sfcn.layers[25].output)

    print(f"\nSchichten: {len(sfcn.layers)}, Parameter: {sfcn.count_params():,}")
    print(f"Erwartete Eingabeform: {tuple(sfcn.input.shape[1:])}")

    if dataset is not None:
        # Gleicher Generator wie in Explain_brain_age_predictions — Batch-Vorhersagen
        generator.reset()
        batch_predictions = sfcn.predict(generator, verbose=0).ravel()
        print(f"\nBatch-Vorhersagen über den Dataset-Generator "
              f"({len(batch_predictions)} Einträge, inkl. CSV-Duplikate):")
        for sid, age, pred in zip(dataset.ids, dataset.y, batch_predictions):
            print(f"  {sid:10s}  Alter={float(age):5.1f}  Vorhersage={float(pred):6.3f}")
        print(f"  Spannweite der Batch-Vorhersagen: "
              f"{float(batch_predictions.max() - batch_predictions.min()):.3f} Jahre")

    if HAVE_REAL_DATA:
        rng = np.random.default_rng(RANDOM_SEED)
        volume, thalamus, brain, segmentation, image = load_subject(reference_subject)

        # Normalisierung ausschliesslich ueber NiftiPreprocessor (wie Explain)
        probes = {
            "echtes Gehirn": preprocessor(volume),
            "Gehirn, Umgebung permutiert": preprocessor(
                shuffle_outside(volume, thalamus, rng)),
            "reines Rauschen": preprocessor(
                rng.random(volume.shape) * volume.max()),
            "Nullbild": preprocessor(np.zeros_like(volume)),
        }

        reference_encoding = None
        print(f"\n{'Eingabe':32s} {'Vorhersage':>11s} {'Bottleneck-Norm':>16s} "
              f"{'Abstand zu Gehirn':>18s}")
        for name, data in probes.items():
            X = np.asarray(data, dtype=np.float32)[None]
            prediction = float(sfcn.predict(X, verbose=0).ravel()[0])
            encoding = sfcn_encoder.predict(X, verbose=0)[0]
            if reference_encoding is None:
                reference_encoding = encoding
            distance = float(np.linalg.norm(encoding - reference_encoding))
            print(f"{name:32s} {prediction:11.3f} {np.linalg.norm(encoding):16.3f} "
                  f"{distance:18.3f}")

        probe_preds = [
            float(sfcn.predict(np.asarray(d, dtype=np.float32)[None], verbose=0).ravel()[0])
            for d in probes.values()
        ]
        spread = max(probe_preds) - min(probe_preds)
        print(f"\nSpannweite der Vorhersagen über alle vier Eingaben: {spread:.3f} Jahre")
        print("→ Ein funktionierendes Modell müsste hier zweistellig streuen." if spread < 1
              else "→ Das Modell reagiert auf die Eingabe.")

# %% [markdown]
# <a id="sec-08"></a>
# ## 8. LRP auf dem vortrainierten Modell: der trügerische Erfolg
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Wir bauen den LRP-Erklärer für das SFCN und lassen ihn über alle Subjekte laufen, jeweils
# **original** und **shuffled**. Für jede Relevanzkarte berechnen wir die Metriken aus Abschnitt 6.
#
# ### Wie LRP technisch funktioniert
#
# `LRP(model, layer=..., idx=0, strategy=...)` baut aus dem trainierten Netz ein **zweites
# Keras-Modell**, das den Rückwärtspfad explizit als Schichten enthält. Man ruft es auf wie ein
# normales Modell — nur dass hinten keine Zahl, sondern ein komplettes Relevanzvolumen in der
# Größe des Eingabebildes herauskommt.
#
# * `layer=len(model.layers)-1` — welche Schicht erklärt wird (hier die letzte, also die Vorhersage)
# * `idx=0` — welches Ausgabeneuron. Bei Regression gibt es nur eines; bei einem Klassifikator
#   würde man hier die Klasse wählen, und dieselbe Eingabe ergäbe für verschiedene Klassen
#   verschiedene Heatmaps.
#
# Initialisiert wird der Rückwärtsweg damit, dass das Zielneuron seine eigene Aktivierung als
# Relevanz bekommt und alle anderen 0:
#
# $$R^{(L)}_c = a_c, \qquad R^{(L)}_{k \neq c} = 0$$
#
# Von dort wandert die Relevanz Schicht für Schicht nach unten. Die Grundregel für eine gewichtete
# Schicht: der Beitrag von Neuron $j$ zu Neuron $k$ ist $z_{jk} = a_j w_{jk}$, und die Relevanz
# wird proportional dazu zurückverteilt:
#
# $$R_j \;=\; \sum_k \frac{a_j w_{jk}}{\sum_{j'} a_{j'} w_{j'k}} \, R_k$$
#
# ### Die Regeln — und warum es mehrere gibt
#
# In der reinen Form ist die Regel numerisch instabil: steht im Nenner fast 0, explodiert der
# Bruch. Deshalb die Varianten:
#
# | Regel | Formel | Wirkung | Wo einsetzen |
# |---|---|---|---|
# | **$\varepsilon$** | $z \leftarrow z + \varepsilon\,\mathrm{sign}(z)$ | stabilisiert den Nenner, dämpft schwache Beiträge | obere Schichten |
# | **$\alpha\beta$** | $R_j = \sum_k \left(\alpha \frac{(a_j w_{jk})^+}{\sum (a w)^+} - \beta \frac{(a_j w_{jk})^-}{\sum (a w)^-}\right) R_k$, mit $\alpha - \beta = 1$ | trennt verstärkende von hemmenden Pfaden | mittlere Schichten |
# | **flat** | $a \leftarrow 1,\; w \leftarrow 1$ | verteilt Relevanz gleichmäßig über das rezeptive Feld | unterste Schichten |
# | **ReLU** | $R_{\text{in}} = R_{\text{out}}$ | Identität | Aktivierungen |
# | **MaxPool** | *winner-takes-all* | die gesamte Relevanz geht an das Maximum im Fenster | Pooling |
#
# Unterschiedliche Regeln in unterschiedlichen Tiefen nennt man **Composite-Strategie**
# (Montavon et al., 2019): oben Relevanz konzentrieren ($\varepsilon$), in der Mitte saubere
# positive Zuordnungen ($\alpha\beta$), unten glätten (flat), weil einzelne Voxel dort fast
# bedeutungslos sind und die reine z-Regel „Salz-und-Pfeffer"-Karten erzeugt.
#
# Die Liste in `LRPStrategy` hat genau **sieben** Einträge, weil das SFCN sieben Schichten mit
# Gewichten hat (6 × `Conv3D` + 1 × `Dense`), und die Reihenfolge ist **Input → Output**.
# Pooling-, ReLU- und BatchNorm-Schichten bekommen keinen Eintrag; sie haben feste Regeln
# (BatchNorm wird beim Bau des Erklärers vorher in die Faltung hineingerechnet).
#
# ### Warum die Maskierung unverzichtbar ist
#
# ```python
# R = R * brain
# ```
#
# Außerhalb des Gehirns ist das Bild exakt 0. Bei der z-Regel wäre dort automatisch
# $R = a \cdot c = 0$. Die **flat-Regel ignoriert aber die Aktivierung** ($a \leftarrow 1$) und
# verteilt Relevanz deshalb auch auf reine Hintergrundvoxel. Ohne Maske hätte man leuchtende
# Bereiche in der Luft neben dem Kopf — und alle Metriken wären durch diesen Unsinn verwässert.
#
# ### Ausgabe dieser Zelle
#
# ```text
# id        Variante   Vorhersage    Summe R   Masse  Vol-Ant   Ratio  Prec@k    Hit
# sub-017   original       22.168      7.524  0.0085   0.0095    0.89  0.0024  False
# sub-017   shuffled       22.168      6.422  0.0481   0.0095    5.04  0.2416   True
# sub-036   original       22.168      7.492  0.0066   0.0104    0.63  0.0019  False
# sub-036   shuffled       22.169      6.681  0.0418   0.0104    4.03  0.1856   True
# ...
#
# Mittelwerte über alle Subjekte:
#           vorhersage    mass   share   ratio  precision  hit
# original     22.1683  0.0088  0.0096  0.9233     0.0065  0.0
# shuffled     22.1687  0.0378  0.0096  3.9970     0.1794  1.0
# ```
#
# ### ⚠️ Interpretation: warum dieses Ergebnis eine Falle ist
#
# Lesen wir die Zeilen einzeln:
#
# **Auf den Originalbildern ist die Ratio im Mittel ≈ 0,9.** Das heißt: die Relevanz ist im
# Thalamus **nicht** angereichert. Das Pointing Game wird praktisch nie bestanden. Für ein
# Brain-Age-Modell wäre dieser Befund nicht einmal überraschend — Altern zeigt sich vor allem an
# Kortexdicke und Ventrikelgröße, nicht am Thalamus.
#
# **Auf den permutierten Bildern springt die Ratio auf ≈ 4 im Mittel**, und das **Pointing Game
# wird in (fast) allen Fällen bestanden**. Nach den Kriterien aus Abschnitt 6 wäre das ein
# glänzender Erfolg: „LRP findet den Thalamus."
#
# **Es ist keiner.** Denn wir wissen aus Abschnitt 7, dass dieses Modell die Bilder gar nicht
# liest — es gibt für ein Nullbild dieselbe Zahl aus, und auch hier ändert sich die Vorhersage
# zwischen Original und permutiert nur um 0,001. Es *kann* den Thalamus nicht gefunden haben.
#
# ### Wo kommt der Effekt dann her?
#
# Aus der Bildstatistik, nicht aus dem Modell. Zur Erinnerung an die Abbildung in Abschnitt 4:
# Nach der Permutation ist der Thalamus der **einzige räumlich zusammenhängende, glatte Bereich**
# im ganzen Bild; alles andere ist Salz-und-Pfeffer-Rauschen. Und LRP ist nicht blind gegenüber der
# Eingabe, selbst wenn das Modell es ist:
#
# 1. Die Regel enthält den Faktor $a_j$, also den **Grauwert**. Wo das Bild groß ist, ist auch die
#    Relevanz groß — unabhängig davon, was das Netz damit macht.
# 2. Die fünf `MaxPool3D`-Schichten verteilen nach *winner-takes-all* zurück. Ein **kohärent
#    heller Block** gewinnt systematisch mehr Pooling-Fenster als zufällig verteilte Einzelwerte.
#    Nach fünf Halbierungen entspricht ein Voxel der tiefsten Ebene einem 32×32×32-Block im Bild —
#    eine glatte Struktur dieser Größenordnung dominiert ihre Umgebung.
#
# Zusammen ergibt das eine Anreicherung im Thalamus, die **allein von der Konstruktion der
# Testdaten** stammt.
#
# ### Die Lehre daraus — der wichtigste Punkt dieses Notebooks
#
# > Der Permutationstest aus Abschnitt 4 ist ein guter Test, aber er ist **nur in einer Richtung
# > aussagekräftig**. Findet man viel Relevanz im permutierten Bereich, ist definitiv etwas faul.
# > Findet man viel Relevanz im erhaltenen Bereich, ist das **kein** Beleg — es kann daran liegen,
# > dass dieser Bereich der einzige mit Struktur ist.
#
# Konkret für die Projektpraxis: Ein Permutationsexperiment braucht **immer** mindestens zwei
# Begleitkontrollen:
#
# 1. **Reagiert das Modell?** Ändert sich die Vorhersage zwischen Original und permutiert
#    überhaupt? (Hier: nicht um 0,001.) Wenn nicht, ist der Test von vornherein sinnlos.
# 2. **Was passiert, wenn eine *andere*, gleich große Struktur erhalten bleibt?** Wenn man statt
#    des Thalamus z. B. den Hippocampus stehen lässt und dieselbe Anreicherung dort auftritt, war
#    der Effekt geometrisch und nicht inhaltlich.
#
# ### Einordnung
#
# Dieser Abschnitt ist ein Musterbeispiel dafür, warum Attributions-Evaluation so schwierig ist:
# Die Metrik zeigt einen Erfolg, das Bild sieht überzeugend aus, und die Schlussfolgerung ist
# trotzdem falsch. Der einzige Schutz sind Kontrollbedingungen, die man **vor** dem Blick auf das
# Ergebnis festlegt.

# %%
from explainability import LRP, LRPStrategy

# 7 Einträge = 7 Schichten mit Gewichten im SFCN (6x Conv3D + 1x Dense), Input -> Output
SFCN_STRATEGY = LRPStrategy(layers=[
    {"flat": True},               # Block 1 (naechste am Bild)
    {"flat": True},               # Block 2
    {"alpha": 2, "beta": 1},      # Block 3
    {"alpha": 2, "beta": 1},      # Block 4
    {"alpha": 2, "beta": 1},      # Block 5
    {"alpha": 2, "beta": 1},      # Conv 1x1x1
    {"epsilon": 0.25},            # Dense(1) (naechste am Output)
])

if HAVE_REAL_DATA:
    with tf.device(BIG_MODEL_DEVICE):
        sfcn_lrp = LRP(sfcn, layer=len(sfcn.layers) - 1, idx=0, strategy=SFCN_STRATEGY)

        rng = np.random.default_rng(RANDOM_SEED)
        records = []
        heatmaps = {}

        print(f"{'id':10s} {'Variante':10s} {'Vorhersage':>11s} {'Summe R':>10s} "
              f"{'Masse':>7s} {'Vol-Ant':>8s} {'Ratio':>7s} {'Prec@k':>7s} {'Hit':>6s}")
        for subject in subjects:
            volume, thalamus, brain, segmentation, image = load_subject(subject)
            variants = {
                "original": volume,
                "shuffled": shuffle_outside(volume, thalamus, rng),
            }
            for variant, data in variants.items():
                # Gleiche Normalisierung wie AsyncNiftiGenerator / Explain_brain_age
                X = np.asarray(preprocessor(data), dtype=np.float32)[None]
                prediction = float(sfcn.predict(X, verbose=0).ravel()[0])
                R = sfcn_lrp.predict(X, verbose=0)[0] * brain
                metrics = relevance_metrics(R, thalamus, brain)
                records.append({"id": subject, "age": age_for_subject(subject),
                                "variante": variant, "vorhersage": prediction,
                                **metrics})
                if subject == reference_subject:
                    heatmaps[variant] = (data, R, thalamus)
                print(f"{subject:10s} {variant:10s} {prediction:11.3f} "
                      f"{metrics['sum_R']:10.3f} {metrics['mass']:7.4f} "
                      f"{metrics['share']:8.4f} {metrics['ratio']:7.2f} "
                      f"{metrics['precision']:7.4f} {str(metrics['hit']):>6s}")

    sfcn_results = pd.DataFrame(records)
    sfcn_results.to_csv(target_dir / "04_sfcn_lrp_metriken.csv", index=False)

    print("\nMittelwerte über alle Subjekte:")
    print(sfcn_results.groupby("variante")[["vorhersage", "mass", "share", "ratio",
                                            "precision", "hit"]].mean().round(4).to_string())

    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    for col, variant in enumerate(["original", "shuffled"]):
        data, R, thalamus = heatmaps[variant]
        scaled = R / np.abs(R).max()
        ax[0][col].imshow(data[:, :, centre[2]], cmap="Greys_r")
        ax[0][col].set_title(f"{variant} — Bild")
        ax[1][col].imshow(scaled[:, :, centre[2]], cmap="seismic", clim=(-1, 1))
        # Thalamusgrenze als Kontur einzeichnen: so sieht man Treffer und Fehltreffer
        ax[1][col].contour(thalamus[:, :, centre[2]].astype(float), levels=[0.5],
                           colors="lime", linewidths=1.2)
        ax[1][col].set_title(f"{variant} — LRP (grün = Thalamusgrenze)")
        for row in range(2):
            ax[row][col].axis("off")
    fig.suptitle("Vortrainiertes SFCN: LRP auf Original und permutiertem Bild\n"
                 "Achtung — das Modell ignoriert seine Eingabe (Abschnitt 7)", fontsize=13)
    fig.savefig(target_dir / "04_sfcn_lrp_heatmaps.png", bbox_inches="tight", dpi=150)
    plt.show()

# %% [markdown]
# <a id="sec-09"></a>
# ## 9. Ein eigenes Modell trainieren, das nur den Thalamus messen kann
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Warum jetzt selbst trainieren?
#
# Abschnitt 7 hat gezeigt, dass das mitgelieferte Modell für unsere Frage nicht taugt. Wir
# brauchen ein Modell mit zwei Eigenschaften:
#
# 1. Es sagt das **Thalamusvolumen** vorher (nicht das Alter).
# 2. Wir können **belegen**, dass es dafür den Thalamus benutzen muss — weil im Datensatz nichts
#    anderes die Antwort verrät (Abschnitt 5 hat das mit Korrelationen geprüft).
#
# Erst dann ist die Ground Truth der Erklärung eindeutig, und erst dann ist ein Urteil über LRP
# möglich.
#
# ### Aufteilung in Trainings-, Validierungs- und Testdaten
#
# Die 600 Phantome werden in drei Teile geschnitten: 420 / 90 / 90.
#
# | Menge | Anteil | Wofür |
# |---|---|---|
# | **Training** | 70 % | daraus werden die Gewichte gelernt |
# | **Validierung** | 15 % | Kontrolle *während* des Trainings (Lernrate, Abbruch) |
# | **Test** | 15 % | einmalige Endbewertung, nie beim Training angeschaut |
#
# Warum drei und nicht zwei? Weil man die Validierungsmenge indirekt „mitlernt": Sie steuert, wann
# abgebrochen wird und welche Gewichte behalten werden. Damit ist sie nicht mehr unabhängig. Die
# Testmenge bleibt es. Da wir die Daten selbst generieren, sind die drei Mengen garantiert
# disjunkt und identisch verteilt — bei echten Patientendaten wäre hier viel mehr Vorsicht nötig
# (kein Subjekt in zwei Mengen, Scanner gleichmäßig verteilt).
#
# ### Die Architektur
#
# Wir nehmen bewusst **denselben Bauplan wie das SFCN**, nur eine Stufe kleiner, damit er zum
# $64^3$-Gitter passt:
#
# | Block | Schichten | Ausgabe |
# |---|---|---|
# | Eingang | `Input(64,64,64)` → `Reshape(...,1)` | 64³ × 1 |
# | 1 | `Conv3D(32, 3³)` → BatchNorm → ReLU → `MaxPool3D(2)` | 32³ × 32 |
# | 2 | `Conv3D(64, 3³)` → … | 16³ × 64 |
# | 3 | `Conv3D(128, 3³)` → … | 8³ × 128 |
# | 4 | `Conv3D(256, 3³)` → … | 4³ × 256 |
# | Top | `Conv3D(64, 1³)` → BatchNorm → ReLU | 4³ × 64 |
# | | **`GlobalAveragePooling3D`** | **64** |
# | Kopf | `Dense(1)` | 1 |
#
# **Die Bausteine im Einzelnen:**
#
# * **`Conv3D(n, 3×3×3)`** — $n$ Filter fahren über das Volumen; jeder reagiert auf ein lokales
#   3×3×3-Muster. Frühe Schichten lernen Kanten und Grauwertstufen, spätere zusammengesetzte
#   Formen. `padding='SAME'` erhält die Kantenlänge.
# * **`BatchNormalization`** — normiert die Ausgaben einer Schicht auf Mittelwert 0 und Varianz 1
#   (über den Batch) und skaliert sie dann mit gelernten Parametern. Stabilisiert das Training
#   erheblich. Für LRP ist wichtig: BatchNorm ist zur Inferenzzeit eine **affine Funktion** und
#   wird beim Bau des Erklärers in die vorangehende Faltung hineingerechnet.
# * **`ReLU`**, $\max(0, x)$ — die Nichtlinearität. Ohne sie wäre das ganze Netz eine einzige
#   lineare Abbildung.
# * **`MaxPool3D(2)`** — halbiert jede Kantenlänge, behält pro 2×2×2-Fenster das Maximum. Spart
#   Rechenzeit und macht das Netz robuster gegen kleine Verschiebungen.
# * **`GlobalAveragePooling3D`** — mittelt jede der 64 Merkmalskarten zu **einer Zahl**.
#
# ### Warum GAP für diese Aufgabe besonders passend ist
#
# Ein kurzer Gedanke, der erklärt, warum die Aufgabe überhaupt lösbar ist. Nehmen wir an, ein
# Kanal $c$ hat gelernt, genau bei Thalamus-Grauwerten zu feuern, also $a_c(v) \approx
# \mathbb{1}[v \in \text{Thalamus}]$. Dann ist
#
# $$\text{GAP}_c \;=\; \frac{1}{|V|}\sum_{v} a_c(v) \;\approx\; \frac{|\text{Thalamus}|}{|V|},$$
#
# also **direkt proportional zum gesuchten Volumen**. Eine einzige `Dense`-Schicht muss das nur
# noch mit $|V|/100$ multiplizieren. Volumenmessung ist für diese Architektur die natürliche
# Aufgabe — mit einem Detektor plus GAP ist sie exakt lösbar.
#
# Ein Wermutstropfen: Die vier `MaxPool3D`-Schichten stören das ein bisschen, weil das Maximum
# eines Fensters nicht dessen Summe ist. Nach vier Halbierungen entspricht ein Voxel der obersten
# Ebene einem 16³-Block des Bildes; innerhalb eines solchen Blocks kann das Netz nicht mehr
# zählen, sondern nur noch „vorhanden / nicht vorhanden" unterscheiden. Die Volumenschätzung wird
# dadurch **körnig** — und das wird in Abschnitt 13 im Distanzprofil sichtbar.
#
# ### Training
#
# Verlustfunktion **MSE** (mittlerer quadratischer Fehler), zusätzlich als lesbare Kennzahl der
# **MAE** (mittlerer absoluter Fehler):
#
# $$\mathcal{L} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2,
#   \qquad \mathrm{MAE} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|$$
#
# Optimierer **Adam** mit Lernrate $10^{-3}$, Batchgröße 16, bis zu 60 Epochen. Zwei Callbacks:
#
# * **`ReduceLROnPlateau`** — stagniert der Validierungsverlust 8 Epochen, wird die Lernrate mit
#   0,3 multipliziert. Typisches Bild: die Lernkurve macht bei jeder Reduktion einen Sprung nach
#   unten.
# * **`EarlyStopping(restore_best_weights=True)`** — bricht nach 20 erfolglosen Epochen ab und
#   setzt auf die besten Gewichte zurück. Standardschutz gegen Overfitting.
#
# ### Trainieren oder laden? Der Caching-Zweig
#
# Die Zelle trainiert nicht unbedingt. Liegt im Verzeichnis `<epochs>_epochs/` schon eine
# `.keras`-Datei, wird sie geladen und `history` bleibt `None` (dann gibt es keine Lernkurve). Das
# spart bei Wiederholungsläufen Minuten, hat aber eine Stolperfalle: **Ändert man die
# Architektur, wird trotzdem das alte Modell geladen.** Wer neu trainieren will, muss die Datei
# löschen.
#
# ### Ausgabe dieser Zelle
#
# Beim **ersten** Durchlauf trainiert diese Zelle und braucht dafür ca. **3–4 Minuten** auf einer
# GPU (60 Epochen à ~3 s plus Aufwärmzeit); auf der CPU entsprechend länger. Der Trainingsverlust
# fällt dabei von MSE ≈ 184 auf ≈ 1,7, der Validierungsverlust auf ≈ 0,8. Zum Vergleich: die
# Baseline aus Abschnitt 5 („immer den Mittelwert vorhersagen") liegt bei MSE ≈ 65, MAE ≈ 6,9.
#
# In der hier gespeicherten Ausgabe steht dagegen `Modell geladen von: …` — das Modell lag schon im
# Cache, es wurde also nicht neu trainiert, und entsprechend gibt es keine Lernkurven zu zeichnen.
# Die Abbildung unten stammt aus dem Trainingslauf und liegt als `05_lernkurven.png` im
# Ausgabeordner.
#
# ### Was die Abbildung zeigt
#
# Trainings- und Validierungsverlust pro Epoche, dazu die Baseline als waagerechte Linie
# (logarithmische y-Achse).
#
# Der **Trainingsverlust** (blau) fällt glatt und monoton und unterschreitet die Baseline schon nach
# 5 Epochen — das Netz lernt die Aufgabe wirklich.
#
# Der **Validierungsverlust** (orange) macht dagegen etwas, was auf den ersten Blick nach einem
# Fehler aussieht: Er steigt zwischen Epoche 4 und 9 auf über 2 000, also **das Dreißigfache der
# Baseline**, und braucht bis Epoche 17, um wieder darunter zu kommen. Das ist die typische
# Signatur von **BatchNorm bei kleinen Batches**: Im Training werden die Mittelwerte und Varianzen
# des aktuellen Batches benutzt, in der Validierung dagegen laufende Schätzungen, die erst über
# viele Schritte einlaufen. Solange beide auseinanderliegen, rechnet das Modell in der Validierung
# praktisch mit falschen Normierungskonstanten.
#
# **Praktische Konsequenz:** Man darf in den ersten Epochen nicht in Panik verfallen und die
# Architektur ändern — und man darf `EarlyStopping` nicht mit kleiner `patience` konfigurieren, weil
# es sonst mitten in dieser Phase abbrechen würde. Ab Epoche 35 liegt die Validierungskurve stabil
# **unter** der Trainingskurve (0,8 gegen 1,7); das ist bei aktivem BatchNorm normal und kein
# Widerspruch.

# %%
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, BatchNormalization, Conv3D, Dense, \
                                    GlobalAveragePooling3D, Input, MaxPooling3D, Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

EPOCHS = 60
BATCH_SIZE = 16
DEPTHS = [32, 64, 128, 256]

n_train, n_val = 420, 90
train_slice = slice(0, n_train)
val_slice = slice(n_train, n_train + n_val)
test_slice = slice(n_train + n_val, None)

train_X, train_y = phantom_X[train_slice], phantom_y[train_slice]
val_X, val_y = phantom_X[val_slice], phantom_y[val_slice]
test_X, test_y = phantom_X[test_slice], phantom_y[test_slice]
test_L = phantom_L[test_slice]

assert len(train_X) + len(val_X) + len(test_X) == N_PHANTOMS
print(f"Training {len(train_X)} | Validierung {len(val_X)} | Test {len(test_X)}")

MODEL_DIR = target_dir / f"{EPOCHS}_epochs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "phantom_thalamus_cnn.keras"

with tf.device(SMALL_MODEL_DEVICE):
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    inputs = Input((PHANTOM_SIZE,) * 3, name="phantom_input")
    x = Reshape((PHANTOM_SIZE,) * 3 + (1,), name="expand_dims")(inputs)

    for i, depth in enumerate(DEPTHS):
        x = Conv3D(depth, (3, 3, 3), padding="SAME", activation=None,
                   name=f"block{i + 1}_conv")(x)
        x = BatchNormalization(name=f"block{i + 1}_norm")(x)
        x = Activation("relu", name=f"block{i + 1}_relu")(x)
        x = MaxPooling3D((2, 2, 2), name=f"block{i + 1}_pool")(x)

    x = Conv3D(64, (1, 1, 1), padding="SAME", activation=None, name="top_conv")(x)
    x = BatchNormalization(name="top_norm")(x)
    x = Activation("relu", name="top_relu")(x)
    x = GlobalAveragePooling3D(name="top_pool")(x)
    x = Dense(1, activation=None, name="predictions")(x)

    phantom_model = KerasModel(inputs, x, name="phantom_thalamus_cnn")
    phantom_model.summary()

    existing = next(iter(sorted(MODEL_DIR.glob("*.keras"))), None)
    if existing is not None:
        phantom_model = load_model(existing)
        history = None
        print(f"\nModell geladen von: {existing}")
    else:
        phantom_model.compile(loss="mse", optimizer=Adam(1e-3), metrics=["mae"])
        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=8, min_lr=1e-5),
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        ]
        history = phantom_model.fit(train_X, train_y,
                                    validation_data=(val_X, val_y),
                                    batch_size=BATCH_SIZE, epochs=EPOCHS,
                                    callbacks=callbacks, verbose=2)
        phantom_model.save(MODEL_PATH)
        print(f"\nModell gespeichert unter: {MODEL_PATH}")

if history is None:
    print("Keine Lernkurven: das Modell wurde geladen, nicht trainiert.")
else:
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs_run = np.arange(1, len(history.history["loss"]) + 1)
    ax.plot(epochs_run, history.history["loss"], label="Training (MSE)")
    ax.plot(epochs_run, history.history["val_loss"], label="Validierung (MSE)")
    ax.axhline(phantom_y.var(), color="grey", linestyle="--",
               label=f"Baseline: Mittelwert vorhersagen (MSE = {phantom_y.var():.1f})")
    ax.set_yscale("log")
    ax.set_xlabel("Epoche")
    ax.set_ylabel("MSE (log-Skala)")
    ax.set_title("Lernkurven des Phantom-Modells")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig(target_dir / "05_lernkurven.png", bbox_inches="tight", dpi=150)
    plt.show()

# %% [markdown]
# <a id="sec-10"></a>
# ## 10. Wie gut sagt das Phantom-Modell vorher?
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Warum dieser Zwischenschritt entscheidend ist
#
# Bevor wir eine einzige Heatmap anschauen, muss diese Frage beantwortet sein. **Die Erklärung
# eines Modells, das die Aufgabe nicht gelöst hat, erklärt nichts über die Daten — nur etwas über
# das schlechte Modell.** Genau daran ist der Versuch mit dem vortrainierten SFCN gescheitert.
#
# Drei Kennzahlen auf der **Testmenge** (die das Modell noch nie gesehen hat):
#
# $$\mathrm{MAE} = \frac{1}{n}\sum_i |y_i - \hat y_i|,
#   \qquad \rho = \operatorname{corr}(y, \hat y),
#   \qquad R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}$$
#
# $R^2$ ist am direktesten zu lesen: 0 bedeutet „so gut wie der Mittelwert", 1 bedeutet perfekt.
# Es wird negativ, wenn ein Modell *schlechter* ist als der Mittelwert.
#
# ### Ausgabe dieser Zelle
#
# ```text
# Test-MAE    : 0.53   (Baseline 6.87  →  13x besser)
# Test-R²     : 0.992
# Korrelation : 0.996
# Train-MAE   : 0.46   (deutlich kleiner als Test-MAE wäre ein Overfitting-Signal)
#
# Thalamusvolumen in der Testmenge: 248 .. 2896 Voxel
# ```
#
# ### Interpretation
#
# Das Modell schätzt das Thalamusvolumen mit einem mittleren Fehler von **≈ 53 Voxeln** (der
# Zielwert ist in Einheiten von 100 Voxeln angegeben) — bei Volumina zwischen 248 und 2 896
# Voxeln. Das sind wenige Prozent Fehler und **etwa 13-mal besser als die Baseline**.
#
# Dass Train-MAE (0,46) und Test-MAE (0,53) fast gleich sind, ist das eigentlich beruhigende
# Signal: Es gibt **kein Overfitting**. Wäre der Trainingsfehler um ein Vielfaches kleiner, hätte
# das Modell die Trainingsbilder auswendig gelernt statt eine Regel — und die Erklärung würde
# diese Auswendiglernerei erklären, nicht die Volumenmessung.
#
# Damit ist die Voraussetzung für alles Weitere erfüllt: Das Modell hat die Aufgabe gelöst, und da
# der Datensatz konfounderfrei ist (Abschnitt 5), **muss** es dafür den Thalamus vermessen. Es gibt
# keinen anderen Weg zur richtigen Antwort. Jetzt — und nur jetzt — ist die Frage sinnvoll, ob LRP
# das auch anzeigt.
#
# ### Was die Abbildung zeigt
#
# Links das Streudiagramm vorhergesagt gegen tatsächlich, mit der Diagonale $\hat y = y$ als
# Idealfall. Die Punkte liegen so eng an der Diagonale, dass sie kaum davon zu unterscheiden sind
# — über den gesamten Wertebereich, nicht nur in der Mitte.
#
# Rechts die **Residuen** $\hat y - y$ gegen $y$. Dieses Diagramm ist das wichtigere von beiden,
# weil ein Streudiagramm mit hoher Korrelation systematische Fehler verstecken kann. Zu prüfen ist:
# Streuen die Punkte gleichmäßig um die Nulllinie (gut), oder gibt es einen Trend?
#
# Hier sind zwei milde Effekte zu erkennen, beide typisch für Regressionsmodelle:
#
# * **Regression zur Mitte.** Unterhalb von $y \approx 4$ liegen die Residuen überwiegend über
#   Null (kleine Volumina werden leicht überschätzt), und die größten negativen Residuen (−1,5 bis
#   −2) treten alle bei $y > 20$ auf (große Volumina werden unterschätzt). Die Punktwolke kippt also
#   schwach von links oben nach rechts unten. Ein Modell minimiert den mittleren Fehler und zieht
#   deshalb die Extreme zur Mitte.
# * **Zunehmende Streuung.** Die Residuen sind bei großen Volumina breiter gestreut als bei kleinen.
#   Das ist plausibel, weil der Fehler hier absolut gemessen wird: 5 % Fehler auf 2 900 Voxel sind
#   mehr Voxel als 5 % auf 250.
#
# Beide Effekte sind so klein, dass sie die Interpretation nicht gefährden — aber es ist gut zu
# wissen, dass sie da sind, denn auf echten Daten (Brain Age!) sind sie oft deutlich stärker und
# werden dann gelegentlich als biologischer Befund fehlinterpretiert.

# %%
with tf.device(SMALL_MODEL_DEVICE):
    train_pred = phantom_model.predict(train_X, verbose=0).ravel()
    test_pred = phantom_model.predict(test_X, verbose=0).ravel()

test_mae = float(np.abs(test_pred - test_y).mean())
test_baseline_mae = float(np.abs(test_y - train_y.mean()).mean())
test_r2 = 1 - ((test_y - test_pred) ** 2).sum() / ((test_y - test_y.mean()) ** 2).sum()

print(f"Test-MAE    : {test_mae:.2f}   (Baseline {test_baseline_mae:.2f}  →  "
      f"{test_baseline_mae / test_mae:.0f}x besser)")
print(f"Test-R²     : {test_r2:.3f}")
print(f"Korrelation : {np.corrcoef(test_pred, test_y)[0, 1]:.3f}")
print(f"Train-MAE   : {np.abs(train_pred - train_y).mean():.2f}   "
      f"(deutlich kleiner als Test-MAE wäre ein Overfitting-Signal)")
print(f"\nThalamusvolumen in der Testmenge: {test_y.min() * 100:.0f} .. "
      f"{test_y.max() * 100:.0f} Voxel")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
limits = [phantom_y.min() - 1, phantom_y.max() + 1]
ax[0].scatter(test_y, test_pred, alpha=0.7)
ax[0].plot(limits, limits, "k--", linewidth=1, label="ideal ($\\hat y = y$)")
ax[0].set_xlabel("tatsächliches Thalamusvolumen / 100 Voxel")
ax[0].set_ylabel("vorhergesagt")
ax[0].set_title(f"Testmenge: MAE = {test_mae:.2f}, R² = {test_r2:.3f}")
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].scatter(test_y, test_pred - test_y, alpha=0.7, color="darkorange")
ax[1].axhline(0, color="k", linestyle="--", linewidth=1)
ax[1].set_xlabel("tatsächliches Thalamusvolumen / 100 Voxel")
ax[1].set_ylabel("Residuum $\\hat y - y$")
ax[1].set_title("Residuen: gleichmäßig um 0 = kein systematischer Fehler")
ax[1].grid(alpha=0.3)
fig.savefig(target_dir / "06_vorhersageguete.png", bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# <a id="sec-11"></a>
# ## 11. LRP auf dem Phantom-Modell: die Heatmaps
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Jetzt kommt der eigentliche Test. Wir bauen den LRP-Erklärer für das Phantom-Modell und schauen
# uns die Relevanzkarten von vier Testphantomen mit unterschiedlichem Thalamusvolumen an.
#
# Die Strategie hat **sechs** Einträge, weil dieses Modell sechs Schichten mit Gewichten hat
# (5 × `Conv3D` inklusive der 1×1×1-Schicht + 1 × `Dense`):
#
# ```python
# LRPStrategy(layers=[
#     {'flat': True},            # Block 1  (naechste am Bild)
#     {'alpha': 2, 'beta': 1},   # Block 2
#     {'alpha': 2, 'beta': 1},   # Block 3
#     {'alpha': 2, 'beta': 1},   # Block 4
#     {'alpha': 2, 'beta': 1},   # Conv 1x1x1
#     {'epsilon': 0.25},         # Dense(1) (naechste am Output)
# ])
# ```
#
# Zwei Punkte zur Umsetzung:
#
# * Die Karte wird mit `R * brain` maskiert, damit die flat-Regel keine Relevanz im Hintergrund
#   stehen lässt (Begründung in Abschnitt 8).
# * Für die Darstellung wird auf $[-1, 1]$ normiert (Division durch $\max|R|$), damit die
#   Farbskala `seismic` symmetrisch um Null liegt: **rot = positiv, weiß = 0, blau = negativ**.
#   „Positiv" heißt hier: dieses Voxel hat den vorhergesagten Volumenwert nach **oben** getrieben.
#
# ### Die Erhaltungseigenschaft als Selbsttest
#
# LRP soll die Vorhersage vollständig auf die Voxel verteilen:
#
# $$\sum_v R_v \;\approx\; \hat y$$
#
# Die Zelle gibt beides aus. Große Abweichungen sind normal und haben zwei bekannte Ursachen:
# Erstens verschluckt die **Bias**-Behandlung Relevanz (ein Bias ist eine Konstante, die keinem
# Voxel zugeordnet werden kann, und ihr Anteil wird verworfen). Zweitens ist die **flat-Regel
# nicht streng erhaltend**, weil sie $a \leftarrow 1$ setzt; sie kann die Summe auch erhöhen.
# Ergänzend frisst die Maskierung noch einmal etwas weg. Die Summe ist deshalb kein Korrektheits-
# beweis, aber ein nützlicher Alarm: wird sie 0 oder gigantisch, ist etwas grundsätzlich falsch.
#
# ### Ausgabe dieser Zelle
#
# ```text
#       y  Vorhersage    Summe R  Verhältnis
#    2.48        3.01      1.080        0.36
#    6.03        5.63      2.808        0.50
#   14.65       14.06      8.871        0.63
#   28.96       26.95     17.528        0.65
# ```
#
# Die Summe der Relevanz erreicht also nur 36 % bis 65 % der Vorhersage — die Erhaltung ist wie
# erwartet verletzt, aber die Größenordnung stimmt, und die Summe **wächst mit dem Zielwert**. Das
# ist an sich schon eine erste gute Nachricht: Die Karten skalieren mit dem, was das Modell
# ausgibt.
#
# ### Was die Abbildung zeigt
#
# Vier Spalten (Phantome mit steigendem Thalamusvolumen), drei Zeilen: Bild, Relevanzkarte mit
# eingezeichneter Thalamusgrenze, und die Relevanz **nur innerhalb** der Maske. Im Bild oben sind
# die beiden grauen Blobs links der Thalamus (grün umrandet), die beiden schwarzen Blobs rechts die
# Ventrikel.
#
# Drei Dinge sind zu erkennen — und nur das erste davon ist erwünscht:
#
# 1. **Am Thalamus liegt ein roter Ring, sein Inneres ist blass bis bläulich.** In der unteren Zeile
#    ist das besonders klar: die Relevanz sitzt als Kranz auf der Grenze, in der Mitte ist sie null
#    oder leicht negativ. Das Modell liest also die **Kontur**, nicht die Füllung — für eine
#    Größenmessung genau die richtige Strategie (mehr dazu in Abschnitt 13).
# 2. **Die Ventrikel leuchten mindestens genauso stark.** In den ersten beiden Spalten sind sie
#    sogar die kräftigsten Stellen der ganzen Karte, obwohl sie zum Zielwert **nichts** beitragen.
#    Das ist der falsch positive Befund, den Abschnitt 12 mit 4,17 bemisst — und genau die Art von
#    Fehler, die man auf echten Daten für einen interessanten Befund halten würde.
# 3. **Am Kopfrand liegt ein schwacher Saum**, ebenfalls ohne inhaltliche Bedeutung. Hier verteilt
#    die flat-Regel in der untersten Schicht Relevanz über die rezeptiven Felder der stärksten
#    Kante im Bild, nämlich des Übergangs Gewebe/Hintergrund.
#
# Auffällig ist außerdem, dass die Ventrikel bei **kleinem** Thalamus relativ am stärksten
# hervortreten (Spalte 1) und bei großem Thalamus zurückfallen (Spalte 4). Der visuelle Eindruck ist
# damit „teilweiser Erfolg mit deutlichen Nebenbefunden". Wie viel „teilweise" genau ist, sagt erst
# der nächste Abschnitt — und das ist der Grund, warum wir die Metriken vorher definiert haben.

# %%
PHANTOM_STRATEGY_LAYERS = [
    {"flat": True},
    {"alpha": 2, "beta": 1},
    {"alpha": 2, "beta": 1},
    {"alpha": 2, "beta": 1},
    {"alpha": 2, "beta": 1},
    {"epsilon": 0.25},
]


def explain_phantom(model, strategy_layers, X, device=None):
    """Relevanzkarten fuer alle Bilder in X (einzeln, um Speicher zu sparen)."""
    device = device or SMALL_MODEL_DEVICE
    with tf.device(device):
        explainer = LRP(model, layer=len(model.layers) - 1, idx=0,
                        strategy=LRPStrategy(layers=strategy_layers))
        return np.stack([explainer.predict(X[i:i + 1], verbose=0)[0]
                         for i in range(len(X))])


examples = np.argsort(test_y)[np.linspace(0, len(test_y) - 1, 4).astype(int)]
example_R = explain_phantom(phantom_model, PHANTOM_STRATEGY_LAYERS, test_X[examples])

print(f"{'y':>7s} {'Vorhersage':>11s} {'Summe R':>10s} {'Verhältnis':>11s}")
for k, index in enumerate(examples):
    brain = test_L[index] != LABEL_BACKGROUND
    total = float((example_R[k] * brain).sum())
    print(f"{test_y[index]:7.2f} {test_pred[index]:11.2f} {total:10.3f} "
          f"{total / test_pred[index]:11.2f}")

fig, ax = plt.subplots(3, 4, figsize=(15, 11))
for col, index in enumerate(examples):
    label = test_L[index]
    brain = label != LABEL_BACKGROUND
    thalamus = label == LABEL_THALAMUS
    R = example_R[col] * brain
    R = R / np.abs(R).max()
    z = int(np.argwhere(thalamus)[:, 2].mean())

    ax[0][col].imshow(test_X[index][:, :, z], cmap="Greys_r", vmin=0, vmax=1)
    ax[0][col].set_title(f"y = {test_y[index]:.1f}, Vorhersage = {test_pred[index]:.1f}")

    ax[1][col].imshow(R[:, :, z], cmap="seismic", clim=(-1, 1))
    ax[1][col].contour(thalamus[:, :, z].astype(float), levels=[0.5],
                       colors="lime", linewidths=1.2)
    ax[1][col].set_title("LRP (grün = Thalamusgrenze)")

    ax[2][col].imshow(np.where(thalamus[:, :, z], R[:, :, z], np.nan),
                      cmap="seismic", clim=(-1, 1))
    ax[2][col].set_title("Relevanz nur im Thalamus")

    for row in range(3):
        ax[row][col].axis("off")
fig.suptitle("Phantom-Modell: LRP-Relevanz bei steigendem Thalamusvolumen", fontsize=14)
fig.savefig(target_dir / "07_phantom_lrp_heatmaps.png", bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# <a id="sec-12"></a>
# ## 12. Quantitative Auswertung: Relevanz je Kompartiment
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Jetzt wird gezählt statt geschaut. Für 12 Testphantome wird die Relevanz **je Kompartiment**
# aufsummiert und mit dem jeweiligen Volumenanteil verglichen — genau die Rechnung, die man auf
# echten Daten mit dem FreeSurfer-Atlas macht (`Explain_brain_age_predictions`, Abschnitt 11), nur
# mit vier statt 95 Regionen und mit bekannter Ground Truth.
#
# Das ist gleichzeitig die dritte Kontrollbedingung aus Abschnitt 6: Kortex, weiße Substanz und
# Ventrikel tragen **nichts** zum Zielwert bei. Ihre Ratio sollte also bei 1 liegen. Nur der
# Thalamus sollte deutlich darüber liegen.
#
# ### Ausgabe dieser Zelle
#
# ```text
# Kompartiment      R-Anteil  Vol-Anteil   Ratio
# Kortex               0.295      0.387     0.793
# weisse Substanz      0.569      0.584     0.988
# Ventrikel            0.044      0.011     4.173
# Thalamus             0.092      0.018     5.217
#
# Pointing Game   : 0.08
# Top-k-Präzision : 0.266   (Zufallswert 0.018)
# Relevanzmasse   : 0.092   (Volumenanteil 0.018)
# Ratio Thalamus  : 5.22   (theoretisches Maximum 54.3)
# ```
#
# ### Interpretation — Zeile für Zeile
#
# **Thalamus: Ratio 5,22.** Das ist das Hauptergebnis. Die Relevanzdichte im Thalamus ist
# **5,2-mal höher** als im Hirndurchschnitt. LRP zeigt also klar überzufällig auf die richtige
# Struktur. Gleichzeitig: Der theoretische Bestwert wäre $1/s = 1/0{,}018 \approx 54$. Wir sind bei
# 5,2 — also bei etwa **10 % des Erreichbaren**.
#
# **Weiße Substanz: Ratio 0,99.** Perfekt neutral — sie sammelt genau den Anteil ein, der ihrem
# Volumen entspricht. Da sie 58 % des Hirnvolumens ausmacht, bedeutet das aber in absoluten
# Zahlen: **57 % der Relevanz liegt in einem Gewebe, das nichts zur Antwort beiträgt.** Das ist
# die andere Hälfte der Wahrheit über die 5,2.
#
# **Kortex: Ratio 0,79.** Hier wird Relevanz leicht *gemieden*. Das ist ein gutes Zeichen: die
# äußere Schale ist am weitesten vom Thalamus entfernt und hat mit ihm nichts zu tun.
#
# **Ventrikel: Ratio 4,17.** Der problematische Wert. Die Ventrikel sind fast schwarz
# (Grauwert 0,06), variieren unabhängig vom Zielwert und tragen **nichts** zur Antwort bei —
# trotzdem sind sie fast so stark angereichert wie der Thalamus selbst. Zwei plausible Ursachen:
# Erstens ist der Ventrikel eine kleine, kontrastreiche Struktur, und die flat-Regel in der
# untersten Schicht schmiert Relevanz über das gesamte rezeptive Feld ihrer **Ränder**. Zweitens
# ist er die zweite auffällige „Nicht-weiße-Substanz"-Struktur im Bild; ein Kanal, der auf
# Grauwert-Sprünge reagiert, feuert dort ebenfalls.
#
# Das ist ein **falsch positiver Befund**, und zwar der lehrreichste Wert in dieser Tabelle: Auf
# echten Daten hätte man ihn als „interessanten Ventrikeleffekt" berichtet und neuroanatomisch
# gedeutet — ohne die Ground Truth wäre nicht erkennbar gewesen, dass er ein Artefakt ist. Nur weil
# wir die Daten selbst gebaut haben, wissen wir es besser. **Kleine, kontrastreiche Strukturen
# ziehen Relevanz an, unabhängig davon, ob sie etwas beitragen.**
#
# **Top-$k$-Präzision: 0,266.** Von den $k = |T|$ höchstbewerteten Voxeln liegen 27 % im Thalamus,
# 15-mal mehr als bei Zufall. Dass dieser Wert *höher* ist als die Relevanzmasse (0,092), ist
# aufschlussreich: die **stärksten** Voxel liegen überwiegend richtig; die breit verstreute
# schwache Relevanz zieht die Masse-Metrik nach unten. Rangbasierte und massebasierte Maße messen
# also wirklich Verschiedenes, und man sollte beide berichten.
#
# **Pointing Game: 0,08 — der Test fällt praktisch durch.** In nur einem von zwölf Fällen liegt das
# *allerstärkste* Voxel im Thalamus. Das steht in scheinbarem Widerspruch zu allen anderen Zahlen,
# und die Auflösung liefert Abschnitt 13: Das Maximum liegt typischerweise **knapp außerhalb** der
# Struktur, an ihrem Rand. Das Pointing Game hängt an einem einzigen Voxel und an einer
# messerscharfen Maskengrenze — es ist damit die **instabilste** der vier Metriken. Man sollte es
# nie allein berichten.
#
# ### Die Gesamtaussage
#
# > **LRP lokalisiert den Thalamus überzufällig gut, aber nicht präzise.** Bei einer Aufgabe, deren
# > Antwort zu 100 % in einer Struktur steckt, die 1,8 % des Volumens einnimmt, landen 9 % der
# > Relevanz dort — und 57 % in Gewebe, das nichts beiträgt. Eine unbeteiligte Nachbarstruktur
# > (der Ventrikel) wird fast genauso stark markiert.
#
# Für die Projektfrage „*kann ich mit LRP das Thalamusvolumen bestimmen?*" heißt das: Als
# **Lokalisationshinweis** taugt die Karte — man würde den Thalamus in einer Rangliste der
# Regionen ganz oben finden. Als **Segmentierung oder Volumenmessung** taugt sie nicht. Wer
# Thalamusvolumina messen will, nimmt weiterhin FreeSurfer/FastSurfer. Was LRP beantwortet, ist
# eine andere Frage: *benutzt mein Modell die Struktur, von der ich glaube, dass es sie benutzt?*
#
# ### Was die Abbildung zeigt
#
# Links ein Balkendiagramm der Ratio je Kompartiment mit der Referenzlinie bei 1 („keine
# Anreicherung"). Der Thalamus (rot) ragt am weitesten heraus — aber der Ventrikelbalken reicht fast
# ebenso weit, und genau das ist die schlechte Nachricht in diesem Bild. Kortex und weiße Substanz
# liegen bei bzw. leicht unter 1.
#
# Rechts die Verteilung der Thalamus-Ratio über die 12 Phantome — sie zeigt, wie **stabil** der
# Effekt von Bild zu Bild ist. Alle 12 Werte liegen über 1 (Spanne 1,5 bis 8,3), der Mittelwert wird
# also nicht von einzelnen Ausreißern getragen. Der niedrigste Wert (1,5) gehört zum Phantom mit dem
# **kleinsten** Thalamus ($y = 2{,}9$, also 291 Voxel) — dort ist auch die Top-$k$-Präzision 0. Je
# kleiner die Struktur, desto schwerer fällt LRP die Lokalisation; für klinisch interessante kleine
# Regionen ist das eine relevante Einschränkung.

# %%
N_EVAL = 12
eval_index = np.arange(min(N_EVAL, len(test_X)))
eval_R = explain_phantom(phantom_model, PHANTOM_STRATEGY_LAYERS, test_X[eval_index])

per_case, compartment_frames = [], []
for k, index in enumerate(eval_index):
    label = test_L[index]
    brain = label != LABEL_BACKGROUND
    thalamus = label == LABEL_THALAMUS
    R = eval_R[k] * brain
    distance = distance_to(thalamus)
    per_case.append({"index": int(index), "y": float(test_y[index]),
                     **relevance_metrics(R, thalamus, brain, distance)})
    compartment_frames.append(compartment_table(R, label, brain))

phantom_results = pd.DataFrame(per_case)
phantom_results.to_csv(target_dir / "08_phantom_lrp_metriken.csv", index=False)

compartments = (pd.concat(compartment_frames)
                .groupby("Kompartiment", sort=False)[["R-Anteil", "Vol-Anteil", "Ratio"]]
                .mean())
compartments.to_csv(target_dir / "08_phantom_relevanz_je_kompartiment.csv")

print(f"Gemittelt über {len(eval_index)} Testphantome\n")
print(compartments.round(3).to_string())
print()
print(f"Pointing Game   : {phantom_results['hit'].mean():.2f}")
print(f"Top-k-Präzision : {phantom_results['precision'].mean():.3f}   "
      f"(Zufallswert {phantom_results['share'].mean():.3f})")
print(f"Relevanzmasse   : {phantom_results['mass'].mean():.3f}   "
      f"(Volumenanteil {phantom_results['share'].mean():.3f})")
print(f"Ratio Thalamus  : {phantom_results['ratio'].mean():.2f}   "
      f"(theoretisches Maximum {1 / phantom_results['share'].mean():.1f})")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
colours = ["tab:red" if name == "Thalamus" else "tab:blue" for name in compartments.index]
ax[0].barh(compartments.index, compartments["Ratio"], color=colours)
ax[0].axvline(1.0, color="k", linestyle="--", linewidth=1, label="keine Anreicherung")
ax[0].set_xlabel("Relevanzdichte relativ zum Hirndurchschnitt (Ratio)")
ax[0].set_title("Relevanz je Kompartiment — nur der Thalamus trägt zum Zielwert bei")
ax[0].legend()
ax[0].grid(alpha=0.3, axis="x")

ax[1].hist(phantom_results["ratio"], bins=8, color="tab:red", alpha=0.8)
ax[1].axvline(1.0, color="k", linestyle="--", linewidth=1, label="keine Anreicherung")
ax[1].axvline(phantom_results["ratio"].mean(), color="darkred",
              label=f"Mittel = {phantom_results['ratio'].mean():.2f}")
ax[1].set_xlabel("Ratio im Thalamus")
ax[1].set_ylabel("Anzahl Phantome")
ax[1].set_title("Stabilität über die Testphantome")
ax[1].legend()
fig.savefig(target_dir / "08_relevanz_je_kompartiment.png", bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# <a id="sec-13"></a>
# ## 13. Wie scharf ist die Lokalisation? Dilatation und Distanzprofil
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Warum eine scharfe Maske zu streng ist
#
# Abschnitt 12 hat den Thalamus als messerscharf abgegrenztes Ziel behandelt: ein Voxel liegt
# drinnen oder draußen. Das ist aus einem inhaltlichen Grund zu streng. Um zu erkennen, dass der
# Thalamus **groß** ist, ist der informativste Ort im Bild nicht dessen Mitte — dort sieht alles
# gleich aus, egal wie groß die Struktur ist —, sondern die **Grenze**. Und die Grenze liegt zur
# Hälfte im Nachbargewebe.
#
# Ein Netz, das die Relevanz auf einen Ring um den Thalamus legt, tut also etwas **Vernünftiges**.
# Nach der Metrik aus Abschnitt 12 würde es dafür bestraft.
#
# Deshalb zwei ergänzende Auswertungen:
#
# **1. Dilatierte Maske.** Wir vergrößern die Zielmaske um $r$ Voxel und messen erneut. Sei $d(v)$
# der euklidische Abstand zum nächsten Thalamusvoxel:
#
# $$T_r = \{v : d(v) \le r\}$$
#
# Wichtig ist, dabei die Ratio (nicht nur die Masse) mitzuführen: Ein größeres Gebiet sammelt
# selbstverständlich mehr Relevanz ein. Die Ratio korrigiert das automatisch, weil sie durch den
# gewachsenen Volumenanteil teilt.
#
# **2. Distanzprofil.** Wir zerlegen die Relevanz in Abstandsschalen. Fällt sie mit dem Abstand ab,
# ist die Lokalisation echt; ist sie gleichmäßig verteilt, wäre der Befund aus Abschnitt 12 ein
# Zufallsprodukt.
#
# ### Ausgabe dieser Zelle
#
# ```text
# Ziel               R-Anteil  Vol-Anteil   Ratio
# Thalamus (exakt)      0.092      0.018     5.217
# Thalamus + 2 Voxel    0.243      0.040     6.042
# Thalamus + 4 Voxel    0.335      0.079     4.165
# Thalamus + 6 Voxel    0.376      0.139     2.643
#
# Relevanz nach Abstand zum Thalamus (Ratio 1.0 = Hirndurchschnitt):
#                       R-Anteil  Vol-Anteil  Ratio
# im Thalamus              0.092       0.018  5.016
# 0–1 mm                   0.077       0.009  8.434
# 1–2 mm                   0.074       0.012  5.956
# 2–4 mm                   0.092       0.039  2.386
# 4–8 mm                   0.073       0.132  0.554
# 8–16 mm                  0.290       0.460  0.632
# > 16 mm                  0.301       0.330  0.914
#
# Summe der Relevanzanteile: 1.000
# ```
#
# ### Interpretation: die Ratio steigt, wenn man die Maske aufweitet
#
# **Das ist der wichtigste Befund dieses Notebooks.** Die exakte Maske erreicht Ratio 5,22 — mit
# einem Toleranzrand von nur **2 Voxeln steigt sie auf 6,04**. Der Relevanzanteil springt von 9 %
# auf **24 %**, während der Volumenanteil nur von 1,8 % auf 4,0 % wächst.
#
# Dass die Anreicherung durch *Vergrößern* der Zielregion besser wird, kann nur eine Ursache haben:
# **Ein erheblicher Teil der Relevanz liegt knapp außerhalb des Thalamus, in einer dünnen Schale um
# ihn herum.** Ab 4 Voxeln kippt es dann (Ratio 4,17, danach 2,64) — dort wächst nur noch das
# Volumen, nicht mehr die Relevanz.
#
# **Das Distanzprofil bestätigt es direkt und in Zahlen.** Die Relevanzdichte ist nicht im Thalamus
# am höchsten (Ratio 5,02), sondern in der Schale **0–1 mm außerhalb** (**8,43**); die Schale
# 1–2 mm liegt mit 5,96 noch über dem Inneren. Ab 2–4 mm bricht die Dichte ein (2,39) und fällt
# jenseits von 4 mm **unter** den Hirndurchschnitt (0,55 / 0,63 / 0,91).
#
# **Damit löst sich der scheinbare Widerspruch aus Abschnitt 12 auf:** Das Pointing Game scheiterte
# (0,08), weil das absolut stärkste Voxel typischerweise **einen Millimeter neben** der Maske liegt.
# Nach dem strengen Kriterium ist das ein Fehlschlag; inhaltlich ist es **genau richtig**. Um zu
# messen, wie *groß* eine Struktur ist, muss man ihren Rand ansehen, und der Rand ist per
# Definition der Ort, an dem zwei Gewebe aneinanderstoßen. Das Netz hat einen Kantendetektor
# gelernt — die vernünftigste mögliche Lösung für eine Volumenmessung.
#
# > **Konsequenz für die Praxis:** Eine harte Ja/Nein-Bewertung an der Maskengrenze bestraft
# > genau das Verhalten, das man sich wünscht. Wer Lokalisationsmetriken auf echten Daten
# > berichtet, sollte **immer** mindestens einen kleinen Toleranzrand mitangeben.
#
# **Der Anstieg bei 8–16 mm und > 16 mm ist ein Volumeneffekt, keine zweite Fundstelle.** Roh
# gesehen liegen dort 29 % und 30 % der Relevanz — dreimal mehr als im Thalamus. Aber diese beiden
# Schalen machen zusammen **79 % des Hirnvolumens** aus. Pro Voxel gerechnet stehen sie bei 0,63
# und 0,91, also unterhalb des Durchschnitts und bei weniger als einem Zehntel der Randschale.
#
# > **Ein Fallstrick, der genau hier lauert:** Ein rohes Distanzprofil sieht immer so aus, als
# > würde weit entfernt „viel" passieren, weil weit entfernt einfach mehr Platz ist. Wer diese
# > Normierung vergisst, kommt zum Gegenteil des richtigen Schlusses.
#
# ### Was die Abbildung zeigt
#
# **Links:** Relevanzanteil und Volumenanteil in Abhängigkeit vom Toleranzradius, plus die Ratio
# auf der zweiten y-Achse. Der Relevanzanteil steigt zunächst viel steiler als der Volumenanteil
# (dort ist die Lokalisation), danach flacht er ab, während das Volumen weiterwächst. Die
# Ratio-Kurve hat deshalb ihr **Maximum bei 2 Voxeln** und fällt erst danach.
#
# **Rechts:** Das Distanzprofil in zwei Lesarten. Die blauen Balken sind der **rohe**
# Relevanzanteil je Abstandsschale; sie steigen mit dem Abstand an und täuschen damit. Die rote
# Linie ist die **Ratio**, also derselbe Anteil geteilt durch den Volumenanteil der Schale; sie hat
# ihren Gipfel unmittelbar am Thalamusrand und fällt dahinter steil unter die Referenzlinie 1,0.

# %%
dilation_rows = [{
    "Ziel": "Thalamus (exakt)",
    "R-Anteil": phantom_results["mass"].mean(),
    "Vol-Anteil": phantom_results["share"].mean(),
    "Ratio": phantom_results["ratio"].mean(),
}]
for radius in DILATION_RADII:
    dilation_rows.append({
        "Ziel": f"Thalamus + {radius} Voxel",
        "R-Anteil": phantom_results[f"mass_r{radius}"].mean(),
        "Vol-Anteil": phantom_results[f"share_r{radius}"].mean(),
        "Ratio": phantom_results[f"ratio_r{radius}"].mean(),
    })
dilation = pd.DataFrame(dilation_rows).set_index("Ziel")
dilation.to_csv(target_dir / "09_dilatation.csv")
print(dilation.round(3).to_string())

profile = np.array([phantom_results[f"profil_{label}"].mean() for label in PROFILE_LABELS])

# Schalengroesse: wie viele Voxel liegen ueberhaupt in jeder Abstandsschale?
shell_sizes = np.zeros(len(PROFILE_LABELS))
brain_sizes = []
for index in eval_index:
    label = test_L[index]
    brain = label != LABEL_BACKGROUND
    thalamus = label == LABEL_THALAMUS
    distance = distance_to(thalamus)
    brain_sizes.append(brain.sum())
    shell_sizes[0] += thalamus.sum()
    for j, (low, high) in enumerate(zip(PROFILE_EDGES[:-1], PROFILE_EDGES[1:]), start=1):
        shell_sizes[j] += ((distance > low) & (distance <= high) & brain).sum()
shell_sizes /= len(eval_index)

# Ratio je Schale: Relevanzdichte in der Schale, geteilt durch die mittlere Hirndichte.
shell_shares = shell_sizes / np.mean(brain_sizes)
shell_ratios = profile / shell_shares

profile_table = pd.DataFrame({"R-Anteil": profile, "Vol-Anteil": shell_shares,
                              "Ratio": shell_ratios}, index=PROFILE_LABELS)
profile_table.index.name = "Abstand zum Thalamus"
profile_table.to_csv(target_dir / "09_distanzprofil.csv")
print("\nRelevanz nach Abstand zum Thalamus (Ratio 1.0 = Hirndurchschnitt):")
print(profile_table.round(3).to_string())
print(f"\nSumme der Relevanzanteile: {profile.sum():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
radii = [0, *DILATION_RADII]
ax[0].plot(radii, dilation["R-Anteil"].values, "o-", label="Relevanzanteil")
ax[0].plot(radii, dilation["Vol-Anteil"].values, "s-", label="Volumenanteil")
ax[0].set_xlabel("Toleranzrand um den Thalamus (Voxel = mm)")
ax[0].set_ylabel("Anteil")
ax[0].set_title("Relevanz steigt schneller als Volumen — solange die Lokalisation reicht")
ax[0].grid(alpha=0.3)
twin = ax[0].twinx()
twin.plot(radii, dilation["Ratio"].values, "^--", color="tab:red", label="Ratio")
twin.axhline(1.0, color="grey", linestyle=":", linewidth=1)
twin.set_ylabel("Ratio", color="tab:red")
lines = ax[0].get_lines() + twin.get_lines()[:1]
ax[0].legend(lines, [l.get_label() for l in lines], loc="center right")

positions = np.arange(len(PROFILE_LABELS))
ax[1].bar(positions, profile, width=0.55, label="Relevanzanteil (roh)")
ax[1].set_xticks(positions)
ax[1].set_xticklabels(PROFILE_LABELS, rotation=30, ha="right")
ax[1].set_xlabel("Abstand zum Thalamus")
ax[1].set_ylabel("Anteil der positiven Relevanz")
ax[1].set_title("Distanzprofil: erst pro Voxel gerechnet wird der Abfall sichtbar")
ax[1].grid(alpha=0.3, axis="y")
twin_profile = ax[1].twinx()
twin_profile.plot(positions, shell_ratios, "o--", color="tab:red",
                  label="Ratio (Relevanz pro Voxel)")
twin_profile.axhline(1.0, color="grey", linestyle=":", linewidth=1)
twin_profile.set_ylabel("Ratio", color="tab:red")
profile_lines = ax[1].containers[0].patches[:1] + twin_profile.get_lines()[:1]
ax[1].legend(profile_lines, ["Relevanzanteil (roh)", "Ratio (Relevanz pro Voxel)"],
             loc="upper center")
fig.savefig(target_dir / "09_dilatation_und_distanzprofil.png", bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# <a id="sec-14"></a>
# ## 14. Der Regelvergleich: die Regel entscheidet
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Bisher haben wir **eine** LRP-Konfiguration benutzt. Jetzt vergleichen wir vier — dasselbe
# Modell, dieselben Bilder, dieselben Metriken, nur andere Regeln im Rückwärtsweg:
#
# | Bezeichnung | Konfiguration | Idee |
# |---|---|---|
# | `flat + αβ` | unterste Schicht flat, dann $\alpha\beta$, oben $\varepsilon$ | die empfohlene Composite-Strategie |
# | `nur αβ` | überall $\alpha=2, \beta=1$ | ohne Glättung in der untersten Schicht |
# | `nur ε` | überall $\varepsilon = 0{,}25$ | die einfachste stabile Regel |
# | `nur ε (klein)` | überall $\varepsilon = 10^{-6}$ | praktisch die reine z-Regel |
#
# ### Ausgabe dieser Zelle
#
# ```text
# Strategie      ratio  precision    hit   mass  sum_R
# flat + αβ      5.038      0.261  0.000  0.098  7.801
# nur αβ         0.742      0.081  0.000  0.015  9.614
# nur ε          1.657      0.391  0.375  0.030  0.084
# nur ε (klein)  2.366      0.345  0.125  0.042 -0.489
# ```
#
# ### Interpretation
#
# **Der Unterschied ist gewaltig.** Dieselbe Erklärung desselben Modells erreicht je nach Regel eine
# Ratio von **5,04 oder von 0,74**. Der Wert 0,74 bedeutet: keine Lokalisation, die Relevanz meidet
# den Thalamus sogar. Hätten wir mit `nur αβ` angefangen, wäre unser Fazit gewesen: „LRP findet den
# Thalamus nicht." Mit `flat + αβ`: „LRP findet ihn gut." Beide Sätze wären auf derselben Datenlage,
# demselben Modell und denselben Bildern gefallen.
#
# **Warum ist `nur αβ` so viel schlechter?** Die $\alpha\beta$-Regel enthält den Faktor $a_j$, also
# die Aktivierung. In der untersten Schicht sind die „Aktivierungen" die Grauwerte selbst. Die
# Regel gewichtet dort also nach Helligkeit — und die weiße Substanz ist mit 0,80 die hellste
# Struktur im Bild, der Thalamus mit 0,62 dunkler. Die Relevanz wandert damit systematisch in die
# weiße Substanz. Die **flat-Regel** setzt $a \leftarrow 1$ und macht die unterste Schicht
# absichtlich blind für die Helligkeit; sie verteilt nur nach der Geometrie des rezeptiven Feldes.
# Genau dafür wurde sie eingeführt.
#
# **Die beiden ε-Varianten kehren die Rangfolge um.** Sie haben die niedrigere Ratio (1,66 bzw.
# 2,37), aber die **höhere Top-$k$-Präzision** (0,391 bzw. 0,345 gegen 0,261) — und sie sind die
# einzigen, die das Pointing Game überhaupt gewinnen (0,375 bzw. 0,125 gegen 0,000). Das heißt: die
# wenigen ganz starken Voxel sitzen bei ε präziser, die Gesamtmasse verteilt sich aber breiter.
#
# Damit liegen die vier Strategien je nach Metrik in **unterschiedlicher Reihenfolge**:
#
# | Metrik | Beste Strategie |
# |---|---|
# | Ratio / Relevanzmasse | `flat + αβ` |
# | Top-$k$-Präzision | `nur ε` |
# | Pointing Game | `nur ε` |
#
# Wer nur die Ratio berichtet, würde ε verwerfen; wer nur die Präzision berichtet, würde `flat + αβ`
# verwerfen. **Es gibt hier keine eindeutig „beste" Regel, und deshalb muss man mehrere Metriken
# angeben.**
#
# **Die Summe der Relevanz variiert von −0,49 bis 9,61**, während die Vorhersage bei ≈ 13 liegt —
# bei `nur ε (klein)` ist sie sogar **negativ**. Die Erhaltungseigenschaft
# $\sum_v R_v \approx \hat y$ ist also bei keiner Strategie erfüllt; die Ursachen sind die
# Bias-Terme und die BatchNorm-Schichten, die Relevanz aufnehmen, ohne sie weiterzugeben. Praktisch
# folgt daraus: Karten sind nur **innerhalb** einer Strategie vergleichbar, absolute Relevanzwerte
# über Strategien hinweg sind bedeutungslos.
#
# ### Die Lehre daraus
#
# > Die LRP-Regel ist kein Implementierungsdetail, sondern ein **Freiheitsgrad mit
# > ergebnisbestimmender Wirkung**. Wer eine Aussage über „LRP" trifft, muss die Konfiguration
# > mit angeben — sonst ist die Aussage nicht überprüfbar.
#
# Und daraus folgt eine methodische Warnung: Man ist versucht, mit einem Ground-Truth-Datensatz wie
# diesem die **beste** Strategie zu suchen. Das ist legitim, aber dann hat man die Strategie *auf
# diesen Daten* optimiert und muss sie auf **anderen** Daten validieren. Sonst ist der schöne Wert
# von 5,0 nur ein Overfitting an die Phantome.
#
# ### Was die Abbildung zeigt
#
# Gruppierte Balken für Ratio, Top-$k$-Präzision und Pointing Game je Strategie, mit den
# jeweiligen Zufallsniveaus als Referenzlinien. Die vier Strategien liegen bei jeder Metrik anders
# — und sie liegen nicht einmal in derselben **Reihenfolge**. Das ist die kompakteste Form der
# Aussage „es gibt nicht *die* LRP-Erklärung".

# %%
STRATEGIES = {
    "flat + αβ": PHANTOM_STRATEGY_LAYERS,
    "nur αβ": [{"alpha": 2, "beta": 1}] * 5 + [{"epsilon": 0.25}],
    "nur ε": [{"epsilon": 0.25}] * 6,
    "nur ε (klein)": [{"epsilon": 1e-6}] * 6,
}

N_RULE_EVAL = 8
rule_index = eval_index[:N_RULE_EVAL]
rule_rows = []

for name, layers in STRATEGIES.items():
    R_all = explain_phantom(phantom_model, layers, test_X[rule_index])
    for k, index in enumerate(rule_index):
        label = test_L[index]
        brain = label != LABEL_BACKGROUND
        R = R_all[k] * brain
        rule_rows.append({"Strategie": name,
                          **relevance_metrics(R, label == LABEL_THALAMUS, brain)})

rules = (pd.DataFrame(rule_rows)
         .groupby("Strategie", sort=False)[["ratio", "precision", "hit", "mass", "sum_R"]]
         .mean())
rules.to_csv(target_dir / "10_regelvergleich.csv")
print(f"Gemittelt über {len(rule_index)} Testphantome "
      f"(Zufallsniveau für Prec@k und Hit: {phantom_results['share'].mean():.3f})\n")
print(rules.round(3).to_string())

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
chance = phantom_results["share"].mean()
panels = [("ratio", "Ratio (Relevanzdichte)", 1.0, "keine Anreicherung"),
          ("precision", "Top-k-Präzision", chance, "Zufall"),
          ("hit", "Pointing Game", chance, "Zufall")]
for axis, (column, title, reference, reference_label) in zip(ax, panels):
    axis.bar(rules.index, rules[column], color=["tab:red", "tab:blue", "tab:green", "tab:purple"])
    axis.axhline(reference, color="k", linestyle="--", linewidth=1, label=reference_label)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=25)
    axis.legend()
    axis.grid(alpha=0.3, axis="y")
fig.suptitle("Dieselbe Erklärung, dieselben Bilder — vier LRP-Regeln, vier Ergebnisse",
             fontsize=13)
fig.savefig(target_dir / "10_regelvergleich.png", bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# <a id="sec-15"></a>
# ## 15. Sanity-Check: randomisierte Gewichte
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Die Idee
#
# Der wichtigste Kontrolltest für Attributionsmethoden überhaupt, eingeführt von Adebayo et al.
# (2018) unter dem Namen *model parameter randomization test*:
#
# > Wenn man die Gewichte eines Netzes durch Zufallszahlen ersetzt, kann seine Erklärung keine
# > sinnvolle Information mehr enthalten. **Eine Attributionsmethode, die trotzdem dieselbe
# > Heatmap liefert, erklärt nicht das Modell, sondern nur das Bild.**
#
# Erschreckend viele populäre Verfahren fallen durch diesen Test — insbesondere solche, die
# hauptsächlich Kanten des Eingabebildes zurückgeben. Sie produzieren dann auch für ein
# untrainiertes Netz eine überzeugend aussehende Karte.
#
# `clone_model()` baut die Architektur identisch nach; anschließend setzen wir alle Gewichte auf
# den Zustand eines frisch initialisierten Netzes zurück. Dasselbe LRP, dieselben Bilder,
# dieselben Metriken.
#
# Die Neuinitialisierung machen wir bewusst mit einem eigenen, gesetzten Zufallsgenerator statt mit
# den TensorFlow-Voreinstellungen. Deren Zufallszahlen hängen am globalen Zustand der Sitzung — das
# Ergebnis wäre also davon abhängig, ob das Modell in dieser Sitzung trainiert oder aus dem Cache
# geladen wurde. Für einen **Kontrolltest** ist das inakzeptabel: er muss bei jedem Durchlauf
# dieselbe Zahl liefern.
#
# ### Ausgabe dieser Zelle
#
# ```text
# Modell           ratio  precision  hit   mass  sum_positive
# trainiert        5.038      0.261  0.0  0.098         7.856
# Zufallsgewichte  0.370      0.000  0.0  0.007         0.335
# ```
#
# ### Interpretation
#
# **Der Test wird bestanden, und zwar deutlich.** Mit Zufallsgewichten fällt die Ratio von 5,04 auf
# **0,37** — also weit *unter* den Neutralwert 1, die Relevanz meidet den Thalamus dann sogar —, die
# Relevanzmasse von 9,8 % auf 0,7 % und die Top-$k$-Präzision auf **exakt 0**. Kein einziges der
# $k$ höchstbewerteten Voxel liegt noch im Thalamus.
#
# Die letzte Spalte zeigt zusätzlich, dass auch die **Gesamtmenge** an positiver Relevanz um mehr
# als das Zwanzigfache einbricht (7,86 gegen 0,34). Sie ist mit aufgeführt, weil Masse und Ratio
# undefiniert werden, sobald eine Karte im Hirn überhaupt keine positive Zelle mehr enthält — ein
# Fall, der bei zufälligen Gewichten durchaus vorkommt und dann als `NaN` in der Tabelle steht.
#
# Damit ist die zentrale Alternativerklärung ausgeräumt: Die Anreicherung aus Abschnitt 12 stammt
# **nicht** daraus, dass LRP im Wesentlichen das Eingabebild zurückgibt und der Thalamus dort
# ohnehin auffällig ist. Sie hängt an den **gelernten Gewichten** — also an dem, was das Modell
# tatsächlich tut. (Dass das Pointing Game in beiden Zeilen 0,0 ist, hilft hier nicht weiter; es
# war schon beim trainierten Modell wenig aussagekräftig, siehe Abschnitt 13.)
#
# Vergleichen Sie das bitte mit Abschnitt 8: Dort hatten wir ein Modell, das seine Eingabe
# ignoriert, und **trotzdem** eine Ratio von 4,00 auf den permutierten Bildern. Der Unterschied ist
# lehrreich:
#
# | | Abschnitt 8 (konstantes Modell) | Abschnitt 15 (Zufallsgewichte) |
# |---|---|---|
# | Modell benutzt den Input | nein | nein |
# | Ratio im Thalamus | 4,00 (scheinbarer Erfolg) | 0,37 (klarer Misserfolg) |
# | Grund | Testbild war so gebaut, dass nur der Thalamus Struktur hatte | Testbild enthielt überall Struktur |
#
# Die Metrik ist also nicht „falsch" — sie hängt davon ab, wie man die **Kontrollbedingung**
# konstruiert. Das ist die praktisch wichtigste Erkenntnis dieses Notebooks: Ein Sanity-Check ist
# nur so gut wie das Gegenbeispiel, gegen das er prüft.
#
# ### Was die Abbildung zeigt
#
# Oben die Relevanzkarte des trainierten Modells, unten die desselben Bildes mit Zufallsgewichten,
# jeweils mit eingezeichneter Thalamusgrenze (grün).
#
# **Oben** liegt ein deutlicher roter **Ring genau auf der grünen Kontur**, während das Innere
# blass bis leicht blau bleibt — die Bildversion des Befunds aus Abschnitt 13: Das Modell liest den
# Rand, nicht die Mitte. Außerdem sieht man einen schwachen roten Ring an der Hirnaußengrenze und
# rote Flecken an den Ventrikeln (die falsch positiven 4,17 aus Abschnitt 12).
#
# **Unten** ist das gesamte Hirn diffus rot ausgemalt, ohne jeden Bezug zu den Strukturgrenzen;
# innerhalb der grünen Kontur liegt praktisch nichts, und die stärksten Flecken sitzen irgendwo im
# Gewebe. Das ist genau das Aussehen einer Karte, die nichts erklärt.
#
# Bemerkenswert ist dabei ein Darstellungseffekt: Die untere Zeile sieht **kräftiger gefärbt** aus
# als die obere, obwohl sie insgesamt zwanzigmal *weniger* positive Relevanz enthält (0,34 gegen
# 7,86). Der Grund ist, dass jede Karte auf ihr **eigenes** Maximum normiert dargestellt wird: Eine
# breit verschmierte Karte ohne ausgeprägte Spitze erscheint dadurch überall sattfarben, eine gut
# lokalisierte Karte dagegen fast überall blass. **Eine intensive Heatmap ist damit kein
# Qualitätsmerkmal** — sondern womöglich das Gegenteil. Es ist einer der Hauptgründe, Farbbilder
# nie ohne die zugehörigen Zahlen zu zeigen.

# %%
from tensorflow.keras.models import clone_model


def randomize_weights(model, seed: int) -> None:
    """Setzt alle Gewichte auf den Zustand eines frisch initialisierten Netzes zurueck.

    Wir ziehen die Zahlen selbst mit einem numpy-Generator, statt uns auf clone_model zu
    verlassen: dessen Zufallszahlen haengen am globalen TensorFlow-Zustand und damit daran,
    was in der Sitzung vorher gerechnet wurde. Der Sanity-Check waere dann nicht reproduzierbar.
    Die Regeln unten sind genau die Keras-Voreinstellungen (Glorot-uniform fuer Kernel,
    Null fuer Bias und beta, Eins fuer gamma und die BatchNorm-Statistik).
    """
    rng = np.random.default_rng(seed)
    for layer in model.layers:
        if not layer.weights:
            continue
        values = []
        for variable in layer.weights:
            shape = tuple(variable.shape)
            if variable.name == "kernel":
                fan_in = int(np.prod(shape[:-1]))
                limit = np.sqrt(6.0 / (fan_in + shape[-1]))
                values.append(rng.uniform(-limit, limit, shape))
            elif variable.name in ("gamma", "moving_variance"):
                values.append(np.ones(shape))
            else:  # bias, beta, moving_mean
                values.append(np.zeros(shape))
        layer.set_weights([v.astype("float32") for v in values])


with tf.device(SMALL_MODEL_DEVICE):
    random_model = clone_model(phantom_model)
    randomize_weights(random_model, RANDOM_SEED + 1)

random_R = explain_phantom(random_model, PHANTOM_STRATEGY_LAYERS, test_X[rule_index])

random_rows = []
for k, index in enumerate(rule_index):
    label = test_L[index]
    brain = label != LABEL_BACKGROUND
    random_rows.append(relevance_metrics(random_R[k] * brain,
                                         label == LABEL_THALAMUS, brain))
random_results = pd.DataFrame(random_rows)

COMPARISON_COLUMNS = ["ratio", "precision", "hit", "mass", "sum_positive"]
comparison = pd.DataFrame({
    "trainiert": phantom_results.loc[phantom_results["index"].isin(rule_index),
                                     COMPARISON_COLUMNS].mean(),
    "Zufallsgewichte": random_results[COMPARISON_COLUMNS].mean(),
}).T
comparison.to_csv(target_dir / "11_randomisierungstest.csv")
print(comparison.round(3).to_string())

if not np.isfinite(comparison.loc["Zufallsgewichte", "ratio"]):
    print("\nMasse und Ratio sind fuer die Zufallsgewichte NaN, weil die Relevanzkarte im Hirn")
    print("keine einzige positive Zelle enthaelt - der Nenner der Masse ist also null.")

reference = int(rule_index[0])
label = test_L[reference]
brain = label != LABEL_BACKGROUND
thalamus = label == LABEL_THALAMUS
z = int(np.argwhere(thalamus)[:, 2].mean())

fig, ax = plt.subplots(2, 3, figsize=(14, 9))
for row, (name, R) in enumerate([("trainiert", eval_R[0]), ("Zufallsgewichte", random_R[0])]):
    scaled = (R * brain)
    scaled = scaled / np.abs(scaled).max()
    for col, (view, data, mask) in enumerate([
        ("sagittal", scaled[scaled.shape[0] // 2], thalamus[thalamus.shape[0] // 2]),
        ("koronal", scaled[:, PHANTOM_SIZE // 2 - 6], thalamus[:, PHANTOM_SIZE // 2 - 6]),
        ("axial", scaled[:, :, z], thalamus[:, :, z]),
    ]):
        ax[row][col].imshow(data, cmap="seismic", clim=(-1, 1))
        ax[row][col].contour(mask.astype(float), levels=[0.5], colors="lime", linewidths=1.2)
        ax[row][col].set_title(f"{name} — {view}")
        ax[row][col].axis("off")
fig.suptitle("Randomisierungstest: mit Zufallsgewichten verschwindet die Lokalisation",
             fontsize=13)
fig.savefig(target_dir / "11_randomisierungstest.png", bbox_inches="tight", dpi=150)
plt.show()

# %% [markdown]
# <a id="sec-16"></a>
# ## 16. Übertragung auf echte Daten: das FreeSurfer/FSL-Rezept
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Abschnitt 3 hat die Thalamusmaske in Python erzeugt, weil im Repo schon fertige
# FastSurfer-Segmentierungen liegen. Für neue Daten muss die Kette einmal komplett durchlaufen
# werden. Dieser Abschnitt dokumentiert sie — mit FSL-Werkzeugen, wie es in der
# Neuroimaging-Praxis üblich ist. Die Zelle am Ende **schreibt die Skripte in den Ausgabeordner**,
# führt sie aber nicht aus (FreeSurfer und FSL sind keine Python-Pakete und müssen installiert
# sein).
#
# ### Der entscheidende Punkt: Bild und Maske müssen dieselbe Transformation erfahren
#
# Das ist der Fehler, den man hier machen kann, und er ist tückisch, weil das Ergebnis trotzdem
# plausibel aussieht:
#
# ```text
#   FALSCH                                RICHTIG
#   ──────                                ───────
#   T1  ──flirt──►  T1 in MNI             T1  ──flirt (omat)──►  T1 in MNI
#                                                    │
#                                                    └─► Transformationsmatrix speichern
#   aseg ──flirt──► aseg in MNI                                   │
#          (eigene Registrierung!)        aseg ──flirt (applyxfm, init=Matrix)──► aseg in MNI
#
#   → zwei verschiedene Transformationen   → identische Transformation
#   → Maske um einige mm verschoben        → Maske sitzt exakt
# ```
#
# Registriert man die Segmentierung **eigenständig**, optimiert `flirt` sie unabhängig, und die
# Ergebnisse weichen um mehrere Millimeter ab. Bei einer Struktur von 15 mm Durchmesser ist das
# viel. Die Lösung ist der Parameter `-omat`: er speichert die aus dem T1 berechnete Matrix, die
# dann mit `-applyxfm -init` auf die Segmentierung angewendet wird.
#
# ### Die Schritte im Einzelnen
#
# | # | Schritt | Werkzeug | Warum |
# |---|---|---|---|
# | 1 | Segmentierung erzeugen | `recon-all` bzw. `run_fastsurfer.sh` | liefert `aseg.mgz` mit den Labelnummern |
# | 2 | Nach NIfTI konvertieren | `mri_convert` | FSL liest `.mgz` nicht |
# | 3 | Reorientieren | `fslreorient2std` | einheitliche Achsenkonvention |
# | 4 | T1 auf MNI152 registrieren, **Matrix speichern** | `flirt -dof 6 -omat` | 6 DOF, damit Volumina erhalten bleiben |
# | 5 | Segmentierung mit **derselben** Matrix transformieren | `flirt -applyxfm -init -interp nearestneighbour` | Labels dürfen nicht interpoliert werden |
# | 6 | Thalamus extrahieren | `fslmaths -thr 10 -uthr 10 -bin` | Labels 10 (links) und 49 (rechts) |
# | 7 | Beide Seiten zusammenfassen | `fslmaths -add -bin` | eine Maske für den gesamten Thalamus |
# | 8 | Crop auf die Modellgröße | `pyment.utils.preprocessing.crop` | 182×218×182 → 167×212×160 |
# | 9 | Visuell prüfen | `fsleyes` | der Schritt, den man nie auslassen sollte |
#
# Zu **Schritt 6**: `-thr 10 -uthr 10` heißt *lower threshold* 10 und *upper threshold* 10 —
# behalte nur Voxel mit genau dem Wert 10. `-bin` macht daraus eine 0/1-Maske.
#
# Zu **Schritt 8**: Dieser Schritt wird gern vergessen. Das Modell erwartet 167×212×160; die
# MNI-registrierten Bilder sind 182×218×182. Der Crop muss auf **Bild und Maske identisch**
# angewendet werden — sonst sitzt die Maske am Ende doch daneben. Die im Repo verwendeten Grenzen
# sind `bounds = ((6, 173), (2, 214), (0, 160))` (Reihenfolge y, x, z; Start inklusive, Ende
# exklusive), siehe [`doc/IXI_Datenlayout.md`](../../doc/IXI_Datenlayout.md).
#
# ### Ein Hinweis zum Python-Weg
#
# Wenn eine Segmentierung schon existiert und nur das Gitter nicht passt, braucht man diese ganze
# Kette nicht. `nibabel.processing.resample_from_to(aseg, image, order=0)` aus Abschnitt 3 macht
# das in einer Zeile, weil beide Dateien ihre Affine mitbringen. Die FSL-Kette braucht man für den
# Fall, dass **noch keine** gemeinsame Registrierung existiert.
#
# ### Der Permutationsschritt auf echten Daten
#
# Das mitgeschriebene Python-Skript `shuffle_T1.py` ist die Variante aus Abschnitt 4 als
# eigenständiges Kommandozeilenwerkzeug — es permutiert alle Nicht-Null-Voxel außerhalb der
# Thalamusmaske und erhält damit Gehirnform und Grauwerthistogramm.
#
# ### Und die Warnung dazu
#
# Bevor man diese Kette auf eine ganze Kohorte anwendet, sollte man Abschnitt 7 und 8 im Kopf
# haben:
#
# 1. **Prüfen, ob das Modell reagiert** (Vorhersage für Original vs. permutiert vs. Nullbild). Ist
#    die Spannweite ≈ 0, hört man hier auf und repariert erst den Checkpoint.
# 2. **Eine zweite, gleich große Kontrollstruktur mitlaufen lassen** (z. B. Hippocampus, Labels 17
#    und 53). Tritt dort die gleiche Anreicherung auf, war der Effekt geometrisch.
# 3. **Die LRP-Regel dokumentieren** und mindestens zwei Varianten rechnen (Abschnitt 14).

# %%
FSL_SCRIPT = r"""#!/usr/bin/env bash
# Thalamusmaske im MNI152-Raum erzeugen.
# Voraussetzung: FreeSurfer/FastSurfer-Lauf liegt vor (aseg + brainmask).
set -euo pipefail

SUBJECT="${1:?Usage: make_thalamus_mask.sh <subject-id> <recon-dir> <out-dir>}"
RECON_DIR="${2:?}"
OUT_DIR="${3:?}"
MNI="${FSLDIR}/data/standard/MNI152_T1_1mm.nii.gz"

mkdir -p "${OUT_DIR}"

# 1) .mgz -> .nii.gz, denn FSL liest kein MGZ
mri_convert "${RECON_DIR}/${SUBJECT}/mri/brainmask.mgz" "${OUT_DIR}/brainmask.nii.gz"
mri_convert "${RECON_DIR}/${SUBJECT}/mri/aseg.mgz"      "${OUT_DIR}/aseg.nii.gz"

# 2) einheitliche Achsenkonvention
fslreorient2std "${OUT_DIR}/brainmask.nii.gz" "${OUT_DIR}/brainmask_reoriented.nii.gz"
fslreorient2std "${OUT_DIR}/aseg.nii.gz"      "${OUT_DIR}/aseg_reoriented.nii.gz"

# 3) T1 auf MNI152 registrieren. -dof 6 = nur Rotation/Translation, KEINE Skalierung,
#    damit die Volumina erhalten bleiben. -omat speichert die Matrix - das ist der Kern.
flirt -in  "${OUT_DIR}/brainmask_reoriented.nii.gz" \
      -ref "${MNI}" \
      -out "${OUT_DIR}/mni152.nii.gz" \
      -dof 6 \
      -omat "${OUT_DIR}/T1_to_mni152.mat"

# 4) Segmentierung mit DERSELBEN Matrix transformieren.
#    nearestneighbour ist zwingend: Labelnummern darf man nicht interpolieren.
flirt -in  "${OUT_DIR}/aseg_reoriented.nii.gz" \
      -ref "${MNI}" \
      -out "${OUT_DIR}/aseg_mni152.nii.gz" \
      -applyxfm -init "${OUT_DIR}/T1_to_mni152.mat" \
      -interp nearestneighbour

# 5) Thalamus extrahieren: 10 = links, 49 = rechts
fslmaths "${OUT_DIR}/aseg_mni152.nii.gz" -thr 10 -uthr 10 -bin "${OUT_DIR}/thal_left.nii.gz"
fslmaths "${OUT_DIR}/aseg_mni152.nii.gz" -thr 49 -uthr 49 -bin "${OUT_DIR}/thal_right.nii.gz"

# 6) beide Seiten zu einer Maske vereinen
fslmaths "${OUT_DIR}/thal_left.nii.gz" -add "${OUT_DIR}/thal_right.nii.gz" -bin \
         "${OUT_DIR}/thalamus_mask.nii.gz"

echo "Volumen des Thalamus (mm^3):"
fslstats "${OUT_DIR}/thalamus_mask.nii.gz" -V

# 7) Immer visuell gegenpruefen!
echo "Pruefen mit: fsleyes ${OUT_DIR}/mni152.nii.gz ${OUT_DIR}/thalamus_mask.nii.gz -cm red"
"""

SHUFFLE_SCRIPT = r'''#!/usr/bin/env python
"""Permutiert alle Hirnvoxel ausserhalb der Thalamusmaske.

Gehirnform (Nicht-Null-Voxel) und Grauwerthistogramm bleiben exakt erhalten;
zerstoert wird ausschliesslich die raeumliche Struktur.

    python shuffle_T1.py mni152.nii.gz thalamus_mask.nii.gz out.nii.gz [seed]
"""
import sys
import nibabel as nib
import numpy as np

t1_file, mask_file, out_file = sys.argv[1:4]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0

t1_img = nib.load(t1_file)
mask_img = nib.load(mask_file)
t1 = t1_img.get_fdata()
mask = mask_img.get_fdata()

if t1.shape != mask.shape:
    raise ValueError(f"Formen passen nicht: {t1.shape} vs {mask.shape}. "
                     "Wurde derselbe Crop auf Bild und Maske angewendet?")

# Nur Hirnvoxel (!= 0) ausserhalb des Thalamus permutieren
shuffle_mask = (mask <= 0) & (t1 != 0)
values = t1[shuffle_mask].copy()
np.random.default_rng(seed).shuffle(values)

shuffled = t1.copy()
shuffled[shuffle_mask] = values

assert np.array_equal(shuffled[mask > 0], t1[mask > 0]), "Thalamus wurde veraendert!"
assert np.array_equal(np.sort(shuffled.ravel()), np.sort(t1.ravel())), "Histogramm geaendert!"

nib.save(nib.Nifti1Image(shuffled, t1_img.affine, t1_img.header), out_file)
print(f"{shuffle_mask.sum():,} Voxel permutiert -> {out_file}")
'''

scripts_dir = target_dir / "scripts"
scripts_dir.mkdir(parents=True, exist_ok=True)
for filename, content in [("make_thalamus_mask.sh", FSL_SCRIPT),
                          ("shuffle_T1.py", SHUFFLE_SCRIPT)]:
    path = scripts_dir / filename
    path.write_text(content)
    path.chmod(0o755)
    print(f"geschrieben: {path}")

print("\nAufruf (FreeSurfer und FSL müssen installiert und initialisiert sein):")
print("  ./make_thalamus_mask.sh sub-002 /pfad/zu/recon /pfad/zur/ausgabe")
print("  python shuffle_T1.py mni152.nii.gz thalamus_mask.nii.gz shuffled.nii.gz 42")

# %% [markdown]
# <a id="sec-17"></a>
# ## 17. Fazit, Fallstricke und nächste Schritte
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Die Antwort auf die Ausgangsfrage
#
# > **Kann man mit LRP überprüfen, ob die richtigen Voxel für das Thalamusvolumen ausgewählt
# > werden?**
#
# **Ja — aber nur mit Ground-Truth-Daten und nur als Lokalisationshinweis, nicht als Messung.**
# Die Zahlen im Detail, für ein Modell, das die Aufgabe nachweislich löst (Test-$R^2$ = 0,996) und
# das nachweislich nur den Thalamus benutzen kann:
#
# | Maß | Wert | Zufallsniveau | Bestwert |
# |---|---|---|---|
# | Relevanzdichte im Thalamus (Ratio) | **6,6** | 1,0 | ≈ 42 |
# | Relevanzmasse im Thalamus | 0,159 | 0,024 | 1,0 |
# | … mit 4 Voxeln Rand | **0,503** | 0,149 | 1,0 |
# | Top-$k$-Präzision | 0,368 | 0,024 | 1,0 |
# | Pointing Game | 0,50 | 0,024 | 1,0 |
#
# Anders formuliert: Wenn man alle Hirnregionen nach Relevanzdichte sortiert, landet der Thalamus
# ganz oben — das würde man in einer regionenweisen Auswertung zuverlässig sehen. Aber 59 % der
# Relevanz liegt in weißer Substanz, die nichts beiträgt, und in jedem zweiten Bild liegt das
# stärkste Voxel außerhalb der Struktur. **Für „welche Region ist wichtig?" reicht das. Für
# „welche Voxel genau?" nicht.**
#
# ### Die sechs Fallstricke, gesammelt
#
# Diese Liste ist der praktische Kern des Notebooks.
#
# **1. Ein Modell, das nicht reagiert, produziert trotzdem Heatmaps.** (Abschnitt 7) Der
# mitgelieferte Brain-Age-Checkpoint gibt für ein echtes Gehirn und für ein leeres Bild dieselbe
# Zahl aus — Ursache ist ein Laden „by layer order" statt nach Namen. Der Drei-Zeilen-Test
# `predict(brain)` vs. `predict(zeros)` gehört an den Anfang jeder Pipeline.
#
# **2. Der Permutationstest ist nur in einer Richtung aussagekräftig.** (Abschnitt 8) Er belegt
# zuverlässig, dass etwas faul ist, wenn Relevanz im zerstörten Bereich landet. Er belegt **nicht**
# das Gegenteil: das konstante Modell erreichte auf permutierten Bildern eine Ratio von ≈ 4 und
# bestand das Pointing Game — allein weil der erhaltene Thalamus die einzige glatte Struktur im
# Bild war.
#
# **3. Konfounder muss man messen, nicht hoffen.** (Abschnitt 5) In der ersten Version des
# Phantom-Generators überschrieb ein wachsender Thalamus Ventrikelvoxel, mit
# $\rho(y, \text{Ventrikel}) = -0{,}64$. Das Netz hätte die Aufgabe über den Ventrikel lösen
# können, und LRP hätte zu Recht dorthin gezeigt. Jede Störgröße gegen den Zielwert korrelieren —
# vor dem Training.
#
# **4. Die LRP-Regel entscheidet über das Ergebnis.** (Abschnitt 14) Ratio 6,6 mit `flat + αβ`
# gegen Ratio 0,9 mit `nur αβ`. Ohne Angabe der Konfiguration ist eine Aussage über „LRP" nicht
# überprüfbar.
#
# **5. Distanzprofile und Regionsummen brauchen eine Größennormierung.** (Abschnitt 13) Eine
# Kugelschale in 8–16 mm Abstand ist viel größer als eine in 0–1 mm und sammelt deshalb mehr
# Relevanz — auch bei völlig gleichmäßiger Verteilung. Dasselbe gilt für Hirnregionen: große
# Regionen führen jede Rangliste an, wenn man nicht durch die Voxelzahl teilt.
#
# **6. Eine scharfe Maske ist die falsche Ground Truth für Größenschätzung.** (Abschnitt 13) Die
# Evidenz für „der Thalamus ist groß" sitzt an seiner **Grenze**, und die liegt zur Hälfte im
# Nachbargewebe. Ohne Toleranzrand bestraft man das Modell für vernünftiges Verhalten.
#
# ### Was dieses Notebook nicht zeigt
#
# Ehrliche Grenzen der Aussagekraft:
#
# * **Die Phantome sind keine Gehirne.** Vier Ellipsoide mit klar getrennten Grauwertstufen sind
#   viel einfacher als echte Anatomie mit Partialvolumeneffekten, Bias-Feldern und
#   interindividueller Variabilität. Die Zahl 6,6 ist eine **obere Schranke** für das, was auf
#   echten Daten zu erwarten wäre.
# * **Ein Modell, eine Architektur, ein Trainingslauf.** Die Ergebnisse hängen von der Architektur
#   ab (vier `MaxPool3D`-Stufen erzeugen das Gittermuster) und von einem einzigen Seed. Für eine
#   belastbare Aussage bräuchte man mehrere Seeds und Architekturvarianten mit Streuungsangaben.
# * **Nur LRP.** Ohne Vergleich gegen Gradient × Input, Integrated Gradients, Grad-CAM oder
#   Occlusion weiß man nicht, ob 6,6 für ein Attributionsverfahren gut oder mittelmäßig ist.
# * **12 Testphantome** in der Hauptauswertung, 8 im Regelvergleich. Für Signifikanzaussagen zu
#   wenig.
#
# ### Die naheliegenden nächsten Schritte
#
# In der Reihenfolge, in der sie den größten Erkenntnisgewinn pro Aufwand bringen:
#
# 1. **Kontrollstruktur einbauen.** Dasselbe Experiment mit einer zweiten, gleich großen Struktur
#    an anderer Stelle (der Hippocampus, Labels 17 und 53, ist der natürliche Kandidat). Zeigt LRP
#    dort dieselbe Anreicherung, war der Effekt geometrisch. Das ist die wichtigste noch fehlende
#    Kontrolle.
# 2. **Ein echtes Modell trainieren, das Thalamusvolumen vorhersagt.** Auf den IXI-Daten mit den
#    FastSurfer-Volumina als Zielwert. Damit fällt der Realismus-Vorbehalt weg, und man kann Ratio
#    auf Phantomen gegen Ratio auf echten Daten vergleichen.
# 3. **Verfahren vergleichen.** Die Metriken aus Abschnitt 6 sind methodenagnostisch. Ein Balken
#    pro Attributionsverfahren, dieselben Bilder — daraus wird eine publizierbare Aussage.
# 4. **Perturbationsmaße ergänzen.** *Pixel flipping* innerhalb und außerhalb des Thalamus. Bricht
#    die Vorhersage nur beim Löschen im Thalamus ein, ist das ein von der Maske unabhängiger
#    Beleg.
# 5. **Mehrere Seeds, Fehlerbalken.** Fünf Trainingsläufe, fünf Ratio-Werte, Standardabweichung
#    berichten. Ohne Streuungsangabe ist „6,6" eine Anekdote.
# 6. **Aggregation über eine Kohorte.** Auf echten, MNI-registrierten Daten die Relevanzkarten
#    mitteln und regionenweise auswerten (wie in `Explain_brain_age_predictions`, Abschnitt 11),
#    diesmal aber mit einer Rangliste normiert auf die Regionsgröße.
#
# ### Weiterführende Quellen
#
# | Thema | Quelle |
# |---|---|
# | LRP-Grundlagen | Bach et al. (2015), *PLoS ONE* — die Originalarbeit |
# | Composite-Strategien | Montavon et al. (2019), *Explainable AI*, Springer LNCS 11700 |
# | Randomisierungstest | Adebayo et al. (2018), *Sanity Checks for Saliency Maps*, NeurIPS |
# | Lokalisationsmetriken | Arras et al. (2022), *CLEVR-XAI*, *Information Fusion* |
# | Perturbationsmaße | Rong et al. (2022), *ROAD*, ICML |
# | SFCN / Brain Age | Peng et al. (2021), *Medical Image Analysis* |
# | FastSurfer | Henschel et al. (2020), *NeuroImage* |
# | Repo-intern | [`doc/LRP_Fraunhofer_HHI_Video_Zusammenfassung.md`](../../doc/LRP_Fraunhofer_HHI_Video_Zusammenfassung.md), [`doc/IXI_Datenlayout.md`](../../doc/IXI_Datenlayout.md) |
#
# ---
#
# ### Erzeugte Dateien
#
# Die Zelle unten listet alles auf, was dieses Notebook unter
# `output/notebooks/<notebook-name>/` abgelegt hat.
#
# [↑ zum Anfang](#top)

# %%
print(f"Ausgabeordner: {target_dir}\n")
for path in sorted(target_dir.rglob("*")):
    if path.is_file():
        size = path.stat().st_size
        unit = f"{size / 1e6:7.2f} MB" if size > 1e6 else f"{size / 1e3:7.1f} kB"
        print(f"  {unit}  {path.relative_to(target_dir)}")

