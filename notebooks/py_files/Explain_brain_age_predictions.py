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
# # Brain Age erklären: LRP-Heatmaps für ein 3D-CNN
#
# ## Worum geht es in diesem Notebook?
#
# Wir nehmen ein **fertig trainiertes neuronales Netz**, das aus einem 3D-MRT-Bild des Kopfes
# das Alter einer Person schätzt, und stellen ihm die Frage: **„Woran hast du das festgemacht?"**
#
# Die Antwort bekommen wir in Form einer *Heatmap* — einer Karte, die jedem einzelnen Voxel
# (3D-Bildpunkt) einen Zahlenwert zuweist: **wie stark hat dieser Bildpunkt zur Vorhersage
# beigetragen?** Rot = hat die Vorhersage nach oben getrieben, Blau = hat sie nach unten gezogen.
#
# Das Verfahren dafür heißt **Layer-wise Relevance Propagation (LRP)**.
#
# ## Der größere Kontext: warum „Brain Age"?
#
# Das Gehirn verändert sich mit dem Alter systematisch: die graue Substanz wird dünner, die
# Ventrikel (flüssigkeitsgefüllte Hohlräume) werden größer, Furchen werden breiter. Ein CNN kann
# aus diesen Mustern das Alter erstaunlich genau schätzen (publizierte Modelle erreichen einen
# mittleren Fehler von ca. 3–4 Jahren).
#
# Interessant ist dabei nicht die Schätzung selbst — das echte Alter kennt man ja —, sondern der
# **Fehler**, das sogenannte *Brain Age Delta* oder *Brain Age Gap*:
#
# $$\Delta = \hat{y}_{\text{geschätzt}} - y_{\text{tatsächlich}}$$
#
# Ein positives $\Delta$ heißt: „Das Gehirn sieht älter aus, als die Person ist." In der
# Neurowissenschaft wird dieses $\Delta$ als möglicher Biomarker diskutiert — erhöhte Werte
# wurden u. a. mit neurodegenerativen Erkrankungen, Diabetes und erhöhter Mortalität in
# Verbindung gebracht.
#
# ## Der größere Kontext: warum XAI?
#
# Damit so ein Biomarker klinisch brauchbar wird, reicht „das Modell ist meistens richtig" nicht
# aus. Der Kernsatz der XAI-Community lautet:
#
# > *„Just because a model is right doesn't mean it got there for the right reason."*
#
# Ein Netz könnte das Alter auch an einem **Artefakt** ablesen — an der Kopfform, an
# Scanner-Rauschen, am Rand des Schädelstrippings, an einem Klinik-spezifischen Detail. Solche
# Fälle heißen „Clever-Hans-Prädiktoren". Erklärungsverfahren wie LRP machen genau das sichtbar.
#
# ## LRP in drei Sätzen
#
# 1. **Observe** — ein normaler Vorwärtsdurchlauf liefert die Vorhersage und merkt sich, wie
#    stark jedes Neuron aktiviert war.
# 2. **Redistribute** — der Ausgabewert wird Schicht für Schicht **rückwärts** verteilt, jeweils
#    proportional dazu, wie stark ein Neuron zur Aktivierung des nachfolgenden beigetragen hat.
# 3. **Reveal** — am Ende landet die gesamte „Relevanz" auf den Eingabevoxeln und ergibt die
#    Heatmap.
#
# Die zentrale Eigenschaft dabei ist die **Erhaltung (Conservation)**: LRP erzeugt und vernichtet
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
#   MRT-Bilder (NIfTI)                        FastSurfer-Segmentierung
#         │                                            │
#         ▼                                            │
#   [ SFCN-CNN ]  ──►  vorhergesagtes Alter            │
#         │                                            │
#         ▼                                            ▼
#   [ LRP rückwärts ] ──► Relevanz je Voxel ──► Relevanz je Hirnregion
#         │                                            │
#         ├─► Einzel-Heatmap                           └─► Relevanz vs. Alter (Scatter)
#         ├─► Gruppenmittel-Heatmap
#         ├─► GIF über die Kohorte
#         └─► Ähnlichkeitsmatrix der Erklärungen
# ```
#
# ## ⚠️ Wichtiger Hinweis zu diesem konkreten Lauf
#
# Die im Repository mitgelieferten Beispieldaten umfassen nur **10 Datensätze** (davon ein
# Subjekt doppelt und eines ohne Altersangabe), und das Modell erreicht darauf einen mittleren
# Fehler von **19,5 Jahren** statt der üblichen 3–4 Jahre. Alle Ergebnisse hier sind deshalb eine
# **methodische Demonstration der Pipeline**, keine wissenschaftliche Aussage über das Altern des
# Gehirns. An den passenden Stellen weisen die Abschnitte darauf hin.
#
# ---
#
# <a id="toc"></a>
# ## Inhaltsverzeichnis
#
# | # | Abschnitt | Inhalt |
# |---|---|---|
# | 1 | [Setup und Projektpfade](#sec-01) | Repository finden, Ausgabeordner anlegen |
# | 2 | [Daten laden](#sec-02) | NIfTI-Dataset, Preprocessing, Batch-Generator |
# | 3 | [Modell laden und vorhersagen](#sec-03) | SFCN-Architektur, Brain-Age-Gewichte |
# | 4 | [Encoder-Repräsentation](#sec-04) | 64-dimensionaler Merkmalsvektor pro Gehirn |
# | 5 | [Brain Age Delta](#sec-05) | Vorhersagefehler bewerten |
# | 6 | [LRP-Erklärer bauen](#sec-06) | Regeln, Strategie, Erhaltungs-Diagnose, erste Heatmap |
# | 7 | [Sanity-Check der ReLU-Regel](#sec-07) | Fließt Relevanz durch inaktive Neuronen? |
# | 8 | [Erklärungen für alle Subjekte](#sec-08) | Von einem Fall zur Kohorte |
# | 9 | [Mittlere Erklärung](#sec-09) | Gruppen-Heatmap |
# | 10 | [Einzelfall vs. Gruppenmittel](#sec-10) | Differenzkarte |
# | 11 | [Relevanz je Hirnregion](#sec-11) | FastSurfer-Atlas, Voxel → Anatomie |
# | 12 | [Relevanz gegen Alter](#sec-12) | Scatterplots je Region |
# | 13 | [Animation über die Kohorte](#sec-13) | GIF, nach Vorhersage sortiert |
# | 14 | [Ähnlichkeit der Erklärungen](#sec-14) | Kosinus-Ähnlichkeit |
# | 15 | [Ähnlichkeitsmatrix](#sec-15) | Heatmap nach Alter sortiert |
# | 16 | [Relevanzsummen prüfen](#sec-16) | Erhaltungseigenschaft als Debugging-Werkzeug |
# | 17 | [Auswertung eines Subjekts](#sec-17) | Regionen-Ranking für einen Fall |
# | 18 | [Fazit und Fallstricke](#sec-18) | Was man mitnehmen sollte |
#
# **Hintergrunddokumente im Repo:** [`doc/LRP_Fraunhofer_HHI_Video_Zusammenfassung.md`](../../doc/LRP_Fraunhofer_HHI_Video_Zusammenfassung.md)
# (LRP-Theorie) und [`doc/IXI_Datenlayout.md`](../../doc/IXI_Datenlayout.md) (wie die Daten auf
# der Platte liegen müssen).

# %% [markdown]
# <a id="sec-01"></a>
# ## 1. Setup und Projektpfade
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Die Zelle sucht ausgehend vom aktuellen Arbeitsverzeichnis nach oben das **Wurzelverzeichnis
# des Repositories** (erkennbar an `pyproject.toml` oder am Ordner `explainability/`) und hängt
# es an `sys.path`. Erst dadurch sind die beiden lokalen Pakete `explainability` (der
# LRP-Code) und `pyment` (Modelle und Daten-Loader) importierbar, egal aus welchem Ordner der
# Kernel gestartet wurde.
#
# Danach wird der Zielordner für erzeugte Dateien (`demo.gif`, `sorted_correlations.png`)
# angelegt.
#
# ### Einordnung
#
# Klingt nach Kleinkram, ist aber der häufigste Grund, warum ein Notebook auf einer anderen
# Maschine nicht läuft. Diese Datei ist über **jupytext** als `py:percent`-Skript mit dem
# `.ipynb` gekoppelt (siehe Kopf der Datei) — dadurch bleibt der Code diff-bar in Git, während
# man trotzdem interaktiv arbeiten kann. Zellgrenzen sind die `# %%`-Marker.

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

target_dir = repo_root / 'output' / 'notebooks' / 'Explain_brain_age_predictions'
target_dir.mkdir(parents=True, exist_ok=True)

print(f'Zielordner ist: {target_dir}')

# %% [markdown]
# <a id="sec-02"></a>
# ## 2. Daten laden: IXI-MRTs als Batch-Generator
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Drei Bausteine werden zusammengesteckt:
#
# | Objekt | Aufgabe |
# |---|---|
# | `NiftiDataset.from_folder(image_folder, target='age')` | verknüpft die Dateien in `cropped/images/*.nii.gz` mit der Spalte `age` aus `cropped/labels.csv` |
# | `NiftiPreprocessor(sigma=255.)` | skaliert die Voxelwerte |
# | `AsyncNiftiGenerator(...)` | liefert Batches à 4 Bilder und lädt die nächsten schon im Hintergrund (8 Threads) |
#
# Das Preprocessing besteht hier aus **genau einer Operation**:
#
# $$X \;\leftarrow\; \frac{X}{\sigma}, \qquad \sigma = 255$$
#
# Die Bilder liegen als 8-Bit-Intensitäten (0–255) vor und landen damit im Bereich $[0, 1]$.
# Es findet **kein** Resampling und keine z-Standardisierung statt.
#
# ### Was die Daten mitbringen müssen
#
# | Eigenschaft | Wert | Grund |
# |---|---|---|
# | Shape | `(167, 212, 160)` | fest verdrahtete Eingabegröße des Modells |
# | Auflösung | 1 mm isotrop | so wurde trainiert |
# | Referenzraum | MNI152, 6-DOF-registriert, skull-stripped | Trainings-Preprocessing |
# | Wertebereich | 0–255 | passt zu `sigma=255` |
#
# „Skull-stripped" heißt: Schädel, Haut und Augen sind entfernt, außerhalb des Gehirns steht
# exakt 0. Diese Nullen werden später noch wichtig (Abschnitt 6, Maskierung).
#
# ### Ausgabe dieser Zelle
#
# Neben TensorFlow-Startmeldungen erscheint:
#
# ```text
# WARNING - Skipping sub-638: Missing labels
# ```
#
# Für `sub-638` gibt es zwar ein Bild, aber keine Zeile in `labels.csv` — der Datensatz wird
# still übersprungen. Umgekehrt steht `sub-554` **zweimal** in der CSV; dieses Duplikat taucht
# später in den Ergebnissen wieder auf (Abschnitte 15 und 16) und ist ein hübscher unfreiwilliger
# Selbsttest.
#
# ### Einordnung
#
# Dass das Preprocessing **bitgenau zum Training passen muss**, ist bei vortrainierten Modellen
# die häufigste Fehlerquelle. Weicht es ab, sieht das Modell eine andere Datenverteilung als im
# Training (*domain shift*) — die Vorhersagen werden systematisch falsch, ohne dass eine
# Fehlermeldung erscheint. Genau das ist hier vermutlich passiert (siehe Abschnitt 5).

# %%
from pyment.data import NiftiDataset, AsyncNiftiGenerator
from pyment.data.preprocessors import NiftiPreprocessor

ixi_folder = os.path.join(repo_root, 'data', 'mri', 'ixi')
#ixi_folder = os.path.join(os.path.expanduser('~/pCloudDrive/media'), 'data', 'neuro-science','IXI', 'for_xai')
image_folder = os.path.join(ixi_folder, 'cropped')
project_folder = os.path.join(os.path.expanduser('~'), 'projects', '')
dataset = NiftiDataset.from_folder(image_folder, target='age')
preprocessor = NiftiPreprocessor(sigma=255.)
generator = AsyncNiftiGenerator(
    dataset=dataset,
    preprocessor=preprocessor,
    batch_size=4,
    threads=8
)

# %% [markdown]
# <a id="sec-03"></a>
# ## 3. Modell laden und Alter vorhersagen
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# `RegressionSFCN(weights='brain-age')` lädt das **Simple Fully Convolutional Network** von
# Peng et al. (2021) mit vortrainierten Brain-Age-Gewichten. Die Gewichtsdatei
# (`regression_sfcn_reg_2025_weights.h5`) wird beim ersten Aufruf nach `output/pyment/models/`
# heruntergeladen. `model.predict(generator)` schickt anschließend alle Batches durch das Netz.
#
# ### Die Architektur
#
# Das Netz ist bewusst schlicht: fünf identisch aufgebaute Faltungsblöcke, die das Volumen
# jeweils halbieren und die Kanalzahl erhöhen.
#
# | Index | Schicht | Ausgabe |
# |---|---|---|
# | 0–1 | Input + Reshape | `(167, 212, 160, 1)` |
# | 2–5 | Block 1: `Conv3D(32)` → BatchNorm → ReLU → `MaxPool3D(2)` | halbe Kantenlänge |
# | 6–9 | Block 2: `Conv3D(64)` → … | ↓ |
# | 10–13 | Block 3: `Conv3D(128)` → … | ↓ |
# | 14–17 | Block 4: `Conv3D(256)` → … | ↓ |
# | 18–21 | Block 5: `Conv3D(256)` → … | ↓ |
# | 22–24 | `Conv3D(64, 1×1×1)` → BatchNorm → ReLU | Kanalmischung |
# | **25** | **`GlobalAveragePooling3D`** | **Vektor der Länge 64** |
# | 26 | Dropout | |
# | 27 | `Dense(1)` | eine Zahl |
# | 28–29 | ReLU + Add | Begrenzung auf $[3, 95]$ Jahre |
#
# „Fully convolutional" heißt: bis Index 24 gibt es **keine** vollverbundene Schicht. Erst das
# Global Average Pooling faltet das ganze Restvolumen zu 64 Zahlen zusammen, und eine einzige
# `Dense`-Schicht macht daraus das Alter. Das hält die Parameterzahl klein — bei 3D-Bildern mit
# ~5,7 Millionen Voxeln pro Fall ein entscheidender Punkt, weil medizinische Datensätze selten
# mehr als ein paar tausend Fälle umfassen.
#
# ### Einordnung
#
# Für die spätere Erklärung sind zwei Dinge relevant:
#
# * Es gibt **nur ein Ausgabeneuron** (Regression statt Klassifikation). Bei LRP wählen wir also
#   nicht „welche Klasse erkläre ich?", sondern erklären schlicht den Zahlenwert.
# * Die letzten beiden Schichten (`ReLU` + `Add`) sind ein nachträgliches **Clipping** auf einen
#   plausiblen Altersbereich. Diese Konstruktion wird uns in Abschnitt 6 bei der
#   Relevanz-Erhaltung noch begegnen.

# %%
from pyment.models import RegressionSFCN

model = RegressionSFCN(weights='brain-age')

predictions = model.predict(generator)

# %% [markdown]
# <a id="sec-04"></a>
# ## 4. Encoder-Repräsentation: das Gehirn als 64 Zahlen
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Statt das Netz bis zum Ende laufen zu lassen, wird es bei **Schicht 25** abgeschnitten — dem
# `GlobalAveragePooling3D`. Ein neues Keras-`Model` mit demselben Input, aber diesem
# Zwischenoutput, liefert pro Gehirn einen **Merkmalsvektor der Länge 64**:
#
# $$\text{Bild} \in \mathbb{R}^{167 \times 212 \times 160} \;\longrightarrow\; e \in \mathbb{R}^{64}$$
#
# Dieser Punkt im Netz heißt *Bottleneck* oder *Embedding*. Alles, was das Modell über ein
# Gehirn „weiß", muss durch diese 64 Zahlen hindurch — die letzte `Dense`-Schicht sieht nichts
# anderes mehr.
#
# ### Einordnung
#
# Embeddings sind der Standardweg, ein vortrainiertes Netz weiterzuverwenden:
#
# * **Transfer Learning** — einen neuen Klassifikator (z. B. Diagnose ja/nein) auf den 64 Zahlen
#   trainieren, statt das ganze 3D-CNN neu zu trainieren.
# * **Ähnlichkeitssuche / Clustering** — welche Gehirne liegen im Merkmalsraum nahe beieinander?
# * **Visualisierung** — UMAP oder t-SNE auf den Embeddings.
#
# In diesem Notebook ist das ein **Nebenstrang**: `encodings` wird berechnet, aber später nicht
# weiterverwendet. Die Zelle zeigt lediglich, wie man an Zwischenrepräsentationen kommt — genau
# dieselbe `Model(input, zwischen_output)`-Technik nutzt Abschnitt 6, um die Relevanz an
# einzelnen Stellen des LRP-Graphen abzugreifen.

# %%
from pyment.models import Model

encoder = Model(model.input, model.layers[25].output)

encodings = encoder.predict(generator)

# %% [markdown]
# <a id="sec-05"></a>
# ## 5. Brain Age Delta: wie gut ist das Modell überhaupt?
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Fälle ohne Altersangabe (`NaN`) werden herausgefiltert, dann wird die Differenz gebildet:
#
# $$\Delta_i = \hat{y}_i - y_i \qquad\text{und}\qquad \text{MAE} = \frac{1}{n}\sum_{i=1}^{n} |\Delta_i|$$
#
# Beachte: die Variable heißt `delta`, ausgegeben wird aber der **mittlere absolute Fehler
# (MAE)**. Durch den Betrag geht die Richtung verloren — man sieht nicht mehr, ob das Modell
# systematisch zu jung oder zu alt schätzt. Für den Biomarker „Brain Age Gap" wäre gerade das
# Vorzeichen die interessante Größe.
#
# ### Ausgabe dieser Zelle
#
# ```text
# Brain age delta: 19.5
# ```
#
# ### Interpretation — und eine wichtige Warnung
#
# **19,5 Jahre mittlerer Fehler ist sehr viel.** Publizierte SFCN-Modelle liegen bei 3–4 Jahren.
# Die Vorhersagen in diesem Lauf liegen alle bei etwa 22 Jahren, während die tatsächlichen Alter
# von 23,5 bis 70,1 Jahren reichen — das Modell antwortet also nahezu konstant. Plausible
# Ursachen:
#
# 1. **Nur 10 Datensätze** — jede Kennzahl ist hier extrem instabil.
# 2. **Preprocessing-Abweichung** — wenn Intensitätsskala, Registrierung oder Crop nicht exakt
#    dem Trainings-Preprocessing entsprechen, kollabiert die Vorhersage typischerweise zum
#    Mittelwert der Trainingsverteilung.
# 3. **Reihenfolge** — `predictions` kommt aus dem Generator, `ages` aus `dataset.y`. Stimmen die
#    Reihenfolgen nicht überein, vergleicht man Äpfel mit Birnen.
#
# ### Einordnung: darf man ein schlechtes Modell erklären?
#
# Ja — man muss nur wissen, was die Erklärung aussagt. LRP beantwortet **immer** die Frage
# *„Welche Voxel haben diese konkrete Ausgabe erzeugt?"*, unabhängig davon, ob die Ausgabe
# richtig ist. Die Heatmaps in diesem Notebook sind also korrekte Aussagen **über das Modell**,
# aber sie erlauben keine Aussage **über das Altern des Gehirns**. Für biologische Schlüsse
# bräuchte man erst ein Modell, das auf diesen Daten funktioniert.
#
# Genau das ist übrigens ein Standard-Einsatzzweck von XAI: Heatmaps eines schlecht
# funktionierenden Modells verraten oft, *woran* es scheitert.

# %%
import numpy as np


ages = dataset.y
predictions = predictions.squeeze()
predictions = predictions[np.where(~np.isnan(ages))]
ages = ages[np.where(~np.isnan(ages))]
delta = predictions - ages
print(f'Brain age delta: {round(np.mean(np.abs(delta)), 2)}')

# %% [markdown]
# <a id="sec-06"></a>
# ## 6. Den LRP-Erklärer bauen — die zentrale Zelle
#
# [↑ Inhaltsverzeichnis](#toc)
#
# Diese Zelle macht drei Dinge auf einmal: sie **konfiguriert** LRP, sie **prüft** das Ergebnis
# schichtweise, und sie **zeichnet** die erste Heatmap. Wir gehen die drei Teile einzeln durch.
#
# ---
#
# ### 6.1 Wie LRP hier technisch funktioniert
#
# `LRP(model, layer=..., idx=0, strategy=...)` baut aus dem trainierten Netz ein **zweites Keras-
# Modell**, das den Rückwärtspfad explizit als Schichten enthält. Man ruft es auf wie ein
# normales Modell — nur dass hinten keine Zahl, sondern ein komplettes Relevanzvolumen in der
# Größe des Eingabebildes herauskommt.
#
# * `layer=len(model.layers)-1` — welche Schicht erklärt wird (hier: die letzte, also die
#   Vorhersage).
# * `idx=0` — welches Ausgabeneuron. Bei Regression gibt es nur eines. Bei einem Klassifikator
#   würde man hier die Klasse wählen, und dieselbe Eingabe ergäbe für verschiedene Klassen
#   verschiedene Heatmaps.
#
# Die Initialisierung setzt die Relevanz des Zielneurons auf seinen Aktivierungswert und alle
# anderen auf 0:
#
# $$R^{(L)}_c = a_c, \qquad R^{(L)}_{k \neq c} = 0$$
#
# ---
#
# ### 6.2 Die LRP-Regeln — und warum es mehrere gibt
#
# Die Grundregel für eine gewichtete Schicht (Dense oder Conv) lautet: der Beitrag von Neuron $j$
# zu Neuron $k$ ist $z_{jk} = a_j w_{jk}$, und die Relevanz wird proportional dazu zurückverteilt:
#
# $$R_j \;=\; \sum_k \frac{a_j w_{jk}}{\sum_{j'} a_{j'} w_{j'k}} \, R_k$$
#
# In der reinen Form ist diese Regel numerisch instabil: steht im Nenner fast 0, explodiert der
# Bruch. Deshalb gibt es Varianten:
#
# | Regel | Formel | Wirkung | Wo einsetzen |
# |---|---|---|---|
# | **$\varepsilon$** | $z \leftarrow z + \varepsilon\,\mathrm{sign}(z)$ | stabilisiert den Nenner, dämpft schwache Beiträge | obere Schichten |
# | **$\alpha\beta$** | $R_j = \sum_k \left(\alpha \frac{(a_j w_{jk})^+}{\sum (a\,w)^+} - \beta \frac{(a_j w_{jk})^-}{\sum (a\,w)^-}\right) R_k$ | trennt verstärkende von hemmenden Pfaden, mit $\alpha - \beta = 1$ | mittlere Schichten |
# | **flat** | $a \leftarrow 1,\; w \leftarrow 1$ | verteilt Relevanz gleichmäßig über das rezeptive Feld | unterste Schichten |
# | **ReLU** | $R_{\text{in}} = R_{\text{out}}$ | Identität, Relevanz fließt unverändert durch | Aktivierungen |
# | **MaxPool** | *winner-takes-all* | die gesamte Relevanz geht an das Maximum im Fenster | Pooling |
# | **AvgPool** | gleichmäßige Rückverteilung | | Pooling |
#
# Warum unterschiedliche Regeln in unterschiedlichen Tiefen? Das ist die sogenannte
# **Composite-Strategie** (Montavon et al., 2019). Die Begründung:
#
# * **Oben** (nah am Ausgang) codieren wenige Neuronen abstrakte Konzepte — hier will man
#   Relevanz eher konzentrieren und Rauschen unterdrücken → $\varepsilon$.
# * **In der Mitte** will man saubere, positive Zuordnungen → $\alpha\beta$ mit $\alpha=2, \beta=1$
#   (positive Beiträge doppelt gewichtet, negative einfach abgezogen).
# * **Unten** (nah am Bild) sind einzelne Voxel fast bedeutungslos; die reine z-Regel erzeugt dort
#   extrem verrauschte, „salz-und-pfeffer"-artige Karten. Die flat-Regel glättet, indem sie die
#   Relevanz gleichmäßig über das rezeptive Feld schmiert.
#
# Im Code:
#
# ```python
# strategy = LRPStrategy(layers=[
#     {'flat': True},                    # Conv Block 1  (nächste am Bild)
#     {'flat': True},                    # Conv Block 2
#     {'alpha': 2, 'beta': 1},           # Conv Block 3
#     {'alpha': 2, 'beta': 1},           # Conv Block 4
#     {'alpha': 2, 'beta': 1},           # Conv Block 5
#     {'alpha': 2, 'beta': 1},           # Conv 1×1×1 (top)
#     {'epsilon': 0.25}                  # Dense(1)     (nächste am Output)
# ])
# ```
#
# Die Liste hat genau **sieben** Einträge, weil das Netz sieben Schichten mit Gewichten hat
# (6 × `Conv3D` + 1 × `Dense`). Die Reihenfolge ist **Input → Output**. Pooling-, ReLU- und
# BatchNorm-Schichten bekommen keinen Eintrag; sie haben feste Regeln. (BatchNorm wird beim Bau
# des Erklärers vorher in die Faltung hineingerechnet.)
#
# ---
#
# ### 6.3 Die Diagnose-Schleife: bleibt Relevanz erhalten?
#
# Die Schleife über `range(29, len(lrp.layers))` baut für jede Schicht des Rückwärtspfads ein
# eigenes Teilmodell und gibt `np.sum` der dortigen Relevanz aus. Das ist ein direkter Test der
# **Erhaltungseigenschaft** $\sum_j R_j = \sum_k R_k$. Die Ausgabe (gekürzt):
#
# | Index | Schicht | $\sum R$ | Kommentar |
# |---|---|---|---|
# | 29 | `Add` (Alters-Clipping) | 22,14 | ≈ die Vorhersage (22,16) |
# | 31–32 | `AddLRP`, `ReLULRP` | 22,14 | erhalten |
# | **33** | **`DenseLRP`** ($\varepsilon$) | **9,89** | **großer Sprung nach unten** |
# | 35–37 | Pooling, ReLU, BatchNorm | 9,89 | erhalten |
# | 38–55 | `Conv3DLRP` / `MaxPoolingLRP` ($\alpha\beta$) | 9,89–9,90 | erhalten bis auf Rundung |
# | 58 | `Conv3DLRP` Block 1 (flat) | 10,07 | leichter Anstieg |
# | 60 | Input | 10,07 | Endergebnis |
#
# **Warum der Sprung von 22,14 auf 9,89?** Die `Dense`-Schicht hat einen **Bias**. Im Code wird
# der Nenner um den Bias erweitert:
#
# $$R \;\leftarrow\; R \cdot \frac{z}{z + b}$$
#
# Der Bias ist ein konstanter Summand, der **nicht vom Bild abhängt** — er kann also keinem Voxel
# zugeordnet werden. Die auf ihn entfallende Relevanz (hier gut die Hälfte) wird schlicht
# verworfen. Anschaulich: von den ~22 vorhergesagten Jahren erklärt das Bild rund 10; der Rest
# ist der „Grundton" des Modells. Auch das Alters-Clipping (`restrict_add`) addiert eine
# Konstante, die nicht aus dem Bild stammt.
#
# **Warum steigt es bei Schicht 58 leicht an?** Die flat-Regel setzt $a \leftarrow 1$ und
# $w \leftarrow 1$ und ist damit nicht mehr streng erhaltend. Dieselbe Eigenschaft hat noch eine
# zweite, wichtigere Nebenwirkung — siehe nächster Punkt.
#
# ---
#
# ### 6.4 Die Maskierung — warum sie unverzichtbar ist
#
# ```python
# mask = np.zeros(X[0].shape)
# mask[np.where(X[0] != 0)] = 1
# explanations = explanations * mask
# ```
#
# Außerhalb des Gehirns ist das Bild exakt 0. Bei der z-Regel wäre dort automatisch
# $R = a \cdot c = 0$. Die **flat-Regel ignoriert aber die Aktivierung** ($a \leftarrow 1$) und
# verteilt Relevanz deshalb auch auf reine Hintergrundvoxel. Ohne Maske hätte man leuchtende
# Bereiche in der Luft neben dem Kopf.
#
# Die Zahlen bestätigen das: vor der Maskierung liegt die Gesamtrelevanz bei 10,07, nach der
# Maskierung nur noch bei ~6,9 (siehe Abschnitt 16). Rund ein Drittel saß im Hintergrund.
#
# Anschließend wird auf $[-1, 1]$ normiert (Division durch $\max|R|$), damit die Farbskala
# `seismic` symmetrisch um Null liegt: **rot = positiv, weiß = 0, blau = negativ**.
#
# ---
#
# ### 6.5 Welche Schnittbilder gezeigt werden
#
# ```python
# idx = np.unravel_index(np.argmax(np.abs(explanations)), explanations.shape)
# ```
#
# Gesucht wird das **Voxel mit der größten absoluten Relevanz**; um dessen Koordinate herum
# werden je acht benachbarte Schichten in allen drei Standardebenen gezeigt:
#
# | Zeile | Inhalt |
# |---|---|
# | 1 | **sagittal** (Seitenansicht), MRT |
# | 2 | sagittal, LRP-Heatmap |
# | 3 | **koronal** (Frontalschnitt), MRT |
# | 4 | koronal, LRP-Heatmap |
# | 5 | **axial** (Draufsicht), MRT |
# | 6 | axial, LRP-Heatmap |
#
# *Randnotiz:* Das `plt.show()` am Zellenende steht **nach** dem `break` und wird nie erreicht —
# Jupyter zeigt die Figur trotzdem, weil es am Ende jeder Zelle offene Matplotlib-Figuren
# automatisch rendert.
#
# ---
#
# ### 6.6 Was man auf der Abbildung sieht
#
# **Rot dominiert fast vollständig.** Blau taucht nur vereinzelt in kleinen Flecken auf. Zwei
# Gründe: erstens ist $\alpha=2, \beta=1$ bewusst positiv gewichtet, zweitens ist die erklärte
# Größe ein *positives* Alter — nahezu jeder Hirnbereich „spricht dafür", dass die Person nicht
# null Jahre alt ist.
#
# **Die Relevanz ist breit gestreut, nicht fokussiert.** Anders als bei einem Katzenbild, wo LRP
# scharf auf Ohren und Schnauze zeigt, ist hier fast das gesamte Hirnparenchym rötlich. Das ist
# für Brain Age plausibel — Altern ist ein globaler Prozess —, kann aber auch heißen, dass die
# Erklärung wenig spezifisch ist.
#
# **Ein sichtbares Gitter-/Schachbrettmuster.** Die Heatmaps zeigen ein regelmäßiges Raster aus
# helleren und dunkleren Punkten. Das ist ein **Artefakt der Rückverteilung**, kein anatomisches
# Signal: Fünf `MaxPool3D`-Schichten mit Schrittweite 2 bedeuten, dass die Relevanz beim
# Rückweg über die *winner-takes-all*-Regel immer wieder auf einzelne Voxel eines
# 2×2×2-Fensters konzentriert wird. Nach fünf Halbierungen entspricht ein Voxel der tiefsten
# Ebene einem 32×32×32-Block im Bild — daher das grobe Raster.
#
# **Die sagittalen Schnitte (Zeile 1) wirken klein und randständig.** Kein Fehler: der
# Schichtindex kommt aus `argmax`, das relevanteste Voxel liegt hier weit lateral, also am
# seitlichen Rand des Gehirns.
#
# ### Einordnung
#
# Für eine erste Sichtprüfung stellt man sich drei Fragen: (1) Liegt die Relevanz überhaupt *im*
# Gehirn? (2) Konzentriert sie sich auf Strukturen, die biologisch etwas mit dem Zielwert zu tun
# haben? (3) Sieht man verdächtige Artefakte am Bildrand, an der Schädelstripping-Kante oder in
# der Luft? Punkt (3) ist der klassische Clever-Hans-Test.

# %%
from plotly.figure_factory import create_distplot

import matplotlib.pyplot as plt

from explainability import LRP, LRPStrategy

alpha=2
beta=1

strategy = LRPStrategy(
    layers=[
        {'flat': True},
        {'flat': True},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'epsilon': 0.25}
    ])

lrp = LRP(model, layer=len(model.layers)-1, idx=0, strategy=strategy)

from pyment.models import Model
import numpy as np
m1 = Model(lrp.input, lrp.layers[57].output)
m2 = Model(lrp.input, lrp.layers[56].output)
m3 = Model(lrp.input, lrp.layers[57].input)

for X, y in generator:
    preds = model.predict(X)
    print(preds[0])
    
    for i in range(29, len(lrp.layers)):
        m = Model(lrp.input, lrp.layers[i].output)
        print(f'{i}: {lrp.layers[i]}')
        e = m.predict(X[:1])[0]
        print(np.sum(e))
        
    explanations = lrp(X[:1])[0].numpy()
    mask = np.zeros(X[0].shape)
    mask[np.where(X[0] != 0)] = 1
    explanations = explanations * mask
    explanations = explanations / np.amax(np.abs(explanations))
    idx = np.argmax(np.abs(explanations))
    idx = np.unravel_index(idx, explanations.shape)
    
    fig, ax = plt.subplots(6, 8, figsize=(15, 15))
    
    for i in range(-4, 4):
        ax[0][i+4].imshow(np.rot90(X[0,idx[0]+i]), cmap='Greys_r')
        ax[0][i+4].axis('off')
        ax[1][i+4].imshow(np.rot90(explanations[idx[0]+i]), cmap='seismic', clim=(-1, 1))
        ax[1][i+4].axis('off')
        ax[2][i+4].imshow(np.rot90(X[0,:,idx[1]+i]), cmap='Greys_r')
        ax[2][i+4].axis('off')
        ax[3][i+4].imshow(np.rot90(explanations[:,idx[1]+i]), cmap='seismic', clim=(-1, 1))
        ax[3][i+4].axis('off')
        ax[4][i+4].imshow(X[0,:,:,idx[2]+i], cmap='Greys_r')
        ax[4][i+4].axis('off')
        ax[5][i+4].imshow(explanations[:,:,idx[2]+i], cmap='seismic', clim=(-1, 1))
        ax[5][i+4].axis('off')

    break

    plt.show()

# %% [markdown]
# <a id="sec-07"></a>
# ## 7. Sanity-Check: fließt Relevanz durch inaktive Neuronen?
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Schicht 56 des Erklärer-Modells ist eine `ReLULRP`-Schicht nahe am Bildeingang. Ihr Eingang ist
# ein **Paar**: die im Vorwärtspass gemessene Aktivierung `a` und die von oben ankommende
# Relevanz `R`. Die Zelle zählt zwei Dinge:
#
# ```python
# print(len(np.where(a.flatten() <= 0)[0]))   # Neuronen, die im Forward-Pass inaktiv waren
# print(len(np.where(R.flatten() == 0)[0]))   # Positionen ohne Relevanz
# ```
#
# ### Ausgabe
#
# ```text
# 169939200    # a <= 0
# 161269088    # R == 0
# ```
#
# Bei 32 Kanälen auf dem vollen Gitter sind das insgesamt $167 \cdot 212 \cdot 160 \cdot 32
# \approx 181{,}3$ Mio. Positionen. Also: rund **94 % der Neuronen waren inaktiv**, aber nur
# 89 % tragen keine Relevanz. Die Differenz von **~8,7 Mio. Positionen** hat Relevanz, obwohl
# das zugehörige Neuron im Vorwärtspass gar nicht gefeuert hat.
#
# ### Interpretation
#
# Die Lehrbuch-Regel für ReLU lautet: *war das Neuron inaktiv, bekommt es 0 Relevanz; war es
# aktiv, fließt die Relevanz unverändert durch.* In dieser Implementierung ist nur die zweite
# Hälfte umgesetzt — `ReLULRP.call` gibt schlicht `R` zurück, die strengere Variante
# `tf.where(a > 0, R, 0)` steht auskommentiert im Code.
#
# Bei der normalen z-Regel ist das unkritisch, weil die *darunterliegende* Faltungsschicht
# ohnehin mit $R_j = a_j \cdot c_j$ multipliziert und inaktive Positionen damit von selbst auf 0
# fallen. Sobald aber die **flat-Regel** im Spiel ist (die $a$ durch 1 ersetzt), greift diese
# Selbstkorrektur nicht mehr — und genau darunter liegen hier die beiden flat-Schichten. Das
# passt zu dem, was Abschnitt 6.4 zeigt: ein spürbarer Teil der Relevanz landet außerhalb des
# aktiven Signals und muss nachträglich weggemaskt werden.
#
# ### Einordnung
#
# Solche Zählungen sind unspektakulär, aber wertvoll. Erklärungsverfahren liefern **immer** ein
# buntes Bild — auch dann, wenn die Implementierung eine Regel falsch anwendet oder ein Tensor
# vertauscht ist. Ein Bild sieht man das nicht an. Deshalb gehören zu jeder XAI-Pipeline
# numerische Selbsttests: Erhaltungssummen (Abschnitte 6.3 und 16), Nullstellen-Checks wie
# dieser, und im Idealfall ein Vergleich mit einer Referenzimplementierung (das Notebook
# `Explain_2D_VGG_predictions` in diesem Repo vergleicht z. B. gegen *iNNvestigate*).

# %%
m3 = Model(lrp.input, lrp.layers[56].input)

a, R = m3.predict(X[:1])
a = a[0]
R = R[0]

print(len(np.where(a.flatten() <= 0)[0]))
print(len(np.where(R.flatten() == 0)[0]))

# %% [markdown]
# <a id="sec-08"></a>
# ## 8. Erklärungen für alle Subjekte berechnen
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Bisher haben wir einen einzelnen Fall erklärt. Jetzt läuft dieselbe Prozedur über die gesamte
# Kohorte. Pro Subjekt werden vier Dinge gespeichert:
#
# | Variable | Inhalt |
# |---|---|
# | `images` | das vorverarbeitete MRT |
# | `labels` | das tatsächliche Alter |
# | `predictions` | das vorhergesagte Alter |
# | `all_explanations` | das maskierte Relevanzvolumen |
#
# `generator.reset()` setzt den Generator zurück, damit wieder bei Batch 0 begonnen wird — die
# vorherigen Zellen haben ihn schon einmal durchlaufen.
#
# Die Erklärungen werden einzeln (`np.expand_dims(image, 0)`) berechnet, nicht batchweise. Das
# ist langsamer, aber speicherschonend: ein Relevanzvolumen ist genauso groß wie das Eingabebild,
# und im Rückwärtspfad liegen zusätzlich alle Zwischenaktivierungen im Speicher.
#
# ### Zwei Stolpersteine im Code
#
# * `all_explanations[i*4 + j]` verdrahtet die **Batchgröße 4** fest. Ändert man `batch_size`
#   oben, schreibt diese Zeile an die falschen Indizes.
# * `labels` kommt als Liste von Arrays der Form `(1,)`. Das abschließende
#   `np.asarray(labels).reshape(-1)` macht daraus einen flachen Vektor — sonst würde
#   `np.argsort` später über eine Achse der Länge 1 sortieren und wirkungslos bleiben.
#
# ### Einordnung
#
# Hier findet der Übergang von der **lokalen** zur **globalen** Erklärung statt. Eine einzelne
# Heatmap beantwortet „warum dieser Fall?"; erst viele Heatmaps zusammen beantworten „was macht
# dieses Modell eigentlich generell?". Die folgenden Abschnitte sind alle Aggregationen über
# `all_explanations`: Mittelwert (9), Differenz zum Mittelwert (10), Zerlegung nach Anatomie
# (11–12), Animation (13) und paarweise Ähnlichkeit (14–15).

# %%
from tqdm import tqdm


images = []
labels = []
predictions = []
all_explanations = np.zeros((len(generator),) +  X[0].shape)

generator.reset()

for i, (X, y) in tqdm(enumerate(generator), total=generator.batches):
    for j in range(len(X)):
        image = X[j]
        labels.append(y[j])
        predictions.append(model.predict(np.expand_dims(image, 0))[0])
        expl = lrp.predict(np.expand_dims(image, 0))[0]
        mask = np.zeros(image.shape)
        mask[np.where(image != 0)] = 1
        expl = expl * mask
        all_explanations[i*4 + j] = expl
        images.append(image)

# Der Generator liefert y je Sample als Array der Form (1,). Flach gemacht sind die
# Labels sortierbar (np.argsort würde sonst über die Achse der Länge 1 sortieren) und
# lassen sich direkt formatieren.
labels = np.asarray(labels).reshape(-1)

# %% [markdown]
# <a id="sec-09"></a>
# ## 9. Die mittlere Erklärung: was macht das Modell *immer*?
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Voxelweiser Mittelwert über alle Subjekte, anschließend Normierung auf $[-1, 1]$:
#
# $$\bar{R}_v = \frac{1}{N}\sum_{i=1}^{N} R^{(i)}_v \qquad\text{dann}\qquad \bar{R} \leftarrow \frac{\bar{R}}{\max_v |\bar{R}_v|}$$
#
# Gezeigt werden diesmal nicht die Schichten um das relevanteste Voxel, sondern die **Mitte des
# Volumens** (`shape / 2`) — bei einer Gruppenkarte gibt es kein individuelles Maximum, das man
# ansteuern wollte.
#
# Dass so ein Mittelwert überhaupt sinnvoll ist, liegt am Preprocessing: alle Bilder sind in den
# **MNI152-Standardraum registriert**. Voxel $(i,j,k)$ bezeichnet damit in jedem Bild
# (näherungsweise) denselben anatomischen Ort. Ohne diese räumliche Normalisierung wäre die
# Mittelung sinnlos.
#
# ### Was man auf der Abbildung sieht
#
# **Die Karte ist deutlich glatter als die Einzelfallkarte in Abschnitt 6.** Genau das erwartet
# man: das subjektspezifische Rauschen mittelt sich weg ($\propto 1/\sqrt{N}$), das systematische
# Muster bleibt stehen. Man erkennt jetzt zusammenhängende rote Gebiete statt eines
# Punkterasters.
#
# **Räumliche Schwerpunkte.** Die stärkste Relevanz liegt in den kortikalen Randbereichen und im
# unteren Bereich der koronalen Schnitte (Temporallappen / Kleinhirnregion). Um die Ventrikel
# und entlang einiger Furchen erscheinen dünne **blaue Säume** — dort spricht die lokale
# Bildinformation gegen ein höheres Alter. Beides ist biologisch anschlussfähig: Kortexdicke und
# Ventrikelgröße sind die klassischen Alterungsmarker im MRT. Bei $N = 10$ und einem Modell mit
# 19,5 Jahren Fehler sollte man daraus allerdings **nichts** ableiten.
#
# **Das Streifenmuster über die Schichten hinweg.** In jeder Heatmap-Zeile wechseln sich kräftige
# und blasse Schichten regelmäßig ab. Das ist wieder das Pooling-Artefakt aus Abschnitt 6.6, hier
# besonders gut sichtbar: die *winner-takes-all*-Rückverteilung der `MaxPool3D`-Schichten mit
# Schrittweite 2 bevorzugt systematisch jede zweite Schicht.
#
# **Ein Mismatch, den man kennen sollte.** Die anatomischen Zeilen (1, 3, 5) zeigen `X[0]` — das
# **erste Bild des letzten Batches**, das noch als Variable herumliegt —, während die
# Heatmap-Zeilen den Kohortenmittelwert zeigen. Sauberer wäre entweder ein
# Mittelwertbild (`np.mean(images, axis=0)`) oder eine MNI-Vorlage als Hintergrund. In der
# Praxis führt so ein Mismatch schnell zu Fehlinterpretationen, weil man rote Flecken einer
# Anatomie zuordnet, zu der sie gar nicht gehören.
#
# ### Einordnung
#
# Gemittelte Attributionskarten sind in der Neuro-XAI Standard, tragen aber eine
# Interpretationsfalle: Der Mittelwert zeigt, was das Modell **im Durchschnitt** tut. Wenn zwei
# Subgruppen gegensätzliche Strategien auslösen, hebt sich das im Mittel auf, und die Karte sieht
# harmlos aus. Deshalb kommt in Abschnitt 15 zusätzlich eine paarweise Ähnlichkeitsanalyse.

# %%
mean_explanation = np.mean(all_explanations, axis=0)
mean_explanation = mean_explanation / np.amax(np.abs(mean_explanation))
idx = (np.asarray(mean_explanation.shape) / 2).astype(int)

fig, ax = plt.subplots(6, 8, figsize=(15, 15))

for i in range(-4, 4):
    ax[0][i+4].imshow(np.rot90(X[0,idx[0]+i]), cmap='Greys_r')
    ax[0][i+4].axis('off')
    ax[1][i+4].imshow(np.rot90(mean_explanation[idx[0]+i]), cmap='seismic', clim=(-1, 1))
    ax[1][i+4].axis('off')
    ax[2][i+4].imshow(np.rot90(X[0,:,idx[1]+i]), cmap='Greys_r')
    ax[2][i+4].axis('off')
    ax[3][i+4].imshow(np.rot90(mean_explanation[:,idx[1]+i]), cmap='seismic', clim=(-1, 1))
    ax[3][i+4].axis('off')
    ax[4][i+4].imshow(X[0,:,:,idx[2]+i], cmap='Greys_r')
    ax[4][i+4].axis('off')
    ax[5][i+4].imshow(mean_explanation[:,:,idx[2]+i], cmap='seismic', clim=(-1, 1))
    ax[5][i+4].axis('off')

plt.show()

# %% [markdown]
# <a id="sec-10"></a>
# ## 10. Einzelfall gegen Gruppenmittel: die Differenzkarte
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Vier Einzelbilder, alle aus der sagittalen Schicht 80:
#
# 1. `im1` — die Erklärung des **ersten Subjekts**, einzeln auf $[-1,1]$ normiert
# 2. `im2` — die **mittlere** Erklärung, ebenfalls einzeln normiert
# 3. `im3 = im1 - im2` — die **Differenz**
# 4. `brain` — das zugehörige anatomische Bild
#
# Die Leitfrage: *Was ist an diesem einen Gehirn besonders — was erklärt das Modell hier anders
# als üblich?*
#
# ### Was man auf den vier Abbildungen sieht
#
# **Bild 1 (Einzelfall).** Deutlich verrauscht, rot und blau durchmischt. Kräftig rote Cluster
# im unteren Bereich (temporal/zerebellär), diffus bläuliche Zonen im mittleren und hinteren
# Bereich. Das grobe Punkteraster des Poolings ist gut sichtbar.
#
# **Bild 2 (Gruppenmittel).** Nahezu flächendeckend rot, glatt, mit kräftigem Schwerpunkt in der
# unteren Bildhälfte. Blau kommt praktisch nicht mehr vor.
#
# **Bild 3 (Differenz).** Überwiegend **blau** — und das ist bemerkenswert, weil eine Differenz
# eigentlich um Null herum streuen sollte. Die Ursache ist eine methodische Falle: beide Karten
# wurden **getrennt** auf ihr jeweiliges Maximum normiert. Die gemittelte Karte hat durch die
# Rauschmittelung eine viel gleichmäßigere Werteverteilung, ihr Maximum ist relativ zum
# Gesamtniveau kleiner — nach der Normierung liegt sie deshalb fast überall **höher** als die
# Einzelkarte. Die Differenz misst also überwiegend den Normierungsunterschied und nicht die
# individuelle Abweichung. Sinnvoller wäre eine **gemeinsame** Skala für beide Karten oder eine
# z-Standardisierung pro Karte.
#
# **Bild 4 (Anatomie): komplett schwarz.** Auch das ist kein Datenfehler, sondern die
# Farbskalierung. Nach dem Preprocessing $X \leftarrow X/255$ liegen die Intensitäten weit unter 1
# — mit `clim=(0, 1)` fällt praktisch alles auf den dunkelsten Farbwert. In Abschnitt 6 sahen die
# anatomischen Bilder normal aus, weil dort **kein** `clim` gesetzt war und Matplotlib
# automatisch auf `[min, max]` skaliert. Merkregel: bei `imshow` entscheidet die Skalierung
# darüber, ob man überhaupt etwas sieht.
#
# ### Einordnung
#
# Differenzkarten sind der Einstieg in die **personalisierte** Erklärung: nicht „worauf achtet
# das Modell allgemein", sondern „was ist an diesem Patienten auffällig". In der Praxis würde man
# dafür die individuelle Karte gegen ein **Normkollektiv** stellen und die Abweichung als z-Wert
# je Voxel ausdrücken:
#
# $$z_v = \frac{R^{(i)}_v - \bar{R}_v}{\mathrm{sd}_v}$$
#
# Damit hätte man eine Skala, die sagt, *wie ungewöhnlich* eine Abweichung ist — und wäre das
# Normierungsproblem aus Bild 3 los.

# %%
im1 = all_explanations[0][80]
im1 = im1 / np.amax(np.abs(im1))
im1 = np.rot90(im1)
plt.imshow(im1, cmap='seismic', clim=(-1, 1))
plt.show()

im2 = mean_explanation[80]
im2 = im2 / np.amax(np.abs(im2))
im2 = np.rot90(im2)
plt.imshow(im2, cmap='seismic', clim=(-1, 1))
plt.show()

im3 = im1 - im2
plt.imshow(im3, cmap='seismic', clim=(-1, 1))
plt.show()

brain = images[0][80]
brain = np.rot90(brain)
plt.imshow(brain, cmap='Greys_r', clim=(0, 1))
plt.show()

# %% [markdown]
# <a id="sec-11"></a>
# ## 11. Von Voxeln zu Hirnregionen: Auswertung mit dem FastSurfer-Atlas
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Warum dieser Schritt?
#
# Eine Voxel-Heatmap ist hübsch, aber schwer zu berichten. Man kann nicht schreiben „Voxel
# (83, 105, 79) war wichtig". Neurowissenschaftlich verwertbar wird es erst, wenn man sagen kann:
# *„Der Hippocampus trug 4 % der Relevanz."* Dafür braucht man eine **anatomische
# Parzellierung** — eine zweite 3D-Karte, die jedem Voxel eine Regionsnummer zuweist.
#
# Diese liefert **FastSurfer**, ein Deep-Learning-Nachbau der klassischen
# FreeSurfer-Segmentierung. Die Datei `aparc.DKTatlas+aseg.deep.mgz` enthält den DKT-Atlas:
# ca. 95 kortikale und subkortikale Strukturen, jede mit einer eigenen Ganzzahl codiert.
#
# ### Was passiert hier im Detail?
#
# Der Knackpunkt ist, dass Erklärung und Segmentierung auf **demselben Voxelgitter** liegen
# müssen. FastSurfer arbeitet im „conformed space" (256³, 1 mm, LIA-Orientierung), unsere Bilder
# sind 167×212×160. Der Code löst das so:
#
# 1. Bild und Relevanzvolumen mit `conform()` auf das 256³-Gitter bringen.
# 2. **Problem:** `conform()` erwartet ein Bild mit Wertebereich 0–255 vom Typ `uint8`.
#    Relevanzwerte sind aber kleine Fließkommazahlen mit Vorzeichen. Deshalb ein Roundtrip:
#
#    $$R_{\text{uint8}} = \left\lfloor 255 \cdot \frac{R - \min R}{\max(R - \min R)} \right\rfloor
#      \qquad\text{und zurück}\qquad
#      R \approx \frac{R_{\text{uint8}}}{255}\cdot \text{range} + \min R$$
#
# 3. Segmentierungsmaske laden und **je Region aufsummieren**:
#
#    $$R_{\text{Region}} = \sum_{v \in \text{Region}} R_v$$
#
# Gespeichert werden drei Dictionaries: `regions[region][id]` (Relevanzsumme),
# `sizes[region][id]` (Anzahl Voxel) und `totals[id]` (Gesamtsumme über das ganze Volumen).
#
# Laufzeit: ca. **35 Sekunden pro Subjekt**, fast ausschließlich für das `conform()`-Resampling
# und die Schleife über alle Regionen.
#
# ### ⚠️ Zwei Artefakte, die man kennen muss
#
# **1. Der uint8-Roundtrip verliert Information.** 256 Stufen für eine Verteilung, deren Werte
# stark um Null konzentriert sind, heißt: die meisten Voxel landen in derselben Stufe. Feine
# Unterschiede verschwinden. Zusätzlich interpoliert `conform()` beim Resampling und vermischt
# damit benachbarte Werte.
#
# **2. Das Padding bekommt einen negativen Wert — und dominiert `totals`.** Beim Umrechnen auf
# 256³ werden viele Voxel neu hinzugefügt, die im Original gar nicht existierten. Sie haben den
# uint8-Wert 0, und die Rückrechnung bildet 0 auf $\min R$ ab — also auf den **negativsten Wert
# der ganzen Karte**. Das betrifft rund
#
# $$256^3 - (167 \cdot 212 \cdot 160) \approx 16{,}8\,\text{Mio.} - 5{,}7\,\text{Mio.} \approx 11\,\text{Mio. Voxel}$$
#
# Die Folge: `totals[id]` misst überwiegend dieses künstliche Padding und wird **negativ**,
# obwohl die echte Relevanz im Gehirn positiv ist. Abschnitt 12 zeigt den direkten Beleg dafür.
# Die Relevanzsummen **innerhalb** echter Hirnregionen sind davon nicht betroffen — nur die
# Normierung durch `totals`.
#
# ### Einordnung
#
# Der Schritt „Attributionskarte × anatomischer Atlas" ist das Bindeglied zwischen Deep Learning
# und klassischer Neuroimaging-Statistik. Er macht Ergebnisse mit der bestehenden Literatur
# vergleichbar (die spricht in Regionsnamen, nicht in Voxelkoordinaten) und reduziert
# ~5,7 Mio. Zahlen auf ~95 — eine Größenordnung, mit der man tatsächlich Statistik betreiben
# kann.

# %%
import nibabel as nib

from collections import Counter
from copy import copy

from FastSurferCNN.data_loader.conform import conform


fastsurfer_folder = os.path.join(ixi_folder, 'fastsurfer')

regions = {}
sizes = {}
totals = {}

def colorize(mask: np.ndarray):
    colours = np.unique(mask.flatten())
    colours = [col for col in colours if col != 0.]
    cmap = plt.cm.get_cmap('gist_rainbow', len(colours))
    colourized = np.zeros(mask.shape + (4,))

    for colour in colours:
        colourized[np.where(mask == colour)] = cmap(colour)
    
    return colourized

for i in tqdm(range(len(all_explanations))):
    id = dataset.ids[i]
    
    if not (os.path.isfile(os.path.join(image_folder, 'images', f'{id}.nii.gz')) and \
            os.path.isfile(os.path.join(fastsurfer_folder, id, 'mri', 'aparc.DKTatlas+aseg.deep.mgz'))):
        continue
    
    image = nib.load(os.path.join(image_folder, 'images', f'{id}.nii.gz'))
    affine = copy(image.affine)
    header = copy(image.header)
    original_image = copy(image.get_fdata())
    image = conform(image)
    
    explanation = all_explanations[i]
    original_explanation = copy(explanation)
    min_value = np.amin(explanation)
    explanation = explanation - min_value
    max_value = np.amax(explanation)
    explanation = explanation / max_value
    explanation = explanation * 255
    explanation = explanation.astype(np.uint8)
    explanation = nib.Nifti1Image(explanation, affine=affine, header=header)
    explanation = conform(explanation)
    
    mask = nib.load(os.path.join(fastsurfer_folder, id, 'mri', 'aparc.DKTatlas+aseg.deep.mgz'))
    
    image = image.get_fdata()
    explanation = explanation.get_fdata()
    mask = mask.get_fdata()
    
    explanation = explanation / 255.
    explanation = explanation * max_value
    explanation = explanation + min_value
    
    totals[id] = np.sum(explanation)
    
    for region in np.unique(mask.flatten()):
        voxels = np.where(mask == region)
        
        if not region in regions:
            regions[region] = {}
            
        regions[region][id] = np.sum(explanation[voxels])
        
        if not region in sizes:
            sizes[region] = {}
            
        sizes[region][id] = len(voxels[0])

# %% [markdown]
# <a id="sec-12"></a>
# ## 12. Relevanz gegen Alter — ein Scatterplot je Hirnregion
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Zuerst werden die **Regionsnummern in Namen übersetzt**. `load_segmentation_labels()` liest
# dafür zwei Lookup-Tabellen aus dem Repo: die FastSurfer-LUT (nur 79 Einträge) und ergänzend die
# vollständige FreeSurfer-LUT. Für Nummern, die in keiner der beiden stehen, gibt es Fallbacks
# (`0.0 → 'Background'`, `2.0 → 'WM'`).
#
# Dann wird für **jede** Region eine Abbildung mit drei Scatterplots erzeugt (x-Achse immer das
# chronologische Alter):
#
# | Plot | Größe | Formel |
# |---|---|---|
# | links | Rohsumme | $R_{\text{Region}}$ |
# | mitte | subjektnormiert | $R_{\text{Region}} \,/\, \lvert R_{\text{total}} \rvert$ |
# | rechts | zusätzlich größennormiert | $R_{\text{Region}} \,/\, (\lvert R_{\text{total}} \rvert \cdot n_{\text{Voxel}})$ |
#
# Der Gedanke hinter den Normierungen: Der **linke** Plot ist nicht vergleichbar, weil Subjekte
# unterschiedlich viel Gesamtrelevanz haben. Der **mittlere** korrigiert das (Anteil statt
# Absolutwert). Der **rechte** korrigiert zusätzlich, dass große Regionen allein wegen ihrer
# Größe mehr Relevanz aufsammeln — er misst also **Relevanzdichte pro Voxel**.
#
# *Namensfalle im Code:* die Variable heißt `age_normalized_relevance`, normiert aber nicht nach
# Alter, sondern nach Gesamtrelevanz des Subjekts.
#
# ### Was man auf den Abbildungen sieht
#
# Ausgegeben werden **über 100 Abbildungen** — eine pro Atlas-Region. Vier repräsentative Fälle:
#
# **`Background`** — Rohwerte zwischen $-280$ und $-720$, subjektnormiert konstant bei
# $\approx -1{,}008$, größennormiert $\approx -6{,}5 \cdot 10^{-8}$. Das ist der **direkte Beleg
# für das Padding-Artefakt** aus Abschnitt 11:
#
# * Aus $-1{,}008 \,/\, (-6{,}5 \cdot 10^{-8}) \approx 1{,}55 \cdot 10^{7}$ Voxel folgt, dass die
#   „Region Hintergrund" fast das gesamte 256³-Gitter umfasst.
# * Der Quotient von fast genau $-1$ heißt: `totals` **besteht praktisch nur aus dem
#   Hintergrund**. Dass der Betrag leicht *über* 1 liegt, passt dazu, dass die positive Relevanz
#   im Gehirn den negativen Gesamtwert etwas verkleinert.
#
# **Konsequenz:** die mittleren und rechten Plots aller Regionen sind durch eine
# artefaktbehaftete Größe geteilt. Für Vergleiche *innerhalb* eines Subjekts ist das egal
# (konstanter Faktor), für Vergleiche *zwischen* Subjekten nicht.
#
# **`Left-Cerebral-White-Matter`** — Rohwerte 0,29–0,62, größennormiert $\approx 4 \cdot 10^{-9}$
# (entspricht ~267.000 Voxeln, plausibel für eine Hemisphäre weißer Substanz). Die Punkte
# streuen ohne erkennbaren Trend über das Alter.
#
# **`Left-Thalamus`** — Rohwerte 0,006–0,08 bei ~8.000 Voxeln (anatomisch stimmig). Der einzige
# 70-Jährige liegt in allen drei Plots am oberen Rand. Das *sieht* nach Alterseffekt aus, ist
# aber ein einzelner Punkt — bei $n = 10$ ohne jede Aussagekraft.
#
# **`Right-vessel`** — Werte im Bereich $10^{-5}$, wild streuend. Winzige Regionen liefern
# überwiegend Rauschen.
#
# ### ⚠️ Warum man hier nichts ableiten darf
#
# * **$n = 10$ Punkte**, davon zwei identisch (`sub-554` steht doppelt in `labels.csv`) — es sind
#   effektiv **9 Subjekte**.
# * **Über 100 Regionen** werden geplottet. Bei so vielen Vergleichen findet man rein zufällig
#   „Trends"; ohne Korrektur für multiples Testen (Bonferroni, FDR) ist jeder davon wertlos.
# * **Kein Test, keine Regressionsgerade, keine Achsenbeschriftung** — die Plots sind explorativ.
# * Der Filter `id != 'IXI237-Guys-1049-T1'` sollte einen Ausreißer entfernen; diese ID stammt aus
#   einem früheren Datensatz und existiert hier nicht mehr, die Zeile läuft also ins Leere.
#
# ### Einordnung
#
# So sieht die Rohform einer **Region-of-Interest-Analyse** aus. Der ausgereifte Weg wäre: eine
# Region pro Zeile, ein Subjekt pro Spalte, dann für jede Region eine lineare Regression
# $R_{\text{Region}} \sim \text{Alter} + \text{Geschlecht} + \text{Scanner}$, anschließend
# FDR-Korrektur über alle Regionen und Darstellung als sortiertes Balkendiagramm. Dafür braucht
# man allerdings Stichproben im dreistelligen Bereich.

# %%
import pandas as pd

from functools import reduce


def load_segmentation_labels() -> pd.DataFrame:
    """Maps the numeric labels of the segmentation to region names.

    Uses the lookup tables shipped with FastSurfer. Its own LUT does not cover
    every label the DKT model emits, so the FreeSurfer LUT fills the gaps.
    """
    config_folder = repo_root / 'FastSurferCNN' / 'config'

    fastsurfer = pd.read_csv(config_folder / 'FastSurfer_ColorLUT.tsv', sep='\t',
                             usecols=['ID', 'LabelName'])
    fastsurfer = fastsurfer.rename(columns={'ID': 'id', 'LabelName': 'name'})

    freesurfer = pd.read_csv(config_folder / 'FreeSurferColorLUT.txt', sep=r'\s+',
                             comment='#', header=None, usecols=[0, 1],
                             names=['id', 'name'])

    labels = pd.concat([fastsurfer, freesurfer])

    return labels.drop_duplicates(subset='id', keep='first').reset_index(drop=True)


fastsurfer_labels = load_segmentation_labels()

ids = dataset.ids
ages = dataset.y
idx = np.argsort(ages)
sorted_ids = np.asarray(ids)[idx]
sorted_ages = np.asarray(ages)[idx]
ages = {sorted_ids[i]: sorted_ages[i] for i in range(len(sorted_ids))}

for key in regions:
    names = fastsurfer_labels.loc[fastsurfer_labels['id'] == key, 'name'].values
    
    if len(names) == 0:
        if key == 0.0:
            name = 'Background'
        elif key == 2.0:
            name = 'WM'
        else:
            name = key
    else:
        name = names[0]
        
    region_ids = [id for id in sorted_ids if id in regions[key] and id != 'IXI237-Guys-1049-T1']
    region_ages = np.asarray([ages[id] for id in region_ids])
    region_relevance = np.asarray([regions[key][id] for id in region_ids])
    region_sizes = np.asarray([sizes[key][id] for id in region_ids])
    region_totals = np.asarray([totals[id] for id in region_ids])
    age_normalized_relevance = region_relevance / np.abs(region_totals)
    size_normalized_relevance = age_normalized_relevance / region_sizes
    
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(region_ages, region_relevance)
    ax[1].scatter(region_ages, age_normalized_relevance)
    ax[2].scatter(region_ages, size_normalized_relevance)
    fig.suptitle(name)
    
    plt.show()

# %% [markdown]
# <a id="sec-13"></a>
# ## 13. Animation: die Kohorte als Daumenkino
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Alle Subjekte werden **nach vorhergesagtem Hirnalter sortiert** und zu einem GIF
# zusammengesetzt. Jedes Einzelbild besteht aus sechs Kacheln:
#
# ```text
# ┌───────────┬───────────┬───────────┐
# │ sagittal  │  koronal  │   axial   │   ← MRT (Graustufen)
# ├───────────┼───────────┼───────────┤
# │ sagittal  │  koronal  │   axial   │   ← LRP (seismic)
# └───────────┴───────────┴───────────┘
#          + Text: "Age=…, brain age …"
# ```
#
# Technische Details:
#
# * **Feste Schnittpositionen** (84 / 106 / 80) statt individueller Maxima — nur so bleibt die
#   Anatomie über die Animation hinweg an derselben Stelle und das Auge kann vergleichen. Möglich
#   ist das wieder nur dank der MNI-Registrierung.
# * `pad_to_size(..., 212)` bringt alle drei Ebenen auf dasselbe quadratische Format.
# * Die Verschiebung `expl = expl / max|expl| + 0.5` ist der entscheidende Trick für die
#   Farbgebung: die `seismic`-Colormap erwartet Werte in $[0, 1]$ und ist bei **0,5 weiß**.
#   Relevanz 0 landet damit auf weiß, positive Werte auf rot, negative auf blau.
# * `duration=40` (Millisekunden pro Bild), `loop=0` (Endlosschleife).
#
# ### Was man sieht
#
# Das GIF wird als Datei nach `output/notebooks/Explain_brain_age_predictions/demo.gif`
# geschrieben und erscheint deshalb **nicht** in der HTML-Version des Notebooks — man muss es
# separat öffnen.
#
# Die Leitfrage beim Betrachten: *Wandert der rote Schwerpunkt systematisch, wenn man von jung
# nach alt durchläuft?* Falls ja, hätte man einen Hinweis darauf, dass das Modell für
# unterschiedliche Altersgruppen unterschiedliche Bildmerkmale nutzt. Bei 10 Bildern und der
# nahezu konstanten Vorhersage dieses Laufs ist damit allerdings nicht zu rechnen — Abschnitt 15
# prüft dieselbe Frage quantitativ.
#
# ### Einordnung
#
# Animationen sind ein unterschätztes Werkzeug in der Bild-XAI. Das menschliche
# Wahrnehmungssystem ist außergewöhnlich gut darin, **Bewegung und Veränderung** in einer
# Bildfolge zu erkennen — Unterschiede, die man beim Nebeneinanderlegen von zehn Standbildern
# übersieht, springen in einer Animation sofort ins Auge. Für die Kommunikation mit Ärztinnen und
# Ärzten ist ein sortiertes Daumenkino oft überzeugender als jede Statistik.

# %%
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple


def load_font(size: int = 20):
    """DejaVuSans wird von matplotlib mitgeliefert, ist also überall vorhanden."""
    path = Path(matplotlib.get_data_path()) / 'fonts' / 'ttf' / 'DejaVuSans.ttf'

    return ImageFont.truetype(path, size)

def pad_to_size(image, size: int = 212, value: Tuple = 0):
    vertical = size - image.shape[0]
    top = int(np.ceil(vertical / 2))
    bottom = vertical - top
    
    horizontal = size - image.shape[1]
    left = int(np.ceil(horizontal / 2))
    right = horizontal - left
    
    return np.pad(image, ((top, bottom), (left, right)), constant_values=value)

def concat_horizontal(i1, i2, color=(0, 0, 0)):
    dst = Image.new('RGB', (i1.width + i2.width, i1.height))
    dst.paste(i1, (0, 0))
    dst.paste(i2, (i1.width, 0))
    return dst

def concat_vertical(i1, i2):
    dst = Image.new('RGB', (i1.width, i1.height + i2.height))
    dst.paste(i1, (0, 0))
    dst.paste(i2, (0, i1.height))
    return dst

idx = np.argsort([pred[0] for pred in predictions])
sorted_labels = np.asarray(labels).reshape(-1)[idx]
sorted_predictions = [predictions[i] for i in idx]
sorted_images = [images[i] for i in idx]
sorted_explanations = [all_explanations[i] for i in idx]

sorted_bitmaps = []

for i in tqdm(range(len(images))):
    expl = sorted_explanations[i]
    expl = expl / np.amax(np.abs(expl))
    expl = expl + 0.5
    
    
    saggital_image = sorted_images[i][84]
    saggital_image = np.rot90(saggital_image)
    saggital_image = pad_to_size(saggital_image)
    saggital_explanations = expl[84]
    saggital_explanations = np.rot90(saggital_explanations)
    saggital_explanations = pad_to_size(saggital_explanations, value=0.5)
    saggital_image = Image.fromarray(np.uint8(cm.Greys_r(saggital_image)*255))
    saggital_explanations = Image.fromarray(np.uint8(cm.seismic(saggital_explanations)*255))
    
    coronal_image = sorted_images[i][:,106]
    coronal_image = np.rot90(coronal_image)
    coronal_image = pad_to_size(coronal_image)
    coronal_explanations = expl[:,106]
    coronal_explanations = np.rot90(coronal_explanations)
    coronal_explanations = pad_to_size(coronal_explanations, value=0.5)
    coronal_image = Image.fromarray(np.uint8(cm.Greys_r(coronal_image)*255))
    coronal_explanations = Image.fromarray(np.uint8(cm.seismic(coronal_explanations)*255))
    
    axial_image = sorted_images[i][:,:,80]
    axial_image = np.rot90(axial_image)
    axial_image = pad_to_size(axial_image)
    axial_explanations = expl[:,:,80]
    axial_explanations = np.rot90(axial_explanations)
    axial_explanations = pad_to_size(axial_explanations, value=0.5)
    axial_image = Image.fromarray(np.uint8(cm.Greys_r(axial_image)*255))
    axial_explanations = Image.fromarray(np.uint8(cm.seismic(axial_explanations)*255))
    
    brain_bitmap = concat_horizontal(concat_horizontal(saggital_image, coronal_image), axial_image)
    explanations_bitmap = concat_horizontal(concat_horizontal(saggital_explanations, coronal_explanations),
                                            axial_explanations)
    bitmap = concat_vertical(brain_bitmap, explanations_bitmap)
    
    draw = ImageDraw.Draw(bitmap)
    font = load_font(20)
    draw.text((180, 180),f'Age={sorted_labels[i]:.2f}, brain age {sorted_predictions[i][0]:.2f}', 
              (255,255,255), font=font)
    
    sorted_bitmaps.append(bitmap)
    
sorted_bitmaps[0].save(target_dir / 'demo.gif',
               save_all=True, append_images=sorted_bitmaps[1:], optimize=False, duration=40, loop=0)

# %% [markdown]
# <a id="sec-14"></a>
# ## 14. Wie ähnlich sind sich die Erklärungen?
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Für **jedes Paar** von Subjekten $(i, j)$ wird ein Ähnlichkeitsmaß zwischen den beiden
# Relevanzvolumina berechnet:
#
# $$s(a, b) \;=\; \frac{\sum_v a_v b_v}{\sqrt{\sum_v a_v^2 \cdot \sum_v b_v^2}}$$
#
# Die Funktion heißt im Code `correlate`, rechnet aber die **Kosinus-Ähnlichkeit** — der
# Unterschied zur Pearson-Korrelation ist, dass hier **nicht zentriert** wird (kein Abzug des
# Mittelwerts). Bei Karten, die fast ausschließlich positive Werte enthalten, macht das einen
# großen Unterschied: die Kosinus-Ähnlichkeit ist dann grundsätzlich hoch, weil beide Vektoren im
# selben Quadranten liegen.
#
# Anschaulich: die beiden Relevanzvolumina werden als Vektoren mit ~5,7 Millionen Komponenten
# aufgefasst, und $s$ ist der Kosinus des Winkels zwischen ihnen. 1 = identische Richtung,
# 0 = orthogonal.
#
# ### Zwei Anmerkungen zum Code
#
# * Die Division durch `np.abs(np.amax(...))` vor dem Aufruf ist **wirkungslos**: Kosinus-
#   Ähnlichkeit ist skaleninvariant, ein konstanter positiver Faktor kürzt sich heraus. (Zudem
#   wird `amax` und nicht `max(|·|)` verwendet — was bei einer überwiegend positiven Karte
#   dasselbe ist.)
# * Die Doppelschleife rechnet **alle** $n^2$ Paare, obwohl die Matrix symmetrisch ist und die
#   Diagonale bekannt (= 1). Bei $n = 10$ egal, bei $n = 500$ wären es 250.000 Vergleiche über je
#   5,7 Mio. Voxel. Effizient wäre, die Volumina einmal zu einer Matrix zu flachen, zeilenweise zu
#   normieren und ein einziges Matrixprodukt zu bilden.
#
# ### Einordnung
#
# Diese Matrix ist ein **Stabilitätsmaß für Erklärungen**. Zwei Extreme wären denkbar:
#
# * Alle Erklärungen fast identisch → das Modell nutzt für jeden eine Einheitsstrategie. Die
#   Erklärungen sind dann wenig personalisiert und tragen kaum Fallinformation.
# * Alle Erklärungen völlig verschieden → entweder erfasst das Modell echte individuelle
#   Unterschiede, oder die Erklärungen sind schlicht instabil (was ein bekanntes Problem
#   gradientenbasierter Verfahren ist).
#
# Die Wahrheit liegt üblicherweise dazwischen, und die Ähnlichkeitsmatrix quantifiziert, wo.

# %%
import tensorflow as tf
import time

def correlate(a, b):
    numerator = np.sum(a * b)
    sums = np.sum(a ** 2) * np.sum(b ** 2)
    denominator = np.sqrt(sums)
    
    return numerator / denominator

correlations = np.zeros((len(all_explanations), len(all_explanations)))

start = time.time()

for i in tqdm(range(len(all_explanations))):
    for j in range(len(all_explanations)):
        correlations[i,j] = correlate(
            all_explanations[i] / np.abs(np.amax(all_explanations[i])),
            all_explanations[j] / np.abs(np.amax(all_explanations[j]))
        )

# %% [markdown]
# <a id="sec-15"></a>
# ## 15. Die Ähnlichkeitsmatrix, nach Alter sortiert
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Die Matrix aus Abschnitt 14 wird **in beiden Dimensionen nach chronologischem Alter sortiert**
# (`correlations[idx][:, idx]`) und als Heatmap gezeichnet. Die Achsenbeschriftungen zeigen an
# sechs Stützstellen das Alter. Gespeichert wird die Abbildung als `sorted_correlations.png`.
#
# Die Sortierung ist der eigentliche Kniff: **wenn** das Modell für junge und alte Gehirne
# unterschiedliche Bildmerkmale nutzt, müssten entlang der Diagonale **Blöcke** entstehen —
# quadratische dunkle Bereiche für Altersgruppen mit ähnlicher Erklärungsstrategie und hellere
# Bereiche dazwischen.
#
# ### Was man auf der Abbildung sieht
#
# **Die Diagonale ist dunkelblau (Wert 1,0).** Jede Erklärung ist mit sich selbst identisch — der
# eingebaute Sanity-Check der Abbildung.
#
# **Die Werte außerhalb der Diagonale liegen fast alle bei 0,55–0,70.** Die Erklärungen der
# verschiedenen Subjekte sind sich also **erheblich ähnlich**. Zwei Gründe, die man
# auseinanderhalten muss:
#
# 1. *Methodisch:* Ohne Zentrierung und bei nahezu ausschließlich positiven Karten ist eine hohe
#    Kosinus-Ähnlichkeit fast unvermeidlich. Zusätzlich sind alle Bilder MNI-registriert, teilen
#    also dieselbe grobe Gehirnschablone.
# 2. *Inhaltlich:* Das Modell sagt in diesem Lauf für alle Subjekte etwa dasselbe Alter voraus
#    (~22 Jahre) — dass es dabei auch überall dasselbe anschaut, ist konsistent.
#
# **Kein Blockmuster, kein Verlauf entlang des Alters.** In dieser Mini-Stichprobe gibt es keinen
# Hinweis auf altersabhängige Erklärungsstrategien. Bei $n = 9$ ist das aber auch kein Befund,
# sondern schlicht fehlende Aussagekraft.
#
# **Der dunkle 2×2-Block unten rechts ist kein Ergebnis, sondern ein Datenfehler.** Die beiden
# ältesten „Subjekte" (beide 70,11 Jahre) sind dieselbe Person: `sub-554` steht **zweimal** in
# `labels.csv`. Deshalb ist die Ähnlichkeit dort exakt 1,0 — genau wie auf der Diagonale.
#
# Das ist zugleich ein unfreiwilliger, aber ausgezeichneter **Validierungstest der gesamten
# Pipeline**: Zweimal dasselbe Bild durch Modell und Erklärer geschickt ergibt eine
# Selbstähnlichkeit von 1,0. Wäre irgendwo Nichtdeterminismus, eine Indexverschiebung oder ein
# Reihenfolgefehler im Spiel, stünde dort ein kleinerer Wert.
#
# ### Einordnung
#
# Für eine belastbare Analyse würde man drei Dinge ändern: die Karten vor dem Vergleich
# **zentrieren** (echte Pearson-Korrelation) oder ein strukturbewusstes Maß wie **SSIM** nutzen;
# nur Voxel **innerhalb** der Hirnmaske einbeziehen; und die Werte gegen eine **Nullverteilung**
# aus permutierten Karten stellen, damit „0,65" überhaupt eine Bedeutung bekommt.

# %%
ages = np.asarray(labels).reshape(-1)
idx = np.argsort(ages)
sorted_correlations = correlations[idx][:,idx]

ticks = np.unique(np.linspace(0, len(idx) - 1, min(len(idx), 6)).astype(int))
tick_labels = [round(float(ages[idx[i]]), 2) for i in ticks]

fig = plt.figure(figsize=(15, 15))
heatmap = plt.imshow(sorted_correlations, cmap='YlGnBu', clim=(0, 1))
plt.colorbar(heatmap)
plt.xticks(ticks, tick_labels)
plt.xlabel('Chronological age')
plt.yticks(ticks, tick_labels)
plt.ylabel('Chronological age')
plt.savefig(target_dir / 'sorted_correlations.png')
plt.show()

# %% [markdown]
# <a id="sec-16"></a>
# ## 16. Relevanzsummen prüfen: die Erhaltungseigenschaft als Debugging-Werkzeug
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Für jedes Subjekt wird die **Gesamtrelevanz** ausgegeben:
#
# $$\sum_v R_v$$
#
# ### Ausgabe
#
# ```text
# 6.879884539817796
# 6.068933617645226
# 7.177748305179130
# 6.903809133454236
# 7.325760428379754
# 6.765166770366263
# 6.611716371654372   ←┐
# 6.611716371654372   ←┘  zweimal derselbe Wert
# 6.914225717935516
# 6.950479158470840
# ```
#
# ### Interpretation
#
# **Die Werte sind stabil.** Alle liegen zwischen 6,07 und 7,33 — die Pipeline verhält sich über
# die Subjekte hinweg konsistent, kein Fall bricht aus.
#
# **Zwei exakt identische Werte** an Position 7 und 8: wieder das doppelt eingetragene Subjekt
# `sub-554`. Dass die Übereinstimmung bis zur letzten Nachkommastelle reicht, bestätigt, dass die
# Berechnung deterministisch ist.
#
# **Die Bilanz gegen die Modellausgabe.** Es lohnt sich, die Kette einmal komplett
# nachzuvollziehen:
#
# | Stufe | Wert | Was passiert ist |
# |---|---|---|
# | Modellvorhersage | ~22,2 | die erklärte Größe |
# | Relevanz nach `DenseLRP` | 9,89 | Bias und Alters-Offset absorbieren gut die Hälfte |
# | Relevanz am Bildeingang | 10,07 | flat-Regel ist nicht streng erhaltend |
# | **nach Maskierung** | **~6,9** | Relevanz im Hintergrund wurde entfernt |
#
# Die Erhaltungseigenschaft $\sum_j R_j = \sum_k R_k$ gilt hier also **nicht global**, und zwar
# aus vier nachvollziehbaren Gründen: (1) der Bias der `Dense`-Schicht gehört zu keinem Voxel,
# (2) das Alters-Clipping addiert eine bildunabhängige Konstante, (3) die flat- und
# $\alpha\beta$-Regeln sind Approximationen, (4) die Maskierung verwirft bewusst Relevanz.
#
# Wichtig ist nicht, dass die Summe erhalten bleibt, sondern dass man **für jede Abweichung eine
# Erklärung hat**. Eine unerklärte Abweichung wäre ein Bug.
#
# ### Einordnung
#
# Erhaltungssummen sind das billigste und wirksamste Diagnosewerkzeug bei LRP. Sie kosten eine
# Zeile Code und fangen genau die Fehler, die man auf einer Heatmap nicht sieht: vertauschte
# Achsen, falsch verdrahtete Skip-Connections, eine Regel, die auf die falsche Schicht angewendet
# wurde. Faustregel: *Erst wenn die Summen erklärbar sind, darf man die Bilder interpretieren.*

# %%
for x in all_explanations:
    print(np.sum(x))

# %% [markdown]
# <a id="sec-17"></a>
# ## 17. Auswertung eines einzelnen Subjekts
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was passiert hier?
#
# Die Zelle sammelt für **ein bestimmtes Subjekt** die Relevanzsummen über alle Hirnregionen —
# der Einstieg in eine fallbezogene Auswertung: *Welche Regionen trugen bei diesem Menschen die
# Vorhersage?*
#
# In diesem Lauf läuft sie allerdings leer. Die ID `IXI012-HH-1211-T1` ist hart eingetragen und
# stammt aus dem ursprünglichen IXI-Datensatz; die mitgelieferten Beispieldaten verwenden
# IDs im Format `sub-XXX`. Die List Comprehension filtert auf `if ... in regions[region]`,
# findet nichts und liefert eine leere Liste — ohne Fehlermeldung. Ausgegeben wird nur der
# `print` in der letzten Zeile: `6.879884539817796`, die Gesamtrelevanz des **ersten** Subjekts
# (identisch mit dem ersten Wert aus Abschnitt 16).
#
# Robuster wäre `subject_id = dataset.ids[0]` statt einer festen Zeichenkette.
#
# ### Einordnung
#
# Der Schritt von der Kohorten- zur Einzelfallauswertung ist der, auf den es klinisch ankommt.
# Ein vollständiger Ablauf sähe so aus:
#
# 1. Relevanzanteil je Region berechnen: $p_r = R_r / \sum_{r'} R_{r'}$ (nur über Hirnregionen,
#    ohne den Hintergrund — siehe die Artefaktwarnung in Abschnitt 11).
# 2. Absteigend sortieren und die zehn stärksten Regionen als Balkendiagramm zeigen.
# 3. Gegen ein Normkollektiv vergleichen: Weicht dieses Profil vom Erwartungswert Gleichaltriger
#    ab?
#
# Das Ergebnis wäre ein Satz wie: *„Die Vorhersage von 67 Jahren stützt sich zu 12 % auf den
# rechten Hippocampus — deutlich mehr als bei Gleichaltrigen üblich."* Das ist eine Aussage, mit
# der eine Ärztin arbeiten kann. Eine nackte Zahl „67" ist es nicht.

# %%
subject_regions = [regions[region]['IXI012-HH-1211-T1'] for region in regions \
                   if 'IXI012-HH-1211-T1' in regions[region]]
print(np.sum(all_explanations[0]))

# %% [markdown]
# <a id="sec-18"></a>
# ## 18. Fazit, Fallstricke und wie es weitergeht
#
# [↑ Inhaltsverzeichnis](#toc)
#
# ### Was dieses Notebook gezeigt hat
#
# Eine vollständige XAI-Pipeline für medizinische 3D-Bilddaten, von der rohen NIfTI-Datei bis zur
# regionenweisen Auswertung:
#
# 1. **Vorhersage** — ein vortrainiertes 3D-CNN schätzt das Hirnalter (Abschnitte 2–5).
# 2. **Erklärung** — LRP mit einer Composite-Strategie propagiert die Vorhersage rückwärts bis
#    auf die Voxelebene (Abschnitt 6).
# 3. **Validierung** — Erhaltungssummen und Nullstellen-Checks prüfen, ob die Erklärung
#    überhaupt korrekt gerechnet ist (Abschnitte 7 und 16).
# 4. **Aggregation** — vom Einzelfall zur Kohorte: Mittelwertkarte, Differenzkarte, Animation
#    (Abschnitte 8–10, 13).
# 5. **Anatomische Zuordnung** — Voxelrelevanz wird über einen Atlas in Regionsnamen übersetzt
#    (Abschnitte 11–12).
# 6. **Stabilitätsanalyse** — wie ähnlich sind sich die Erklärungen verschiedener Fälle
#    (Abschnitte 14–15).
#
# Diese sechs Stufen sind auf praktisch jedes bildgebende Deep-Learning-Problem übertragbar.
#
# ### Die wichtigsten Fallstricke — gesammelt
#
# | # | Fallstrick | Abschnitt |
# |---|---|---|
# | 1 | Das Modell erreicht 19,5 Jahre Fehler; die Erklärungen erklären ein schlecht funktionierendes Modell | 5 |
# | 2 | Die flat-Regel verteilt Relevanz auch in den Hintergrund → Maskierung zwingend | 6.4 |
# | 3 | Bias und Alters-Offset absorbieren die Hälfte der Relevanz → Erhaltung gilt nicht global | 6.3, 16 |
# | 4 | Das Schachbrettmuster in den Heatmaps ist ein Pooling-Artefakt, kein Signal | 6.6, 9 |
# | 5 | Anatomie-Hintergrund und Mittelwertkarte stammen von verschiedenen Bildern | 9 |
# | 6 | Getrennt normierte Karten darf man nicht direkt voneinander abziehen | 10 |
# | 7 | `clim=(0,1)` auf durch 255 geteilten Bildern ergibt ein schwarzes Bild | 10 |
# | 8 | Der uint8-Roundtrip vor `conform()` quantisiert die Relevanz grob | 11 |
# | 9 | Padding-Voxel bekommen $\min R$ → `totals` ist artefaktdominiert | 11, 12 |
# | 10 | Über 100 Regionen ohne Korrektur für multiples Testen | 12 |
# | 11 | `sub-554` steht doppelt in `labels.csv`; `sub-638` hat kein Label | 2, 15, 16 |
# | 12 | Hart codierte Batchgröße 4 und feste IXI-IDs | 8, 12, 17 |
#
# Punkte 11 und 12 sind auch in [`doc/IXI_Datenlayout.md`](../../doc/IXI_Datenlayout.md)
# dokumentiert.
#
# ### Naheliegende nächste Schritte
#
# * **Erst das Modell reparieren.** Solange die Vorhersage bei ~22 Jahren festklebt, sind alle
#   Heatmaps Aussagen über einen Fehlerzustand. Verdächtig ist das Preprocessing — ein Vergleich
#   der Intensitätshistogramme mit den Trainingsdaten wäre der erste Test.
# * **Mehr Daten.** Für Aussagen auf Regionsebene braucht es Stichproben im dreistelligen Bereich.
# * **Regeln variieren.** Wie stabil ist die Heatmap gegenüber $\alpha, \beta, \varepsilon$? Eine
#   Erklärung, die bei jeder Parameterwahl anders aussieht, ist keine.
# * **Gegen andere Verfahren prüfen.** Integrated Gradients, Grad-CAM, Occlusion. Wenn mehrere
#   unabhängige Methoden auf dieselben Regionen zeigen, steigt das Vertrauen erheblich.
# * **Quantitativ evaluieren.** Der *Pixel-Flipping*-Test (die relevantesten Voxel schrittweise
#   entfernen und beobachten, wie schnell die Vorhersage einbricht) macht die Qualität einer
#   Erklärung messbar statt nur ansehnlich.
#
# ### Weiterführende Quellen
#
# * Bach et al. (2015), *On Pixel-Wise Explanations …* — das LRP-Originalpaper:
#   [doi:10.1371/journal.pone.0130140](https://doi.org/10.1371/journal.pone.0130140)
# * Montavon et al. (2019), *Layer-Wise Relevance Propagation: An Overview* — Composite-Regeln
#   und praktische Empfehlungen
# * Peng et al. (2021), *Accurate brain age prediction with lightweight deep neural networks* —
#   das SFCN-Modell
# * Interaktive Demo: [lrpserver.hhi.fraunhofer.de](https://lrpserver.hhi.fraunhofer.de/)
# * Im Repo: [`doc/LRP_Fraunhofer_HHI_Video_Zusammenfassung.md`](../../doc/LRP_Fraunhofer_HHI_Video_Zusammenfassung.md)
#   und das Notebook `Explain_2D_VGG_predictions` (LRP auf einem 2D-Bild, mit Vergleich gegen
#   *iNNvestigate*)
#
# [↑ Zum Anfang](#top)

# %%

# %%
