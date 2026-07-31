# IXI-Datenlayout für `Explain_brain_age_predictions`

Diese Übersicht beschreibt, wie die IXI-Daten auf der Platte organisiert sein müssen, damit
`notebooks/py_files/Explain_brain_age_predictions.py` (bzw. das zugehörige `.ipynb`) durchläuft,
und welche Dateien aus einem FreeSurfer-`recon-all`- bzw. FastSurfer-Lauf dafür gebraucht werden.

---

## 1. Wurzelverzeichnis

Das Notebook setzt den Datenpfad in der ersten Datenzelle:

```python
ixi_folder = os.path.join(os.path.expanduser('~/pCloudDrive/media'),
                          'data', 'neuro-science', 'IXI', 'for_xai')
image_folder = os.path.join(ixi_folder, 'cropped')       # Bilder + Labels
fastsurfer_folder = os.path.join(ixi_folder, 'fastsurfer')  # Segmentierungen
```

Alles Weitere hängt an diesen drei Pfaden. Zusätzlich gibt es zwei **hart codierte Pfade außerhalb**
von `ixi_folder`, siehe Abschnitt 6.

---

## 2. Soll-Verzeichnisstruktur

```
<ixi_folder>/                                  # z.B. .../IXI/for_xai
├── cropped/
│   ├── images/
│   │   ├── <id>.nii.gz                        # 167 × 212 × 160, 1 mm iso, Werte 0–255
│   │   ├── <id>.nii.gz
│   │   └── ...
│   └── labels.csv                             # Spalten: id, age
└── fastsurfer/
    └── <id>/                                  # ein Ordner pro Subjekt, Name == id
        └── mri/
            └── aparc.DKTatlas+aseg.deep.mgz   # 256³, 1 mm, LIA
```

Wichtig: **`<id>` muss in allen drei Orten identisch sein** — Dateiname in `cropped/images/`,
Wert in der Spalte `id` von `labels.csv` und Ordnername unter `fastsurfer/`. Die ID wird von
`pyment` aus dem Dateinamen als Teil vor dem ersten Punkt abgeleitet
(`sub-002.nii.gz` → `sub-002`, `IXI002-Guys-0828-T1.nii.gz` → `IXI002-Guys-0828-T1`).

### 2.1 `cropped/images/`

Wird von `NiftiDataset.from_folder(image_folder, target='age')` gelesen; die Defaults der
Methode sind `images='images'`, `labels='labels.csv'`, `suffix='nii.gz'` — deshalb muss der
Unterordner genau `images` heißen und die Endung genau `.nii.gz` sein.

| Eigenschaft | Anforderung | Grund |
|---|---|---|
| Shape | `(167, 212, 160)` | Default-`input_shape` von `RegressionSFCN` |
| Voxelgröße | 1 mm isotrop | Modell wurde so trainiert |
| Referenzraum | MNI152, 6-DOF-registriert, skull-stripped | Trainings-Preprocessing |
| Wertebereich | 0–255 | `NiftiPreprocessor(sigma=255.)` teilt nur durch 255 |
| Datentyp | float (wird via `get_fdata()` gelesen) | — |

Es findet **keine** Resampling- oder Normalisierungs-Korrektur zur Laufzeit statt. Ein Bild mit
abweichender Shape führt direkt zu einem Keras-Shape-Fehler.

### 2.2 `cropped/labels.csv`

```csv
id,age
sub-002,35.800136892539356
sub-012,38.78165639972622
```

* Spalte `id` ist **Pflicht** (wird als String gelesen).
* Spalte `path` ist **verboten** (wird intern gesetzt, würde verworfen).
* Jede weitere Spalte wird zu einer Label-Variable. Das Notebook nutzt `target='age'`, also
  muss `age` existieren; `sex` o. Ä. ist optional.
* Nur die Schnittmenge aus IDs in der CSV und Dateien in `images/` wird verwendet; für den Rest
  gibt es eine Warnung und die Fälle werden übersprungen. Fehlende Altersangaben (`NaN`) werden
  bei der Delta-Berechnung herausgefiltert, laufen aber durch das Modell.

### 2.3 `fastsurfer/<id>/mri/`

Wird erst ab der Zelle für die regionenweise Relevanz-Auswertung gebraucht. Das Notebook prüft
pro Subjekt explizit auf Existenz und überspringt es sonst stillschweigend:

```python
if not (os.path.isfile(os.path.join(image_folder, 'images', f'{id}.nii.gz')) and \
        os.path.isfile(os.path.join(fastsurfer_folder, id, 'mri',
                                    'aparc.DKTatlas+aseg.deep.mgz'))):
    continue
```

**Entscheidender Punkt:** Die Segmentierung muss auf **denselben gecroppten MNI-Bildern** laufen
wie die Brain-Age-Prädiktion, nicht auf den Roh-T1s. Das Notebook bringt Bild und Erklärungskarte
mit `FastSurferCNN.data_loader.conform.conform()` auf das konformierte Gitter (256³, 1 mm, LIA)
und vergleicht dann voxelweise mit der Maske. Nur wenn die Maske aus derselben Quelldatei stammt,
liegen beide auf demselben Gitter. Segmentiert man stattdessen den nativen T1, passen die
Voxelindizes nicht zusammen und die Regionen-Summen sind bedeutungslos.

---

## 3. Welche Dateien aus `recon-all` / FastSurfer werden gebraucht?

Das Notebook selbst liest aus einem Segmentierungslauf **genau eine Datei pro Subjekt**:

| Datei | Herkunft | Verwendung im Notebook |
|---|---|---|
| `mri/aparc.DKTatlas+aseg.deep.mgz` | FastSurfer (`run_fastsurfer.sh` / `FastSurferCNN/run_prediction.py`) | Regionenmaske für die voxelweise Aufsummierung der LRP-Relevanz je Hirnregion |

Bei einem klassischen FreeSurfer-`recon-all`-Lauf heißt das Äquivalent
**`mri/aparc.DKTatlas+aseg.mgz`** (DKT-Atlas, entsteht mit vollem `recon-all -all`).
Diese Datei muss dann als `aparc.DKTatlas+aseg.deep.mgz` kopiert oder verlinkt werden, oder der
Dateiname im Notebook wird angepasst. Mögliche Alternativen aus demselben Lauf, falls ein anderer
Atlas gewünscht ist:

* `mri/aparc+aseg.mgz` — Desikan-Killiany
* `mri/aparc.a2009s+aseg.mgz` — Destrieux
* `mri/aseg.mgz` — nur subkortikale Strukturen, keine kortikale Parzellierung

Alles andere aus `recon-all` (Surfaces unter `surf/`, `label/`, `stats/`, `scripts/`) wird vom
Notebook **nicht** gelesen.

### 3.1 `recon-all` in der Preprocessing-Kette davor

Für das Erzeugen der `cropped/images/` wird `recon-all` zwar gebraucht, aber nur für einen
einzigen Output. Die Bausteine liegen in `pyment/utils/preprocessing/`:

| Schritt | Funktion im Repo | Kommando | Benötigte Datei |
|---|---|---|---|
| 1. Reorientieren | `fsl.fslreorient2std` | `fslreorient2std` | Roh-T1 (`.nii.gz`) |
| 2. Bias-Korrektur + Skull-Stripping | `freesurfer.autorecon1` | `recon-all -s <id> -sd <dir> -i <t1> -autorecon1` | → **`mri/brainmask.mgz`** |
| 3. Konvertieren | `freesurfer.convert_mgz_to_nii_gz` | `mri_convert <src> <dest> -ot nii` | `brainmask.mgz` |
| 4. MNI152-Registrierung | `fsl.flirt` | `flirt -dof 6` gegen MNI152 1 mm | Ergebnis 182 × 218 × 182 |
| 5. Crop | `crop.crop_mri` / `crop_folder` | — | Ergebnis **167 × 212 × 160** |

`pyment/utils/preprocessing/utils.py::extract_brainmasks_from_recon` sammelt Schritt 2 ein und
prüft ebenfalls nur auf `mri/brainmask.mgz`. Aus dem Preprocessing-Lauf wird also aus
`recon-all` ausschließlich `brainmask.mgz` verwendet — `-autorecon1` genügt, ein voller
`recon-all -all` ist dafür nicht nötig (wohl aber, wenn die DKT-Parzellierung aus 3. mit
FreeSurfer statt FastSurfer erzeugt werden soll).

Zu Schritt 5: Im Repo sind **keine Crop-Koordinaten hinterlegt**. Aus dem Vergleich vorhandener
`mni152/`- und `cropped/`-Daten der historischen Läufe ergibt sich
`bounds = ((6, 173), (2, 214), (0, 160))` (Reihenfolge y, x, z; Start inklusive, Ende exklusive).
`crop_mri` übernimmt die Affine unverändert, korrigiert also den Ursprungs-Offset des Crops
nicht — für die Auswertung im Notebook ist das unkritisch, solange Bild und Segmentierung aus
derselben gecroppten Datei stammen.

---

## 4. Modellgewichte

`RegressionSFCN(weights='brain-age')` lädt keine lokale Datei aus dem Datenordner, sondern:

```
~/.pyment/models/regression_sfcn_reg_2025_weights.h5
```

Fehlt die Datei, wird sie beim ersten Aufruf von GitHub (`estenhl/pyment-public`) heruntergeladen.
`'brain-age'` und `'reg-2025'` verweisen auf denselben Checkpoint.

---

## 5. Zusätzliche Datei außerhalb von `ixi_folder`

Die Zelle zur Benennung der Hirnregionen liest:

```python
fastsurfer_labels = pd.read_csv('~/data/IXI/fastsurfer_labels.csv')
```

Erwartete Spalten: `id` (numerisches Label aus der Segmentierung) und `name` (Regionsname).
Fallbacks im Notebook: `0.0 → 'Background'`, `2.0 → 'WM'`, sonst wird die Zahl selbst angezeigt.

Die Datei lässt sich aus der im Repo mitgelieferten LUT erzeugen:
`FastSurferCNN/config/FastSurfer_ColorLUT.tsv` (Spalten `ID`, `LabelName`, `R`, `G`, `B`, `A`) —
also `ID → id`, `LabelName → name`. Alternativ `$FREESURFER_HOME/FreeSurferColorLUT.txt`.

Beachte: Dieser Pfad liegt **nicht** unter `ixi_folder`, sondern unter `~/data/IXI/`. Entweder
diese Datei dort ablegen oder die Zeile auf `ixi_folder` umstellen.

---

## 6. Checkliste vor dem Ausführen

| Voraussetzung | Status prüfen |
|---|---|
| `cropped/images/*.nii.gz` mit Shape 167 × 212 × 160 | vorhanden (563 Dateien) |
| `cropped/labels.csv` mit `id,age` | vorhanden (588 Subjekte) |
| `fastsurfer/<id>/mri/aparc.DKTatlas+aseg.deep.mgz` | **fehlt** — Regionen-Zellen liefern sonst leere Ergebnisse |
| `~/data/IXI/fastsurfer_labels.csv` | **fehlt** — muss aus der LUT erzeugt werden |
| `~/.pyment/models/regression_sfcn_reg_2025_weights.h5` | wird bei Bedarf automatisch geladen |

### Bekannte hart codierte Stellen, die angepasst werden müssen

* `sorted_bitmaps[0].save('/home/esten/demo.gif', ...)` — fremder Home-Pfad.
* `plt.savefig('/home/esten/sorted_correlations.png')` — fremder Home-Pfad.
* `ImageFont.truetype('arial.ttf', 20)` — unter Linux meist nicht vorhanden, z. B.
  `DejaVuSans.ttf` verwenden.
* `np.arange(0, 600, 100)` in den Achsen-Ticks der Korrelations-Heatmap setzt ≥ 600 Subjekte
  voraus.
* `all_explanations[i*4 + j]` setzt `batch_size=4` voraus; bei anderer Batch-Größe anpassen.
* IDs im Stil `IXI237-Guys-1049-T1` bzw. `IXI012-HH-1211-T1` sind in zwei Zellen fest
  eingetragen (Ausschluss eines Ausreißers, Einzelsubjekt-Auswertung). Die aktuelle
  `labels.csv` nutzt dagegen `sub-XXX`-IDs — diese Zellen laufen damit ins Leere bzw. in einen
  `KeyError`.
