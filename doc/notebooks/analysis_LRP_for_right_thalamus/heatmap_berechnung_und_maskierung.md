# Heatmap-Berechnung, `mask_explanation()` und die Differenz zwischen `pred` und `sum_R`

Quelle: `notebooks/ipynb_files/analysis_LRP_for_right_thalamus_volume_based_on_CNN_prediction.ipynb`
(bzw. das jupytext-Pendant `notebooks/py_files/…py`), Abschnitte A.5, A.7, A.9 und C.

Kurz vorweg: Die Differenz zwischen der Vorhersage und der Relevanzsumme der gespeicherten
Heatmap ist **kein Erhaltungsproblem des LRP**. Über alle Schichten hinweg bleibt die Relevanz
zu ~100,7 % erhalten — die fehlenden ~29 % entstehen erst *nach* dem LRP, durch
`mask_explanation()`, das die Relevanz auf allen Hintergrundvoxeln wegwirft. Für `ixi/sub-470`
ist ΣR am Voxel-Eingang **8214,73** (gegenüber `pred = 8156,85`), und erst die Maskierung
drückt das auf die **5833,55** in der Auswertungstabelle aus A.9.


## 1. Wie die Heatmap berechnet wird

Die Kette pro Subject steckt in A.7 und ist nur vier Zeilen lang:

```python
vol = load_volume(path)
y_pred = float(np.squeeze(model.predict(np.expand_dims(vol, 0), verbose=0)))
R_raw = lrp(np.expand_dims(vol, 0))[0].numpy()
R_masked = mask_explanation(vol, R_raw)
```

1. **Laden** — `cropped.nii.gz` wird durch `cfg.preprocessing.normalization_factor` geteilt und
   auf das Trainings-FOV `167×212×160×1` gebracht. Der Hintergrund (Luft um den Kopf, plus alles,
   was der Skull-Strip entfernt hat) ist exakt `0`; die Division ändert daran nichts.
2. **Vorwärts** — `model.predict` liefert das skalare Thalamus-Volumen in mm³.
3. **Rückwärts (LRP)** — `LRP(model, layer=len(model.layers)-1, idx=0, strategy)` maskiert den
   Ausgang auf Neuron 0 und propagiert diesen einen Wert schichtweise zurück. Die
   Composite-Strategie aus A.5, in Vorwärtsreihenfolge gelesen:

   | Schicht | Regel |
   | --- | --- |
   | `block-0_conv`, `block-1_conv` | `flat` |
   | `block-2_conv`, `block-3_conv`, `block-4_conv`, `top_conv` | `α=2, β=1` |
   | `predictions` (Dense) | `ε=0.25` |

   Ergebnis ist `R_raw` mit derselben Form wie das Bild — ein Relevanzwert pro Voxel.
4. **Maskieren + Speichern** — `mask_explanation()`, dann `save_heatmap_nifti()` mit der Affine
   des Original-NIfTIs, damit die Heatmap im Viewer deckungsgleich über dem T1 liegt.


## 2. `mask_explanation()` — was, warum, Ergebnis

```python
def mask_explanation(volume: np.ndarray, explanation: np.ndarray) -> np.ndarray:
    x = volume.squeeze()
    expl = explanation.squeeze().astype(np.float32)
    return expl * (x != 0).astype(np.float32)
```

Das ist eine reine Nullsetzung: Voxel mit `x == 0` bekommen `R = 0`, **alle anderen Werte bleiben
unverändert**. Es wird nichts umskaliert oder renormiert.

### Warum man das macht

Das eigentliche Ärgernis ist die `flat`-Regel an den beiden eingangsnahen Convolutions. Bei `αβ`
und `ε` gilt am Ende `R_i = a_i · c_i` — ein Voxel mit Aktivierung `a_i = 0` bekommt zwangsläufig
`R_i = 0`. `flat` setzt dagegen intern `a ← 1` und `w ← 1`
(`explainability/layers/layer.py`):

```python
if self.flat:
    a = tf.ones_like(a)
    w = tf.ones_like(w)
```

Damit wird die Relevanz eines Filter-Outputs *gleichmäßig über sein gesamtes rezeptives Feld*
verteilt, völlig unabhängig davon, was in den Voxeln steht. Ein Filter, der auf der
Hirnoberfläche feuert, schiebt einen Teil seiner Relevanz in die Luft daneben. Das ist ein
Artefakt der Regelwahl, keine Aussage über das Modell — die Luftvoxel sind bei *jedem* Subject
identisch 0 und können per Konstruktion keine Information über das Thalamusvolumen tragen.

Konkret muss man sie aus drei Gründen loswerden:

* **Anatomische Interpretierbarkeit** — Relevanz außerhalb des Kopfes ist nicht deutbar und würde
  in jeder ROI-Statistik als Rauschen mitlaufen.
* **Normierung** — `pct_|R|_left` / `pct_|R|_right` / `pct_|R|_outside` sollen Anteile an der
  **Hirn**-Relevanz sein. Ohne Maskierung wäre der Nenner um ~29 % aufgebläht und die
  Thalamus-Anteile künstlich kleiner.
* **Darstellung** — `plot_lrp_overlay` und der 3D-Plot setzen `vmax = max|R|`. Ein heller
  Hintergrundring würde die Farbskala kapern und den Kontrast im Gehirn plattdrücken.

### Ergebnis

Ein Volumen gleicher Form `167×212×160`, das nur noch auf dem Gehirn definiert ist. Für
`sub-470` sind das `n_nonzero = 2 007 343` von `167·212·160 = 5 664 640` Voxeln, also
**35,4 % Gehirn, 64,6 % verworfen**. Dieses maskierte Volumen ist das, was als NIfTI auf der
Platte landet und in A.9 wieder eingelesen wird — deshalb ist `sum_R` in der Tabelle die
Relevanz *im Gehirn*, nicht die Gesamtrelevanz.


## 3. Die Bilanz Schritt für Schritt (`ixi/sub-470`)

Zahlen aus
`output/notebooks/analysis_LRP_for_right_thalamus_volume_based_on_CNN_prediction/layerwise_relevance/sum_R_A_original_model_ixi_sub-470.csv`,
gelesen vom Ausgang zum Eingang:

| Schritt | ΣR | Δ | Anteil am Start |
| --- | --- | --- | --- |
| `f(x)` aus `model.predict` | 8156,85 | — | — |
| `mask:output_logit` (Start Rückwärtspfad) | 8152,17 | −4,68 | 100,00 % |
| `dense:predictions`, `ε=0.25` | 8142,58 | −9,59 | 99,88 % |
| `gap:top_pool` + 4× `conv` mit `α=2,β=1` + MaxPools | 8142,40 | −0,18 | 99,88 % |
| `conv:block-1_conv`, `flat` | 8137,16 | −5,25 | 99,82 % |
| `conv:block-0_conv`, `flat` → **Voxelebene** | **8214,73** | **+77,58** | **100,77 %** |
| nach `mask_explanation()` → gespeicherte Heatmap | **5833,55** | **−2381,18** | 71,5 % |

So liest sich das:

**Schritt 1 — `f(x)` → Maskenausgang (−0,06 %).** Der Lambda-Layer greift nur den Ausgangswert ab,
mathematisch sollte hier gar nichts passieren. Die 4,68 mm³ sind Float32-Rundung: `model.predict`
und der Vorwärtspfad im umgebauten LRP-Modell akkumulieren über ~5,7 Mio. Voxel und mehrere
Conv-Schichten in leicht unterschiedlicher Reihenfolge. Bei einem Zahlenwert um 8000 sind 0,06 %
genau die erwartete Größenordnung.

**Schritt 2 — ε am Dense (−0,12 %).** Der Stabilisator addiert `ε·sign(z)` auf den Nenner, damit
`R/z` bei `z ≈ 0` nicht explodiert. Das ist bewusst nicht erhaltend — der Preis für numerische
Stabilität. Mit `adjust_epsilon=True` könnte man das exakt zurückskalieren, hier ist es aber mit
9,59 mm³ irrelevant.

**Schritt 3 — GAP, αβ-Convs und MaxPools (−0,002 %).** Über sechs Schichten hinweg driftet ΣR um
insgesamt 0,18 mm³. Das ist praktisch perfekte Erhaltung und der eigentliche Sanity-Check: Wäre
in der Implementierung etwas kaputt, sähe man hier Größenordnungssprünge. Sehenswert ist dabei
die Spalte `sum_abs`: Bei `top_conv` steht ΣR = 8142,53, aber Σ|R| = 22045,67 — die αβ-Regel
splittet in +15094 und −6952, die sich zum Nettowert aufheben. Positive und negative Relevanz
wachsen also stark an, die *Summe* bleibt.

**Schritt 4 — die beiden `flat`-Convs (netto +0,89 %).** `block-1_conv` verliert 5,25,
`block-0_conv` erzeugt 77,58. `flat` ist streng genommen nicht erhaltend (Bias-Korrektur
`R·z/(z+bias)` und Randbehandlung), aber ein Prozent Abweichung ist unkritisch. Wichtiger als der
Betrag ist, *wohin* dieser letzte Schritt verteilt: Er ist die Schicht, die auf die Bildvoxel
abbildet, und er tut das gleichverteilt über die rezeptiven Felder — genau hier fließt Relevanz
in den Hintergrund.

**Schritt 5 — die Maskierung (−29,0 %).** Von den 8214,73 liegen 2381,18 auf Voxeln mit `x == 0`
und werden verworfen. Ein Detail dazu: Vergleicht man mit Σ|R|, verliert die Maske 2434,53 an
Betragsmasse bei 2381,18 an Nettomasse — d. h. rund 99 % der Hintergrundrelevanz ist **positiv**.
Das passt exakt zum Mechanismus: `flat` verteilt gleichmäßig, und das Vorzeichen der Relevanz ist
bereits vorher überwiegend positiv (der Regressor sagt ja ein positives Volumen vorher).

**Endstand:** `pred − sum_R = 8156,85 − 5833,55 = 2323,30`, also 28,5 % der Vorhersage. Für
`sub-634` sind es `8191,39 − 6154,75 = 2036,64`, also 24,9 % — dieselbe Mechanik, leicht anderes
Hirn-zu-Luft-Verhältnis.


## 4. Erhaltungsquoten aller vier Läufe

Aus `sum_R_conservation_summary.csv` (Teil C, jeweils erstes Holdout-Subject):

| Modell | Dataset | Subject | `y_pred` | ΣR Start | ΣR Ende (Voxel) | erhalten |
| --- | --- | --- | --- | --- | --- | --- |
| `A_original_model` | ixi | `sub-470` | 8156,85 | 8152,17 | 8214,73 | 100,77 % |
| `A_original_model` | ukb | `2784068_20252_2_0` | 6926,22 | 6923,03 | 6972,46 | 100,71 % |
| `B_jittered_model` | ixi | `sub-470` | 10000,00 | 10000,00 | 9960,36 | 99,60 % |
| `B_jittered_model` | ukb | `2784068_20252_2_0` | 10000,00 | 10000,00 | 9958,07 | 99,58 % |

Alle vier liegen innerhalb von ±1 % — die Relevanzerhaltung gilt also, wie erwartet, für beide
Modelle und beide Datensätze.


## 5. Anmerkungen

* A.7 speichert `sum_R_unmasked` *vor* der Maskierung in `preds_by_dataset`, und A.9 rechnet
  daraus `pct_discarded_background`. Fehlen diese beiden Spalten in der Auswertungstabelle, wurde
  A.7 in derselben Kernel-Session nicht (neu) ausgeführt. Sind sie gefüllt, steht die ~29 % direkt
  in der Zeile statt nur in der Teil-C-CSV.
* Unabhängig von der Maskierung bleibt ein inhaltlicher Punkt bestehen: `pct_|R|_right = 0,73 %`
  (`sub-470`) bzw. `4,15 %` (`sub-634`) ist für ein Modell, das das *rechte* Thalamusvolumen
  vorhersagen soll, auffällig niedrig — und diese Zahl ist bereits auf die Hirn-Relevanz normiert,
  also nicht durch die Maskierung erklärbar. Dafür liefert Teil B (Jitter-Experiment) den
  Gegentest.
