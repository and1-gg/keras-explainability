# Unit-Tests – Dokumentation

## Übersicht

Dieses Dokument beschreibt alle Unit-Tests für das `explainability`-Paket.
Die Tests befinden sich im Verzeichnis `tests/` und decken alle Layer-Implementierungen,
Hilfsutilitys und das zentrale LRP-Modell ab.

**Ergebnis beim letzten Ausführen:** 92 passed, 1 xfailed (bekannter Keras-3-Bug,
dokumentiert in `test_lrp_integration.py::test_lrp_include_prediction_flag`)

---

## Tests ausführen

### Voraussetzung

Die Tests laufen in der `uv`-verwalteten virtuellen Umgebung unter `.venv/`.
Python und alle Abhängigkeiten sind damit bereits installiert.

```bash
# Im Repo-Root:
.venv/bin/pytest tests/ -v
```

Alternativ mit Filter auf eine bestimmte Datei:

```bash
.venv/bin/pytest tests/test_dense_lrp.py -v
```

Mit Coverage-Report (sofern `pytest-cov` installiert ist):

```bash
.venv/bin/pytest tests/ --cov=explainability --cov-report=term-missing
```

### Erwartete Ausgabe

```
92 passed, 1 xfailed in ~11 seconds
```

Das `xfailed` ist kein Fehler – es dokumentiert einen bekannten
Inkompatibilitätsbug zwischen `include_prediction=True` und Keras 3
(doppelte Layer-Namen im Ausgabemodell).

---

## Testdateien im Überblick

### Bestehende Tests (vor dieser Ergänzung)

| Datei | Getestetes Modul | Anzahl Tests |
|---|---|---|
| `test_dense_lrp.py` | `layers/dense.py` – `DenseLRP` | 14 |
| `test_conv2d_lrp.py` | `layers/conv.py` – `Conv2DLRP` | 7 |
| `test_conv3d_lrp.py` | `layers/conv.py` – `Conv3DLRP` | 2 |
| `test_maxpool_lrp.py` | `layers/pooling.py` – `MaxPoolingLRP` | 6 |
| `test_batchnormalization_lrp.py` | `layers/normalization.py` – `BatchNormalizationLRP` | 1 |
| `test_relu_lrp.py` | `layers/activations.py` – `ReLULRP` | (kommentiert, Designentscheidung) |
| `test_add.py` | `layers/arithmetic.py` – `AddLRP` | 2 |
| `test_subtract.py` | `layers/arithmetic.py` – `SubtractLRP` | 1 |
| `test_reshape_lrp.py` | `layers/reshape.py` – `ReshapeLRP` | 1 |
| `test_fuse_batchnorm.py` | `model/utils/fuse_batchnorm.py` | 1 |
| `test_remove_activation.py` | `model/utils/remove_activation.py` | 6 |
| `test_infer_graph_structure.py` | `utils/infer_graph_structure.py` | 1 |
| `test_topological_sort.py` | `utils/topological_sort.py` | 1 |
| `test_lrp.py` | `model/lrp.py` – End-to-End LayerNorm | 1 |
| `test_lrp_strategy.py` | `utils/strategies/lrp_strategy.py` | 6 |
| `test_restructured_lrp.py` | `model/restructured_lrp.py` | (kommentiert, Feature in Entwicklung) |

### Neue Tests (ergänzt)

| Datei | Getestetes Modul | Anzahl Tests |
|---|---|---|
| `test_noop_lrp.py` | `layers/noop.py` – `NoOpLRP` | 4 |
| `test_avgpool_lrp.py` | `layers/pooling.py` – `AveragePoolingLRP` | 5 |
| `test_topological_sort_extended.py` | `utils/topological_sort.py` | 5 |
| `test_fuse_batchnorm_extended.py` | `model/utils/fuse_batchnorm.py` | 5 |
| `test_arithmetic_lrp_extended.py` | `layers/arithmetic.py` | 9 |
| `test_lrp_integration.py` | `model/lrp.py` – Integrationsszenarien | 8 |
| `test_remove_activation_extended.py` | `model/utils/remove_activation.py` | 7 |

---

## Detailbeschreibung der neuen Tests

### `test_noop_lrp.py` – NoOpLRP

`NoOpLRP` ist eine Passthrough-Schicht: Sie ignoriert die Aktivierung `a` und gibt
die Relevanz `R` unverändert zurück. Sie wird für Schichten ohne eigene
LRP-Regel verwendet (z. B. Dropout oder Lambda-Layer).

| Testname | Was geprüft wird | Warum gut |
|---|---|---|
| `test_noop_lrp_passes_relevance_unchanged` | R wird bit-genau durchgereicht | Kernvertrag der Schicht |
| `test_noop_lrp_ignores_activation` | Verschiedene a-Werte → gleiche Ausgabe | Unabhängigkeit von a |
| `test_noop_lrp_preserves_zeros` | Null-Relevanz bleibt Null | Edge-Case |
| `test_noop_lrp_is_lrp_layer` | Ist Unterklasse von `LRPLayer` | Typ-Sicherheit im Dispatch |

---

### `test_avgpool_lrp.py` – AveragePoolingLRP

`AveragePoolingLRP` propagiert Relevanz durch Average-Pooling-Schichten zurück.
Die Notebooks verwenden Average-Pooling intensiv in 3D-CNN-Architekturen
(vgl. `Train_and_explain_synthetic_brain_regression_model.py`).

| Testname | Was geprüft wird | Warum gut |
|---|---|---|
| `test_avgpool2d_wrong_layer_raises` | Falscher Layer-Typ → AssertionError | Defensives API |
| `test_avgpool2d_redistribute_relevance_sum_preserved` | Gesamtrelevanz bleibt bei redistribute erhalten | Summenregel der LRP |
| `test_avgpool2d_flat_uniform_distribution` | flat-Strategie verteilt gleichmäßig | Konsistenz der Strategie |
| `test_global_avgpool2d_in_lrp_model` | GlobalAveragePooling2D im vollen LRP-Modell | End-to-End wie im Notebook |
| `test_avgpool2d_strategy_invalid_raises` | Unbekannte Strategie → ValueError | Fehlerbehandlung |

---

### `test_topological_sort_extended.py` – topological_sort

`topological_sort` bestimmt die Reihenfolge, in der LRP die Schichten eines
Modells rückwärts durchläuft. Eine falsche Sortierung würde stille Fehler
in der Relevanzpropagation verursachen.

| Testname | Was geprüft wird | Warum gut |
|---|---|---|
| `test_topological_sort_linear_chain` | A→B→C→D wird korrekt sortiert | Normalfall |
| `test_topological_sort_single_node` | Einzelner Knoten ohne Kanten | Trivialfall / Edge-Case |
| `test_topological_sort_two_roots` | Zwei Quellen → gemeinsamer Knoten | Parallele Eingaben (Multi-Input-Modell) |
| `test_topological_sort_returns_all_nodes` | Jeder Knoten genau einmal | Vollständigkeit |
| `test_topological_sort_diamond` | Diamant-Graph: 0→1, 0→2, 1→3, 2→3 | Skip-Connections wie in ResNets |

---

### `test_fuse_batchnorm_extended.py` – fuse_batchnorm

`fuse_batchnorm` faltet BatchNorm-Parameter mathematisch in die vorherige
Dense/Conv-Schicht. LRP benötigt das, weil BatchNorm keine eigenständige
LRP-Regel hat. Fehler hier würden stille Abweichungen in allen LRP-Erklärungen
verursachen.

| Testname | Was geprüft wird | Warum gut |
|---|---|---|
| `test_fuse_dense_batchnorm_output_unchanged` | Modell-Output bleibt nach Fuse gleich (±0.01) | Korrektheit des Fuse |
| `test_fuse_dense_batchnorm_output_unchanged_no_bias` | Dense ohne Bias liefert endliche Ausgabe | Edge-Case: kein Bias |
| `test_fuse_batchnorm_neutralizes_bn_parameters` | BN-Parameter: gamma=1, beta=0, mean=0, var=1 | BN wirkt nach Fuse als Identität |
| `test_fuse_conv2d_batchnorm_output_unchanged` | Conv2D+BN Output bleibt gleich (±0.005) | Anwendung auf CNN |
| `test_fuse_batchnorm_identity_when_no_bn` | Modell ohne BN wird nicht verändert | Keine ungewollten Seiteneffekte |

**Hinweis zur Toleranz:** Der Fuse-Vorgang führt Float32-Berechnungen durch
(Gewichts-Skalierung), die zu geringen numerischen Abweichungen führen.
Die Toleranz von 0.01 (Dense) bzw. 0.005 (Conv2D) ist bewusst gewählt und
entspricht der Präzision, die für LRP-Erklärungen ausreichend ist.

---

### `test_arithmetic_lrp_extended.py` – AddLRP / SubtractLRP

`AddLRP` und `SubtractLRP` verteilen Relevanz auf beide Eingabe-Tensoren einer
Addition bzw. Subtraktion proportional zu deren Beitrag. Sie werden in
Skip-Connection-Architekturen (ResNets) und in Modellen mit mehreren Eingaben
benötigt.

| Testname | Was geprüft wird | Warum gut |
|---|---|---|
| `test_compute_add_lrp_relevance_splits_proportionally` | Proportionale Aufteilung a/(a+b) | Mathematische Korrektheit |
| `test_compute_add_lrp_relevance_sum_preserved` | R_a + R_b ≈ R | Summenregel |
| `test_unpack_binary_two_tensors` | Zwei-Tensor-Eingang → constant=False | Normalpfad |
| `test_unpack_binary_constant_operand` | Einzel-Tensor → constant=True | Keras-3-Spezialfall |
| `test_unpack_binary_wrong_length_raises` | 3 Tensoren → ValueError | Fehlerbehandlung |
| `test_add_lrp_wrong_layer_raises` | AddLRP mit Subtract-Layer → AssertionError | Typ-Sicherheit |
| `test_add_lrp_constant_operand_passthrough` | constant_operand-Pfad gibt R zurück | Pfad-Abdeckung |
| `test_subtract_lrp_wrong_layer_raises` | SubtractLRP mit Add-Layer → AssertionError | Typ-Sicherheit |
| `test_subtract_lrp_negate_b` | b wird negiert vor der Aufteilung | Korrektheit der Subtraktion |

---

### `test_lrp_integration.py` – LayerwiseRelevancePropagator

Integrationstests, die typische Nutzungsmuster aus den Notebooks abbilden.

| Testname | Was geprüft wird | Warum gut |
|---|---|---|
| `test_lrp_relevance_conservation_dense` | Summe der Eingangsrelevanz = Ausgangsrelevanz | Fundamentale LRP-Eigenschaft |
| `test_lrp_include_prediction_flag` | `include_prediction=True` (xfail) | Dokumentiert bekannten Keras-3-Bug |
| `test_lrp_non_flat_output_raises` | Conv-Ausgabe → NotImplementedError | Klare Fehlermeldung |
| `test_lrp_negative_idx_raises` | idx=-1 → NotImplementedError | Klare Fehlermeldung |
| `test_lrp_conv2d_model_explanation_shape` | Erklärungs-Shape = Eingabe-Shape | Wie `xai_with_2dmnist.py` |
| `test_lrp_with_strategy_sets_epsilon` | LRPStrategy setzt epsilon schichtweise | Strategie-Mechanismus |
| `test_lrp_with_strategy_and_epsilon_raises` | Strategie + epsilon → AssertionError | Konflikt-Erkennung |
| `test_lrp_dense_synthetic_3class` | LRP hebt richtige Features hervor | Analog zu Notebook-Validierung |

**Bezug zu den Notebooks:**
- `test_lrp_conv2d_model_explanation_shape` entspricht dem Muster in `xai_with_2dmnist.py`
- `test_lrp_dense_synthetic_3class` entspricht der Sanity-Check-Logik in
  `Train_and_explain_dummy_geometric_data.py`: Wir wissen, welche Features
  relevant sind, und prüfen, ob LRP das korrekt identifiziert.

---

### `test_remove_activation_extended.py` – remove_activation

`remove_activation` entfernt Softmax/Sigmoid von der letzten Ausgabeschicht,
damit LRP auf rohen Logits arbeiten kann. Ein Fehler hier würde die gesamte
LRP-Erklärung unbrauchbar machen.

| Testname | Was geprüft wird | Warum gut |
|---|---|---|
| `test_remove_softmax_changes_activation` | Aktivierung wird auf `linear` gesetzt | Kernfunktion |
| `test_remove_sigmoid_changes_activation` | Dasselbe für Sigmoid | Vollständigkeit |
| `test_remove_activation_preserves_weights` | Gewichte bleiben unverändert | Keine ungewollten Seiteneffekte |
| `test_remove_activation_preserves_output_shape` | Output-Shape bleibt gleich | API-Stabilität |
| `test_remove_activation_skips_non_matching` | Andere Aktivierung → Original-Modell | Keine ungewollten Änderungen |
| `test_remove_activation_non_dense_last_layer_skips` | Letzte Schicht kein Dense → unverändert | Edge-Case |
| `test_remove_softmax_logits_match_linear_model` | Logits = Dense-Rohausgabe ohne Aktivierung | Numerische Korrektheit |

---

## Warum diese Tests gut sind

### 1. Abdeckung der LRP-Grundeigenschaften
Die wichtigste Eigenschaft von LRP ist die **Relevanzerhaltung**: Die Summe der
Relevanz an der Eingabe entspricht der Relevanz am Ausgang. `test_lrp_relevance_conservation_dense`
prüft genau das. Ein Modell, das diese Eigenschaft verletzt, produziert stille,
schwer zu erkennende Fehler.

### 2. Orientierung an Notebook-Szenarien
Die Integrationstests in `test_lrp_integration.py` sind direkt von den Notebooks
abgeleitet:
- Das 3-Klassen-Szenario mit synthetischen Daten (Notebook:
  `Train_and_explain_dummy_geometric_data.py`) prüft, ob LRP die richtigen
  Features hervorhebt – **die einzige objektive Qualitätsprüfung, die bei
  synthetischen Daten möglich ist**.
- Das CNN-Szenario (`xai_with_2dmnist.py`) stellt sicher, dass die Erklärungen
  die richtige räumliche Dimension haben.

### 3. Mathematische Korrektheit durch Handrechnung
Tests wie `test_compute_add_lrp_relevance_splits_proportionally` und
`test_dense_lrp` (bestehend) verwenden händisch berechnete Erwartungswerte.
Das ermöglicht eine **exakte Verifikation der Implementierung**, nicht nur eine
Plausibilitätsprüfung.

### 4. Explizite Fehlerfall-Tests
Jede LRP-Schicht akzeptiert nur den korrekten Keras-Layer-Typ. Tests wie
`test_add_lrp_wrong_layer_raises` stellen sicher, dass Fehler früh und mit
klarer Meldung signalisiert werden.

### 5. Dokumentation bekannter Bugs
`test_lrp_include_prediction_flag` ist als `xfail` markiert und **dokumentiert
einen bekannten Inkompatibilitätsbug** mit Keras 3. Er wird nicht als Fehler
gewertet, aber wenn der Bug behoben wird, fällt das sofort auf (weil der Test
dann unerwartet besteht).

---

## Hinweise

- **Zufallsabhängigkeit:** Einige Tests verwenden `np.random.rand`. Sie sind
  deterministisch genug, weil die Toleranzen groß genug gewählt wurden.
  Bei Bedarf kann ein globaler Seed in `conftest.py` gesetzt werden.

- **GPU-Nutzung:** Die Tests laufen auf CPU. Für GPU-Tests gibt es separate
  Makefile-Targets (`make test-gpu`).

- **TensorFlow-Warnungen:** Die 6 `DeprecationWarnings` stammen aus Keras selbst
  (NumPy-Scalar-Konvertierung) und sind harmlos.
