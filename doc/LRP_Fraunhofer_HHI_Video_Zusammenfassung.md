# Layer-Wise Relevance Propagation (LRP) – Zusammenfassung

**Quelle:** [Layer-Wise Relevance Propagation: A Visual Introduction | XAI Methods Explained](https://www.youtube.com/watch?v=b26IZ2aYGjU)
**Kanal:** Fraunhofer HHI · **Länge:** 10:58 · **Veröffentlicht:** 2026-01-08

---

## Überblick

Layer-Wise Relevance Propagation (LRP) ist eine grundlegende Methode der **Explainable AI (XAI)**, 2015 von Fraunhofer HHI zusammen mit der TU Berlin eingeführt ([Bach et al., 2015](https://doi.org/10.1371/journal.pone.0130140)). LRP erklärt die Vorhersage eines neuronalen Netzes, indem es den **Output-Score rückwärts durch die Schichten propagiert** und dabei jedem Neuron – und am Ende jedem Eingabepixel – einen **Relevanzwert** zuweist.

Das Video ist die erste Folge einer XAI-Serie und nutzt einen **Katzen/Hunde-Klassifikator (CNN)** als durchgehendes Beispiel.

---

## Kapitel / Timestamps

| Zeit | Kapitel | Inhalt |
|------|---------|--------|
| 00:00 | Introduction | Warum sind Entscheidungen von KI-Systemen schwer nachvollziehbar? |
| 00:31 | A cats and dogs classifier | Motivierendes Beispiel: Tierklassifikation |
| 01:19 | Black box model | Regeln stecken verteilt in Millionen Parametern → nicht direkt lesbar |
| 03:32 | Fully-connected layers | Relevanz-Redistribution in dichten Schichten |
| 05:20 | Conservation property | Relevanz-Erhaltung über alle Schichten |
| 05:57 | Non linearities | Behandlung von ReLU (Identitätsregel) |
| 07:08 | Max pooling | Rückverteilung durch Pooling-Schichten |
| 07:55 | Convolution | Lokale Rückverteilung in Faltungsschichten |
| 08:23 | Explanation | Erzeugung der Heatmap, rote/blaue Features |
| 09:29 | Recap | Observe → Redistribute → Reveal |

---

## Das Kernproblem (Black Box)

- Ein Netz klassifiziert ein Tier korrekt und mit hohem Score – aber **wir wissen nicht, *warum***.
- Hat es sich auf die spitzen Ohren / die Schnauze konzentriert? Oder auf ein **Artefakt** wie den weißen Hintergrund oder die sitzende Pose?
- **Zentrale Aussage:** *„Just because a model is right doesn't mean it got there for the right reason."* – Nur weil ein Modell richtig liegt, muss der Weg dorthin nicht korrekt sein.
- Im Gegensatz zur klassischen Programmierung werden die Regeln nicht von Hand definiert, sondern aus Daten gelernt und in Gewichten/Aktivierungen versteckt. LRP ist eine **Post-hoc-Technik**, die auf ein bereits trainiertes Modell angewendet wird.

---

## Der LRP-Prozess: Observe → Redistribute → Reveal

1. **Observe (Beobachten):** Ein Forward-Pass liefert die Vorhersage und erfasst, wie stark jedes Neuron aktiviert wurde.
2. **Redistribute (Rückverteilen):** Der Output-Score wird Schicht für Schicht rückwärts propagiert – proportional dazu, wie stark jedes Neuron zur Aktivierung des nachfolgenden beigetragen hat.
3. **Reveal (Aufdecken):** Am Ende entsteht eine **Heatmap** über den Eingabepixeln.

### Grundgleichungen (Fully-Connected Layer)

Beitrag von Neuron `j` zu Neuron `k`:

```
z_jk = a_j · w_jk
```

Rückverteilte Relevanz von `k` nach `j` (Normierung über alle Beiträge zu `k`):

```
R_{j←k} = (z_jk / Σ_j z_jk) · R_k
```

Gesamt-Relevanz von Neuron `j` (Summe über alle nachfolgenden `k`):

```
R_j = Σ_k (z_jk / Σ_j' z_j'k) · R_k
```

**Bias-Term:** Wird als zusätzliches Eingangsneuron `j=0` mit konstanter Aktivierung 1 und Gewicht = Bias modelliert, damit die Relevanz erhalten bleibt.

### Erhaltungseigenschaft (Conservation)

```
Σ_j R_j = Σ_k R_k   (für jede Schicht, bis hinunter zum Input)
```

LRP **erzeugt oder vernichtet keine Relevanz**, es verteilt sie nur um. Wichtig: **Relevanz ≠ Aktivierung** – ein schwach aktiviertes Neuron kann hohe Relevanz haben, wenn es entscheidend für die Vorhersage war.

### Initialisierung

Man setzt die Relevanz des Zielneurons (z. B. „Corgi") gleich seiner Aktivierung, alle anderen auf 0:

```
R_corgi = a_corgi ,   R_(andere) = 0
```

### Nichtlinearitäten – ReLU (Identitätsregel)

- War das Neuron im Forward-Pass **inaktiv** (Beitrag 0) → es erhält **0 Relevanz**.
- War es **aktiv** → die Relevanz fließt **unverändert** hindurch.
- Gilt für die meisten elementweisen 1-zu-1-Operationen.

### Max Pooling

Zwei gängige Strategien zur Rückverteilung innerhalb des Pooling-Fensters:
- **Uniform:** Relevanz gleichmäßig auf alle Pixel verteilen.
- **Winner-takes-all:** gesamte Relevanz geht an das Pixel mit der höchsten Aktivierung im Fenster.

### Convolution

Faltung = geteilte Gewichte (Filter), die mit Pixeln multipliziert und summiert werden. Im Backward-Pass wird die Relevanz **lokal** verteilt, jeder Pixel bekommt einen Anteil proportional zu seinem Beitrag zur Aktivierung.

---

## Rote und blaue Farben – die Entscheidungsfindung (Details siehe eigener Abschnitt unten)

- **Rot** = Features, die **für** die Entscheidung sprechen (positive Relevanz).
- **Blau** = Features, die **gegen** die Entscheidung sprechen (negative Relevanz).

Für den Hund heben rote Bereiche die entscheidungsrelevanten Merkmale hervor (statt Artefakten wie dem Hintergrund). Wendet man LRP auf das **Katzen-Neuron** an, aktivieren spitze Ohren, kleine Pfoten und flauschiges Fell die Katzen-Klasse (rot), während Gesicht/Augen/Schnauze das Netz **von** einer Katze wegdrücken (blau).

---

## Werkzeuge & Erweiterungen

- **Zennit** (PyTorch), **LXT** (LRP für Transformer, PyTorch), **iNNvestigate** (Keras/TensorFlow).
- LRP ist nicht auf CNNs beschränkt: auch rekurrente, graphbasierte und **Transformer**-Modelle.
- Anwendbar auf Klassifikation, Regression, Zeitreihen – u. a. in Hochrisiko-Domänen wie der Krebsdiagnostik.
- Weiterentwicklung: **Concept Relevance Propagation (CRP)** – erklärt nicht nur *wo*, sondern *welche Konzepte* das Modell nutzt.

**Weitere Quellen:**
- Original-Paper: [Bach et al., 2015](https://doi.org/10.1371/journal.pone.0130140)
- Übersicht: [Montavon et al., 2019](https://link.springer.com/chapter/10.1007/978-3-030-28954-6_10)
- Interaktive Demo: [lrpserver.hhi.fraunhofer.de](https://lrpserver.hhi.fraunhofer.de/)

---

## Detailerklärung: Rot/Blau zwischen 9:00 und 10:00

Dieser Zeitbereich (Kapitel *Explanation* → *Recap*) erklärt, wie aus den propagierten Relevanzwerten die **farbige Heatmap** entsteht und wie sie die Entscheidung sichtbar macht.

### 1. Was die Farbe kodiert

Am Ende der Rückpropagation hat **jedes Eingabepixel `p` einen Relevanzwert `R_p`**. Die Farbe kodiert **Vorzeichen und Betrag** dieses Werts:

```
R_p > 0   → ROT    (Feature spricht FÜR die erklärte Klasse)
R_p < 0   → BLAU   (Feature spricht GEGEN die erklärte Klasse)
R_p ≈ 0   → neutral / weiß
Farbintensität ∝ |R_p|
```

Ausgesagt im Video: *„The areas lit up in red are the model's key suspects … These parts carried the most weight when deciding for a corgi classification."*

### 2. Woher kommt das Vorzeichen? (die eigentliche Mathematik)

Das Vorzeichen von `R_p` stammt aus dem **Beitragsterm** `z_jk = a_j · w_jk` und dessen Normierung. Beim Rückverteilen

```
R_j = Σ_k (a_j · w_jk / Σ_j' a_j' · w_j'k) · R_k
```

kann der Zähler `a_j · w_jk` **positiv oder negativ** sein:

- **Positiver Beitrag** (`a_j · w_jk > 0`): das Neuron/Pixel hat die Aktivierung des Zielpfads **verstärkt** → es bekommt **positive** Relevanz → **rot**.
- **Negativer Beitrag** (`a_j · w_jk < 0`): das Neuron/Pixel hat **entgegengewirkt** → **negative** Relevanz → **blau**.

Da bei ReLU die Identitätsregel gilt und die Relevanz erhalten bleibt (`Σ R = const`), pflanzt sich dieses Vorzeichen bis zu den Pixeln fort. Rot und Blau sind also keine reine Visualisierungs-Deko, sondern das **Vorzeichen der aufsummierten Beiträge** entlang aller Pfade zum Zielneuron.

### 3. Wovon die Farbe abhängt: von der erklärten Klasse

Entscheidend ist, **welches Output-Neuron** man initialisiert. Die Farbe eines Pixels ist **relativ zur gewählten Zielklasse `c`**:

```
Initialisierung:  R_c = a_c ,  alle anderen Outputs = 0
danach:           Backward-Pass  →  R_p  für jedes Pixel
```

Das Video demonstriert genau das mit zwei Durchläufen auf demselben Bild:

1. **Ziel = Hund/Corgi:** rote Bereiche = die charakteristischen Hundemerkmale (Schnauze, Ohren), auf die sich die Entscheidung stützt – **nicht** der Hintergrund. Das zeigt, dass das Modell aus den „richtigen Gründen" richtig liegt.
2. **Ziel = Katze** (erneut LRP, jetzt am Katzen-Neuron gestartet, alle anderen 0):
   - **Rot (für Katze):** spitze Ohren, kleine Pfoten, flauschiges Fell → diese Features **aktivierten** das Katzen-Neuron.
   - **Blau (gegen Katze):** Gesicht, insbesondere **Augen und Schnauze** → diese Merkmale **drückten das Netz von** der Katzen-Klassifikation weg.

Dasselbe Pixel kann also für die eine Klasse rot und für eine andere blau sein – die Heatmap beantwortet immer die Frage: *„Was sprach für bzw. gegen **genau diese** Klasse?"*

### 4. Zusammenspiel mit der Erhaltungseigenschaft

Weil `Σ_p R_p` gleich dem Score der erklärten Klasse ist, gilt intuitiv:

```
(Summe der roten Relevanz) − (Betrag der blauen Relevanz) ≈ Klassen-Score
```

Rot „baut" den Score auf, Blau „zieht ihn ab". Wenn also für die Katze viel Blau (Augen/Schnauze) auftritt, erklärt das, **warum die Katzen-Klasse trotz einiger passender Features (rot) am Ende nicht gewinnt** – die hemmenden Beiträge überwiegen. Genau das ist mit *„What details spoke for a cat? And why didn't they carry enough weight?"* gemeint.

### 5. Kurz-Fazit des Abschnitts

- **Rot = positive Relevanz = pro Zielklasse**, **Blau = negative Relevanz = contra Zielklasse**, Intensität = Betrag.
- Das Vorzeichen kommt aus `a_j · w_jk` (verstärkend vs. hemmend), fortgepflanzt durch die erhaltende Rückverteilung.
- Die Farbe ist **immer relativ zur gewählten Ausgabeklasse** – deshalb liefern Hund- und Katzen-Neuron unterschiedliche Heatmaps.
- Die Erhaltungseigenschaft macht die Bilanz „Rot minus Blau ≈ Score" faithful zur tatsächlichen Modellentscheidung.
