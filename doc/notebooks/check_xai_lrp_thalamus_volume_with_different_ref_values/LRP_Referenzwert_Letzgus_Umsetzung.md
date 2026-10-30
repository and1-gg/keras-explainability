# LRP mit Referenzwert nach Letzgus et al. (arXiv:2112.11407)ss

Umsetzung der **Restructuring**-Variante aus

> S. Letzgus, P. Wagner, J. Lederer, W. Samek, K.-R. Müller, G. Montavon,
> *Toward Explainable AI for Regression Models*, IEEE SPM 2022
> ([arXiv:2112.11407](https://arxiv.org/abs/2112.11407)), Abschnitt **IV-B**.

im Paket `explainability` dieses Repos.


| Datei                                                                                 | Rolle                                                           |
| ------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `explainability/model/utils/reference.py`                                             | Flooding, ReLU-Verdreifachung, Netz-Umbau                       |
| `explainability/model/lrp.py`                                                         | Optionaler Parameter `reference_value` in `LRP`                 |
| `tests/test_lrp_reference.py`                                                         | Unit-Tests (Conservation, Flooding, Clip-Kopf)                  |
| `notebooks/ipynb_files/check_xai_lrp_thalamus_volume_with_different_ref_values.ipynb` | Praxisvergleich klassisch vs. Referenz (dieses Doc gehört dazu) |


Ohne `reference_value` bleibt das bisherige LRP-Verhalten unverändert (implizit \widetilde{y}=0).

---



## 1. Warum Regression anders erklärt werden muss

Klassisches LRP erklärt typischerweise f(\boldsymbol{x}) relativ zu **0**.
Bei Klassifikation ist das oft sinnvoll (Logit / Entscheidungsschwelle).
Bei Regression hat die Vorhersage oft eine Einheit (z. B. mittlere Thalamus-Intensität), und die Frage lautet eher:

> Warum ist f(\boldsymbol{x}) **nicht** \widetilde{y}?

Formal erklärt man deshalb


g(\boldsymbol{x}) = f(\boldsymbol{x}) - \widetilde{y}.


Nur den Bias der letzten Schicht anzupassen reicht nicht: LRP behandelt Biases oft als „unerklärt“, und die interne Darstellung bleibt zu grob. Das Paper schlägt deshalb **Restructuring** vor: die letzten Schichten werden **ohne Neutraining** so umgeschrieben, dass g aus einer Standard-ReLU-Architektur kommt und LRP unverändert darauf laufen kann.

### Mini-Beispiel (Auktion aus dem Paper)

Zwei Gebote P_1=1100, P_2=900, Verkaufspreis y=\max(P_1,P_2)=1100.


| Referenz \widetilde{y} | Intuition der Erklärung                                       |
| ---------------------- | ------------------------------------------------------------- |
| 0                      | Beide Bieter „teilen“ grob Verantwortung für den Preis über 0 |
| 1000 (Mindestpreis)    | Nur Bieter 1 trägt den Vorteil über dem Mindestpreis          |


Genau diese Kontextualisierung ist das Ziel der Umstrukturierung.

---



## 2. Annahmen an die Netzspitze

Restructuring setzt voraus (nach BatchNorm-Fusion):

```text
pre (Conv2D / Conv3D / Dense, linear)
  → ReLU
  → [Dropout | GlobalAvgPool | Flatten]*
  → Dense (linear, Zielschicht)
```

Das passt zu Regression-SFCN: `top_conv` → ReLU → `GlobalAveragePooling3D` → `Dense(1)`.

Ein optionaler Range-Clip (`restrict_relu` + `restrict_add`) wird vor dem Umbau abgeschnitten (`_strip_range_clip`), weil die Paper-Formel auf dem **linearen Dense-Ausgang** definiert ist.

---



## 3. Schritt 1 im Paper: Referenzaktivierungen (Flooding)



### 3.1 Mathematik

Sei die letzte lineare Schicht (effektiv, inkl. Global-Avg-Pool)


y = \sum_j a_j c_j + b,


wobei a_j=\mathrm{ReLU}(z_j) die Aktivierungen der letzten versteckten Schicht sind und c_j die effektiven Ausgangsgewichte.

Gesucht ist \widetilde{\boldsymbol{a}} mit


\sum_j \widetilde{a}_j c_j = \widetilde{y},


dann gilt


y - \widetilde{y} = \sum_j (a_j - \widetilde{a}_j) c_j + b.


**Flooding:** grobe, große Aktivierungen werden zuerst „überflutet“, feine Effekte bleiben übrig:


\widetilde{\boldsymbol{a}}^{(\mathrm{flood})} = \bigl(\boldsymbol{a} - t\cdot \boldsymbol{d}\bigr)_+.


- \boldsymbol{d}=+1: Einheiten, deren Beitrag Richtung \widetilde{y} abgebaut werden muss  
- \boldsymbol{d}=-\kappa: Gegenseite (Paper Abschn. VI-A: \kappa=0.25; \kappa=0 = einseitiges Flooding)  
- Skalar t pro Sample per **Bisektion**, so dass \sum \widetilde{a}_j c_j = \widetilde{y}



### 3.2 Code: `ReferenceShift`

Klasse in `explainability/model/utils/reference.py`:

```python
# Richtung: muss man die Summe nach unten oder oben schieben?
g0 = tf.reduce_sum(a * c, axis=axes, keepdims=True)
direction = tf.where(g0 >= y_ref, tf.ones_like(g0), -tf.ones_like(g0))
cd = c * direction
d = tf.where(cd > 0, tf.ones_like(cd),
             tf.where(cd < 0, -kappa * tf.ones_like(cd),
                      tf.zeros_like(cd)))

# a_ref = (a - t * d)_+  mit t aus Bisektion auf h(t)=(g(t)-y_ref)*direction
a_ref = tf.nn.relu(a - t * d)
```

Effektive Koeffizienten c (auch hinter Global Average Pooling):

```python
# Dense-Gewichte → Shape der ReLU-Ausgabe; bei GAP: broadcast / V
c = output_coefficients(model, idx=0)
```

Bei GAP über V Raumpositionen gilt c_{\mathrm{spatial}} = w / V, damit


f(x) = \sum_{\mathrm{Orte},k} a_{\mathrm{Orte},k} c_{\mathrm{Orte},k} + b.


### 3.3 Numerisches Beispiel (Dense-Regressor aus den Tests)

Modell: 4 Inputs → Dense(6)+ReLU → Dense(1), positive Gewichte, Input
[0.5, 0.1, 0.3, 0.2].

```text
y      ≈ 1.136
y_ref = 0.3 * y ≈ 0.341
→ g(x) = y - y_ref ≈ 0.795
```

Flooding liefert \boldsymbol{a}*{\mathrm{ref}} mit \sum_j a*{\mathrm{ref},j} c_j \approx y_{\mathrm{ref}}
(geprüft in `test_flooding_matches_reference_value`).

---



## 4. Schritt 2 im Paper: verschobene ReLU → drei Standard-ReLUs



### 4.1 Mathematik


\rho(z_j) - \widetilde{a}_j
= \rho(z_j - \widetilde{a}_j) + \rho(-z_j) - \rho(-z_j + \widetilde{a}_j).


Jedes Neuron wird verdreifacht:


| Kopie | Eingangsgewichte | Bias-Offset      | Ausgangsgewichte |
| ----- | ---------------- | ---------------- | ---------------- |
| 1     | \times 1         | -\widetilde{a}_j | \times 1         |
| 2     | \times -1        | 0                | \times 1         |
| 3     | \times -1        | +\widetilde{a}_j | \times -1        |


Weil \widetilde{a} **sample-abhängig** ist, stecken die Offsets nicht als feste Keras-Biases, sondern in der Schicht `ReferenceShift`.

### 4.2 Code: `restructure_for_reference`

```python
# Eingangsseite: (W, -W, -W) und (b, -b, -b)
kernel_tri = np.concatenate([kernel, -kernel, -kernel], axis=-1)
bias_tri   = np.concatenate([bias,   -bias,   -bias],   axis=-1)

z_tri = pre_tri(pre.input)          # 3× Kanäle / Units
x = ReferenceShift(...)(z_tri)      # + (-a_ref, 0, +a_ref)
x = Activation('relu')(x)

# Ausgangsseite Dense: (w, w, -w)
w_tri = np.concatenate([w, w, -w], axis=...)
```

Ergebnis: funktional äquivalentes Netz mit g(x)=f(x)-\widetilde{y}, aber nur Standard-ReLUs → normale LRP-Regeln greifen.

---



## 5. Einbindung in `LRP`

```python
from explainability import LRP, LRPStrategy

# klassisch: erklärt f(x) (implizit y_ref = 0)
lrp0 = LRP(model, layer=layer_idx, idx=0, strategy=strategy)

# Paper: erklärt f(x) - y_ref
lrp_ref = LRP(
    model,
    layer=layer_idx,
    idx=0,
    strategy=strategy,
    reference_value=y_ref_net,      # Einheit = Netzausgabe
    reference_asymmetry=0.25,       # κ aus Abschn. VI-A
)
R = lrp_ref(np.expand_dims(vol, 0))
```

Ablauf in `LayerwiseRelevancePropagator.__init__`:

1. Optional Range-Clip → linearer Dense-Kopf (`find_linear_dense_head`)
2. `remove_activation` / `fuse_batchnorm` wie bisher
3. Falls `reference_value is not None`: `restructure_for_reference(...)` → `model` wird das umgeschriebene Netz
4. Danach derselbe LRP-Graph wie ohne Referenz



### LRP über `ReferenceShift`

Die Shift-Offsets werden **nicht** als Bias in LRP-0 mitgeführt (sonst würde Flooding wieder „weggerechnet“). Stattdessen: Shift überspringen, Bias der verdreifachten Pre-Schicht ignorieren (`ignore_bias=True`), analog Abschn. VI-A des Papers (Biases in FC-LRP ignoriert). Die Referenz steckt bereits in den verdreifachten ReLU-Aktivierungen.

```python
if isinstance(layer, ReferenceShift):
    pre = layers[i + 1]   # verdreifachte Conv/Dense
    kwargs['ignore_bias'] = True
    layer = pre
    i += 2
```

Zusätzliche Attribute nach dem Bau:

- `lrp.restructured_model` → x \mapsto f(x)-\widetilde{y}
- `lrp.reference_activations_model` → x \mapsto \boldsymbol{a}_{\mathrm{ref}}
- `lrp.reference_value` / `lrp.reference_asymmetry`

---



## 6. Conservation: was die Heatmap summiert

Idealerweise (ohne Stabilisierungs-Artefakte):


\sum_i R_i \approx
\begin{cases}
f(x) & \text{klassisch} 
f(x) - \widetilde{y} & \text{mit Referenz}
\end{cases}


Unit-Test-Idee:

```python
y = float(model(data))
y_ref = 0.3 * y
R = LRP(model, layer=3, idx=0, reference_value=y_ref)(data).numpy()
assert np.isclose(R.sum(), y - y_ref, atol=5e-4)
```

---



## 7. Anwendungsbeispiel: Thalamus-Intensität (UKB)

Notebook:
`notebooks/ipynb_files/check_xai_lrp_thalamus_volume_with_different_ref_values.ipynb`

Nur Modellfall **(1)** (unveränderte Scans).


| Variante  | `reference_value`          | Frage der Heatmap                                |
| --------- | -------------------------- | ------------------------------------------------ |
| klassisch | `None` (implizit 0)        | Warum diese Intensität vs. 0?                    |
| Letzgus   | Trainingsmittel der Labels | Warum weicht das Subject vom Trainingsmittel ab? |


Referenz auf der **Label-Skala** (Mittelwert der Trainings-TSV):

```python
y_ref_orig = float(train_df[PRED_VAR].mean())
```

Das Intensitätsmodell ist **z-score**-trainiert. Im Netzraum gilt daher


\widetilde{y}*{\mathrm{Netz}} = \frac{\bar{y}*{\mathrm{Train}} - \mu}{\sigma} \approx 0,


wenn \bar{y}_{\mathrm{Train}}=\mu. Der Output-Offset gleicht dann dem klassischen Fall; der Kartenunterschied kommt aus der **Umstrukturierung** (Flooding der letzten ReLU), nicht aus einem großen Bias-Shift. Heatmaps werden danach wieder mit \sigma auf die Intensitätsskala skaliert (`R · σ`), analog zu den anderen UKB-Notebooks.

Für Volumen-Modelle ohne z-Score (z. B. `Right-Whole_thalamus`) ist \widetilde{y}=\overline{V}_{\mathrm{Train}} vs. 0 oft der sichtbarere Kontrast.

---



## 8. Parameter-Kurzreferenz


| Parameter             | Bedeutung                                                                              |
| --------------------- | -------------------------------------------------------------------------------------- |
| `reference_value`     | \widetilde{y} in **Einheiten der Netzausgabe** (bei z-Score: z-Raum)                   |
| `reference_asymmetry` | \kappa; `0` einseitig, `0.25` Paper VI-A (auch wenn f(x)<\widetilde{y})                |
| `layer`               | Zielschicht; bei Clip-Kopf wird bei gesetztem `reference_value` der Dense-Kopf gewählt |


Typischer Aufruf im Notebook:

```python
LAYER_IDX = find_linear_dense_head(model)
y_ref_net = (y_ref_orig - mu) / sigma   # falls z-score, sonst y_ref_orig

R_classic = LRP(model, layer=LAYER_IDX, idx=0, strategy=strategy)(x)
R_paper   = LRP(
    model, layer=LAYER_IDX, idx=0, strategy=strategy,
    reference_value=y_ref_net, reference_asymmetry=0.25,
)(x)
```

---



## 9. Was bewusst *nicht* umgesetzt wurde


| Paper-Option                                                  | Status hier                                    |
| ------------------------------------------------------------- | ---------------------------------------------- |
| **Restructuring** (IV-B)                                      | umgesetzt                                      |
| **Retraining** eines Surrogats \hat{g} (IV-A)                 | nicht nötig für SFCN-Kopf; nicht implementiert |
| Removal-/Gradient-Methoden mit neuem Root-Punkt \widetilde{x} | außerhalb dieses LRP-Pfads                     |


---



## 10. Tests ausführen

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/test_lrp_reference.py -q
```

Abgedeckt u. a.:

- ohne `reference_value` → kein `restructured_model`
- g(x)=f(x)-\widetilde{y} nach Umbau
- \sum R \approx f-\widetilde{y}
- Flooding trifft \widetilde{y}
- Conv+GAP und Range-Clip-Kopf
- klassische vs. Referenz-Heatmap unterscheiden sich bei \widetilde{y}\neq 0

