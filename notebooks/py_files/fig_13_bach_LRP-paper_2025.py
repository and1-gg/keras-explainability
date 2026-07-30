# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: py-uv_keras-xai (uv)
#     language: python
#     name: py-uv_keras-xai
# ---

# %%
"""
Isolierter Nenner-Vergleich fuer die Pixel-Dichtekarten aus Formel (65).

Zeigt nebeneinander:
  (A) d_k(p)      = Zaehler / Nenner   -- die ORIGINALE Formel 65
  (B) d_k^num(p)  = Zaehler / 1        -- NENNER KUENSTLICH AUF 1 GESETZT

(B) ist also einfach der rohe Zaehler (auf sein eigenes Maximum normiert,
damit man ihn ueberhaupt als Bild darstellen kann -- Werte wuerden sonst
in die Hunderte/Tausende gehen, da ueber ~6000 Trainingsbilder pro Klasse
aufsummiert wird).

Der Vergleich macht sichtbar, dass die roten Balken in (A) NICHT daher
kommen, dass eine Klasse an bestimmten Positionen besonders VIELE Pixel
setzt (das saehe man auch in (B)), sondern daher, dass sie dort die
EINZIGE Klasse ist, die ueberhaupt etwas beitraegt -- ein Effekt, der nur
durch die Division durch den (dort kleinen) Nenner sichtbar wird.
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Daten laden ueber tensorflow_datasets.
#    Wichtig: tfds.as_numpy liefert bereits NumPy-Arrays. Ab hier NICHT mehr
#    tf.cast() verwenden, sonst werden x_train/y_train wieder zu TF-Tensoren
#    und die spaetere NumPy-Rechnung (np.eye[...], .reshape, @) bricht bzw.
#    verhaelt sich anders als erwartet. Normalisierung deshalb rein mit NumPy.
# ---------------------------------------------------------------------------
ds = tfds.load("mnist", split="train", batch_size=-1, as_supervised=True)
x_train, y_train = tfds.as_numpy(ds)          # x: (60000, 28, 28, 1), y: (60000,)
x_train = x_train[..., 0]                     # Kanaldimension entfernen -> (60000, 28, 28)

# In [0, 1] skalieren (NumPy statt tf.cast, siehe Hinweis oben).
x_train = x_train.astype(np.float32) / 255.0  # (N, 28, 28)
y_train = y_train.astype(np.int64)            # (N,)  -- int fuer Indexierung/Onehot

NUM_CLASSES = 10
H, W = 28, 28


# ---------------------------------------------------------------------------
# 2. Zaehler: class_sum[k] = sum_{N in digits(k)} N(p)   -> Form (10, 28, 28)
#    (per One-Hot-Matmul, komplett in NumPy)
# ---------------------------------------------------------------------------
onehot = np.eye(NUM_CLASSES, dtype=np.float32)[y_train]      # (N, 10)
flat   = x_train.reshape(-1, H * W)                          # (N, 784)

class_sum = onehot.T @ flat                                  # (10, 784)  == Zaehler
class_sum = class_sum.reshape(NUM_CLASSES, H, W)              # (10, 28, 28)


# ---------------------------------------------------------------------------
# 3a. NENNER = Gesamtaktivitaet ueber alle Klassen (Formel 65, Original).
# ---------------------------------------------------------------------------
total = class_sum.sum(axis=0)                                 # (28, 28)  == Nenner
eps = 1e-12
density_formula65 = class_sum / (total[np.newaxis, :, :] + eps)
density_formula65 = np.where(total[np.newaxis, :, :] > 0.0, density_formula65, 0.0)


# ---------------------------------------------------------------------------
# 3b. NENNER = 1 (kuenstlich abgeschaltet). Uebrig bleibt der reine Zaehler.
#     Fuer eine vergleichbare Farbskala normieren wir jede Klassenkarte auf
#     ihr eigenes Maximum -> reine Form/Verteilung, kein Bezug zu anderen
#     Klassen mehr.
# ---------------------------------------------------------------------------
denominator_one = np.ones_like(total)                          # Nenner ist ueberall 1
numerator_only = class_sum / denominator_one[np.newaxis, :, :] # = class_sum selbst
numerator_only = numerator_only / (numerator_only.max(axis=(1, 2), keepdims=True) + eps)


# ---------------------------------------------------------------------------
# 4. Vergleichsplot: fuer jede Klasse (A) Formel 65 vs. (B) Nenner=1
#    Platz fuer eine gemeinsame Colorbar rechts wird per gridspec reserviert
#    (extra, schmale Spalte statt der 10 Bildspalten).
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    2, 11, figsize=(21, 5),
    gridspec_kw={"width_ratios": [1] * 10 + [0.15]}   # letzte Spalte = Colorbar
)

im = None  # merkt sich das letzte imshow-Objekt fuer die Colorbar
for k in range(NUM_CLASSES):
    im = axes[0, k].imshow(density_formula65[k], cmap="jet", vmin=0.0, vmax=1.0)
    axes[0, k].set_title(f"$d_{{{k}}}$\n(Formel 65)", fontsize=9)
    axes[0, k].axis("off")

    axes[1, k].imshow(numerator_only[k], cmap="jet", vmin=0.0, vmax=1.0)
    axes[1, k].set_title(f"Zaehler$_{{{k}}}$\n(Nenner=1)", fontsize=9)
    axes[1, k].axis("off")

# die 11. Spalte in beiden Zeilen zu einer einzigen Colorbar-Achse verschmelzen
axes[0, 10].remove()
axes[1, 10].remove()
cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])   # [links, unten, breite, hoehe] in Figure-Koordinaten
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Wert (0 = kein Beitrag, 1 = maximaler Beitrag)")

fig.suptitle("Nenner-Effekt isoliert: d_k (Formel 65) vs. reiner Zaehler (Nenner=1)", fontsize=13)
#plt.savefig("mnist_denominator_effect.png", dpi=150, bbox_inches="tight")
print("gespeichert: mnist_denominator_effect.png")

# %%
"""
Minimalbeispiel zu Formel (65) mit 4 ERKENNBAREN 14x14-Ziffernbildern.

Statt (wie im vorigen 4x4-Beispiel) die Pixel von Hand zu tippen, rendern
wir hier echte Ziffern "6" und "8" mit einer Schriftart, binarisieren sie
und erzeugen durch leichte Variation (Schriftschnitt/Offset) 2 Bilder pro
Klasse -- genau wie im 4x4-Beispiel: 2x "6", 2x "8", macht 4 Bilder total.

Der Rest der Rechnung (Zaehler, Nenner, Formel 65, Nenner=1-Vergleich) ist
IDENTISCH zum 4x4-Beispiel, nur eben auf einem 14x14-Raster, auf dem man
die Ziffern jetzt auch optisch als "6" und "8" erkennt.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

np.set_printoptions(precision=2, suppress=True, linewidth=150)

SIZE = 14  # Rasterbreite/-hoehe in Pixeln


# ---------------------------------------------------------------------------
# 1. Eine Ziffer als 14x14-Binaerbild rendern (0/1), leicht parametrisierbar
#    ueber Schriftart und horizontalen Offset, um 2 leicht unterschiedliche
#    Varianten pro Ziffer zu erzeugen (so wie A/B bzw. C/D im 4x4-Beispiel).
# ---------------------------------------------------------------------------
def render_digit(digit: str, font_path: str, font_size: int, x_offset: int = 0,
                  y_offset: int = 0, supersample: int = 8) -> np.ndarray:
    # In hoher Aufloesung zeichnen und dann auf SIZE x SIZE herunterskalieren
    # (Supersampling), damit die Kontur beim Verkleinern sauberer aussieht.
    big = SIZE * supersample
    img = Image.new("L", (big, big), color=0)          # schwarzer Hintergrund
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size * supersample)

    bbox = draw.textbbox((0, 0), digit, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pos = ((big - w) // 2 - bbox[0] + x_offset * supersample,
           (big - h) // 2 - bbox[1] + y_offset * supersample)
    draw.text(pos, digit, fill=255, font=font)

    img = img.resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    binary = (arr > 0.35).astype(np.float32)            # binarisieren: 0 oder 1
    return binary


FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SERIF   = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

# Klasse "6": zwei leicht unterschiedliche "Handschriften" -- bewusst etwas
# nach OBEN verschoben, damit die 6 (wie in echten Handschriften oft der
# Fall) mit ihrem Bogen weiter nach oben reicht als die 8.
A = render_digit("6", FONT_REGULAR, font_size=13, x_offset=0,  y_offset=-3)
B = render_digit("6", FONT_SERIF,   font_size=13, x_offset=1,  y_offset=-2)

# Klasse "8": zwei leicht unterschiedliche "Handschriften" -- bewusst etwas
# nach UNTEN verschoben, damit oben eine Zone bleibt, die die 8 NIE erreicht.
C = render_digit("8", FONT_REGULAR, font_size=13, x_offset=0,  y_offset=1)
D = render_digit("8", FONT_SERIF,   font_size=13, x_offset=1,  y_offset=2)

images = {"A (6)": A, "B (6)": B, "C (8)": C, "D (8)": D}


# ---------------------------------------------------------------------------
# 2. Zaehler pro Klasse: elementweise Summe der Bilder EINER Klasse.
# ---------------------------------------------------------------------------
class_sum_6 = A + B
class_sum_8 = C + D


# ---------------------------------------------------------------------------
# 3. Nenner: Summe UEBER ALLE Klassen.
# ---------------------------------------------------------------------------
total = class_sum_6 + class_sum_8


# ---------------------------------------------------------------------------
# 4. d_6(p) = Zaehler_6 / Nenner   (Formel 65, mit 0/0 := 0)
# ---------------------------------------------------------------------------
d6 = np.divide(class_sum_6, total, out=np.zeros_like(class_sum_6), where=total != 0)


# ---------------------------------------------------------------------------
# 5. Zum Vergleich: Nenner = 1, also nur der rohe Zaehler (unnormiert).
# ---------------------------------------------------------------------------
numerator_only_6 = class_sum_6 / 1.0


# ---------------------------------------------------------------------------
# 6. Konsolenausgabe der Zahlenwerte (gerundet), damit man einzelne Zeilen
#    nachrechnen kann -- genau wie im 4x4-Beispiel.
# ---------------------------------------------------------------------------
print("=" * 70)
print(f"Formen: jedes Bild ist {SIZE}x{SIZE}, Werte 0 (Hintergrund) / 1 (Tinte)")
print("=" * 70)

print("\nZeile 0 (ganz oben) der vier Rohbilder:")
for name, img in images.items():
    print(f"  {name}: {img[0].astype(int)}")

print(f"\nZeile 0 von Zaehler_6 (= A+B): {class_sum_6[0]}")
print(f"Zeile 0 von Zaehler_8 (= C+D): {class_sum_8[0]}")
print(f"Zeile 0 von Nenner (= Zaehler_6+Zaehler_8): {total[0]}")
print(f"Zeile 0 von d_6 = Zaehler_6/Nenner: {d6[0]}")

mittlere_zeile = SIZE // 2
print(f"\nZur Kontrolle Zeile {mittlere_zeile} (Bildmitte, wo beide Ziffern praesent sind):")
print(f"  Zaehler_6: {class_sum_6[mittlere_zeile]}")
print(f"  Zaehler_8: {class_sum_8[mittlere_zeile]}")
print(f"  d_6: {d6[mittlere_zeile]}")


# ---------------------------------------------------------------------------
# 7. Visualisierung: Rohbilder oben, Rechenschritte unten.
# ---------------------------------------------------------------------------
def heat(ax, matrix, title, vmin=0, vmax=None, cmap="jet"):
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return im


fig, axes = plt.subplots(2, 4, figsize=(13, 7))

heat(axes[0, 0], A, "Bild A (Klasse 6)", vmin=0, vmax=1, cmap="gray")
heat(axes[0, 1], B, "Bild B (Klasse 6)", vmin=0, vmax=1, cmap="gray")
heat(axes[0, 2], C, "Bild C (Klasse 8)", vmin=0, vmax=1, cmap="gray")
heat(axes[0, 3], D, "Bild D (Klasse 8)", vmin=0, vmax=1, cmap="gray")

heat(axes[1, 0], class_sum_6, "Zaehler$_6$ = A+B", vmin=0, vmax=2)
heat(axes[1, 1], total, "Nenner = Zaehler$_6$+Zaehler$_8$", vmin=0, vmax=4)
im_d6 = heat(axes[1, 2], d6, "d$_6$ = Zaehler$_6$/Nenner\n(Formel 65)", vmin=0, vmax=1)
heat(axes[1, 3], numerator_only_6, "Zaehler$_6$/1\n(Nenner kuenstlich=1)", vmin=0, vmax=2)

# gemeinsame Colorbar fuer die untere Zeile (0..max, hier zur Orientierung
# zwei getrennt skalierte Groessen, deshalb zwei kleine Colorbars)
fig.colorbar(im_d6, ax=axes[1, 2], fraction=0.046, pad=0.04, label="Anteil (0-1)")

fig.suptitle("14x14-Beispiel: erkennbare Ziffern 6 und 8 -> Balken oben in d$_6$",
             fontsize=13)
plt.tight_layout()
#plt.savefig("toy_example_14x14.png", dpi=150, bbox_inches="tight")
print("\ngespeichert: toy_example_14x14.png")

# %%
"""
Erste 100 Sechsen aus MNIST plotten, dann:
  (a) d_6(p) nach Formel (65)  -- Zaehler / Nenner
  (b) nur der Zaehler von Formel (65), ohne Division durch den Nenner

Formel (65):
               sum_{N in digits(k)} N(p)
    d_k(p) = ---------------------------------
             sum_{l=0..9} sum_{N in digits(l)} N(p)

Fuer (a) brauchen wir den Zaehler ausschliesslich ueber die 100 Sechsen
und den Nenner ueber ALLE Trainingsbilder aller 10 Klassen (nicht nur
100 Bilder -- der Nenner ist die Gesamtaktivitaet des ganzen Datensatzes).
Fuer (b) lassen wir die Division einfach weg und zeigen den rohen Zaehler.
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. MNIST laden (Trainingsset) ueber tensorflow_datasets.
#    Nach tfds.as_numpy sind x_train/y_train NumPy-Arrays -> ab hier bei
#    NumPy bleiben (kein tf.cast mehr, siehe frueherer Bugfix).
# ---------------------------------------------------------------------------
ds = tfds.load("mnist", split="train", batch_size=-1, as_supervised=True)
x_train, y_train = tfds.as_numpy(ds)             # x: (60000, 28, 28, 1), y: (60000,)
x_train = x_train[..., 0]                        # Kanaldimension entfernen -> (60000, 28, 28)

x_train = x_train.astype(np.float32) / 255.0      # Werte in [0, 1]
y_train = y_train.astype(np.int64)

NUM_CLASSES = 10
H, W = 28, 28


# ---------------------------------------------------------------------------
# 2. Die ersten 100 Bilder mit Label "6" herausfiltern.
# ---------------------------------------------------------------------------
idx_sixes = np.where(y_train == 6)[0][:100]        # Indizes der ersten 100 Sechsen
sixes_100 = x_train[idx_sixes]                     # Form (100, 28, 28)
print(f"Gefundene Sechsen: {len(idx_sixes)}")


# ---------------------------------------------------------------------------
# 3. Die ersten 100 Sechsen als 10x10-Grid plotten.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(sixes_100[i], cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
fig.suptitle("Die ersten 100 Sechsen aus dem MNIST-Trainingsset", fontsize=14)
plt.tight_layout()
plt.savefig("mnist_100_sixes_grid.png", dpi=150, bbox_inches="tight")
print("gespeichert: mnist_100_sixes_grid.png")


# ---------------------------------------------------------------------------
# 4a. ZAEHLER fuer Klasse 6, aber NUR ueber die 100 herausgepickten Bilder
#     (so wie es der Plot-Titel "alle 100 Bilder in eines" verlangt).
# ---------------------------------------------------------------------------
zaehler_6_100 = sixes_100.sum(axis=0)              # (28, 28), Summe der 100 Sechsen


# ---------------------------------------------------------------------------
# 4b. NENNER: Gesamtaktivitaet UEBER ALLE 10 Klassen, ueber das GESAMTE
#     Trainingsset (nicht nur 100 Bilder!). Das ist die korrekte Umsetzung
#     von Formel (65): der Nenner bezieht sich auf die volle Datenbasis.
# ---------------------------------------------------------------------------
onehot = np.eye(NUM_CLASSES, dtype=np.float32)[y_train]     # (60000, 10)
flat   = x_train.reshape(-1, H * W)                          # (60000, 784)

class_sum_all = onehot.T @ flat                               # (10, 784)
class_sum_all = class_sum_all.reshape(NUM_CLASSES, H, W)      # (10, 28, 28)

nenner_gesamt = class_sum_all.sum(axis=0)                     # (28, 28) -- alle 10 Klassen, alle Bilder


# ---------------------------------------------------------------------------
# 5a. (a) d_6 nach Formel 65: Zaehler (nur die 100 Sechsen) / Nenner (alle
#     Klassen, ganzer Datensatz).
# ---------------------------------------------------------------------------
eps = 1e-12
d6_formula65 = np.divide(zaehler_6_100, nenner_gesamt,
                          out=np.zeros_like(zaehler_6_100), where=nenner_gesamt > 0)


# ---------------------------------------------------------------------------
# 5b. (b) Nur der Zaehler, ohne Division durch den Nenner. Zur besseren
#     Vergleichbarkeit auf [0,1] normiert (durch das eigene Maximum).
# ---------------------------------------------------------------------------
zaehler_only_normiert = zaehler_6_100 / (zaehler_6_100.max() + eps)


# ---------------------------------------------------------------------------
# 6. (a) und (b) nebeneinander plotten.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

im0 = axes[0].imshow(d6_formula65, cmap="jet", vmin=0, vmax=1)
axes[0].set_title("(a) $d_6$ nach Formel 65\n"
                   "Zaehler = 100 Sechsen, Nenner = alle 10 Klassen (ganzer Datensatz)")
axes[0].axis("off")
fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Anteil (0-1)")

im1 = axes[1].imshow(zaehler_only_normiert, cmap="jet", vmin=0, vmax=1)
axes[1].set_title("(b) Nur Zaehler (Nenner weggelassen)\n"
                   "= Summe der 100 Sechsen, auf eigenes Maximum normiert")
axes[1].axis("off")
fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="normierte Intensitaet (0-1)")

plt.tight_layout()
#plt.savefig("mnist_100_sixes_formula65_vs_numerator.png", dpi=150, bbox_inches="tight")
print("gespeichert: mnist_100_sixes_formula65_vs_numerator.png")

# %%
"""
Reproduktion von Figure 11 aus Bach et al. (2015):
"Taylor-approximated pixel-wise predictions for a multilayer neural network
 trained and tested on the MNIST data set."

Aufbau je Vierergruppe (wie im Paper, von links nach rechts):
    1. Eingabeziffer x
    2. Taylor-Wurzelpunkt x0
    3. Gradient Df_k(x0) der Vorhersagefunktion fuer Zielklasse k
    4. Approximierte pixelweise Beitraege R_d nach Gl. (53)

Reproduziert werden die beiden Gruppen aus dem hochgeladenen Ausschnitt:
    (A) Eingabe "5", x0 = leere Kachel,        Zielklasse 3
    (B) Eingabe "2", x0 = naechste "6",        Zielklasse 2

Formeln:
    Gl. (18):  f(x) ~= sum_d  df/dx_d (x0) * (x_d - x0_d),   mit f(x0) = 0
    Gl. (53):  R_d   =  (x - x0)_d  *  df/dx_d (x0)
    Gl. (35):  Normierung fuer die Farbdarstellung auf [-1, +1]

WICHTIG: Fig 11 ist nicht bit-genau reproduzierbar -- das Paper nennt weder
Random Seed noch die konkreten Testbilder. Reproduziert werden Architektur,
Trainingssetup und Methodik; die Heatmaps sehen qualitativ aehnlich aus.
"""

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ===========================================================================
# 1. Daten laden
# ===========================================================================
def load_split(split):
    ds = tfds.load("mnist", split=split, batch_size=-1, as_supervised=True)
    x, y = tfds.as_numpy(ds)
    return x[..., 0].astype(np.float32) / 255.0, y.astype(np.int64)


x_train, y_train = load_split("train")
x_test,  y_test  = load_split("test")

H, W = 28, 28
NUM_CLASSES = 10


# ===========================================================================
# 2. Eingabenormalisierung (Praemisse aus Abschnitt "MNIST experiments I"):
#    "Input data is normalized so that the sum of pixels is on average zero,
#     and the variance of pixel values is on average one."
#
#    Erst dadurch bekommen Hintergrundpixel einen von 0 verschiedenen Wert
#    und koennen ueberhaupt Relevanz tragen (Faktor (x-x0)_d in Gl. 53).
# ===========================================================================
MU  = x_train.mean()
SIG = x_train.std()
print(f"Normalisierung: mu={MU:.4f}, sigma={SIG:.4f}")

def normalize(a):
    return (a - MU) / SIG

x_train_n = normalize(x_train)
x_test_n  = normalize(x_test)
print(f"Wertebereich nach Normalisierung: [{x_train_n.min():.2f}, {x_train_n.max():.2f}]")


# ===========================================================================
# 3. Netzarchitektur exakt nach Paper:
#    784 -> 400 (tanh) -> 400 (tanh) -> 10, danach Softmax.
#
#    Die Decomposition arbeitet auf dem Output der LETZTEN LINEAREN SCHICHT,
#    also VOR dem Softmax (Paper: "calculated based on the output of the last
#    linear layer without taking the succeeding softmax normalization layer
#    into account"). Deshalb bauen wir das Modell mit linearem Ausgang und
#    wenden Softmax nur in der Loss-Funktion an.
# ===========================================================================
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(H, W)),
    tf.keras.layers.GaussianNoise(0.1),      # nur im Training aktiv (Paper: Noise-Layer)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(400, activation="tanh"),
    tf.keras.layers.Dense(400, activation="tanh"),
    tf.keras.layers.Dense(NUM_CLASSES, activation=None),   # LINEAR, kein Softmax
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Paper: Mini-Batches der Groesse 25, Gaussian-Noise-Layer pro Iteration,
# 50 000 Iterationen. 50000 Iterationen * 25 Samples / 60000 ~= 21 Epochen.
#
# Das Gaussian-Noise-Layer sitzt als erste Schicht im Modell. Keras aktiviert
# es automatisch nur waehrend des Trainings (training=True) und schaltet es
# bei Inferenz/Gradientenberechnung ab -- genau das Verhalten, das wir wollen:
# das Rauschen ist Trainings-Regularisierung, darf aber die spaetere
# Decomposition nicht verfaelschen.
EPOCHS = 10
BATCH  = 50

print("\nTraining laeuft (kann einige Minuten dauern)...")
model.fit(
    x_train_n, y_train,
    epochs=EPOCHS, batch_size=BATCH, verbose=2,
    validation_data=(x_test_n, y_test),
)

test_loss, test_acc = model.evaluate(x_test_n, y_test, verbose=0)
print(f"\nTestgenauigkeit: {test_acc*100:.2f}%  (Paper: 98.25%)")


# ===========================================================================
# 4. Gradient Df_k(x0): Ableitung des Klassifikator-Ausgangs fuer Zielklasse k
#    nach den Eingabepixeln, ausgewertet AM WURZELPUNKT x0 (nicht am Bildpunkt!).
#
#    Genau das ist der Unterschied zu klassischen Sensitivitaetskarten, den das
#    Paper betont: der Gradient am Vorhersagepunkt zeigt nicht notwendig zu
#    einer nahen Nullstelle und ist daher fuer die Erklaerung ungeeignet.
# ===========================================================================
def gradient_at(x0, target_class):
    """Df_k(x0) -- Form (28, 28)."""
    x0_t = tf.convert_to_tensor(x0[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x0_t)
        logits = model(x0_t, training=False)
        f_k = logits[0, target_class]          # Ausgang fuer die Zielklasse
    grad = tape.gradient(f_k, x0_t)
    return grad.numpy()[0]


# ===========================================================================
# 5. Wurzelpunkt x0 suchen.
#
#    Anforderung laut Paper: f(x0) = 0 (Zustand maximaler Unsicherheit) und
#    x0 moeglichst nah bei x. Umsetzung per Liniensuche (Paper, Abschnitt
#    "Taylor-type decomposition"): Auf der Strecke l(a) = a*x + (1-a)*x'
#    zwischen x und einem Kandidaten x' mit entgegengesetztem Vorzeichen
#    muss eine Nullstelle liegen; sie wird per Intervallhalbierung gefunden.
# ===========================================================================
def f_k(x, target_class):
    """Skalarer Klassifikator-Ausgang f_k(x)."""
    logits = model(tf.convert_to_tensor(x[np.newaxis, ...], dtype=tf.float32),
                   training=False)
    return float(logits[0, target_class])


def point_on_line(x, x_candidate, a):
    """l(a) = a*x + (1-a)*x_candidate.  a=1 -> x,  a=0 -> x_candidate,
    a<0 -> ueber den Kandidatenpunkt hinaus extrapoliert."""
    return a * x + (1.0 - a) * x_candidate


def find_root_on_segment(x, x_candidate, target_class, iters=50):
    """Sucht auf der Geraden durch x und x_candidate einen Punkt mit f_k = 0.

    Paper: "the line l(a) = ax + (1-a)x' must contain a root of f which can be
    found by interval intersection" -- das gilt, sobald f(x) und f(x') ver-
    schiedene Vorzeichen haben. Ist das auf dem Segment [0,1] nicht der Fall
    (das Netz haelt auch den Kandidaten noch fuer die Zielklasse), erweitern
    wir die Suche ueber den Kandidatenpunkt hinaus (a < 0), bis ein Vor-
    zeichenwechsel auftritt.
    """
    f_x    = f_k(x, target_class)                    # a = 1
    f_cand = f_k(x_candidate, target_class)          # a = 0

    a_lo, f_lo = 0.0, f_cand
    a_hi, f_hi = 1.0, f_x

    # Falls kein Vorzeichenwechsel auf [0,1]: nach a < 0 extrapolieren.
    if np.sign(f_lo) == np.sign(f_hi):
        found = False
        a_try = 0.0
        for _ in range(60):
            a_try -= 0.25                            # Schritt ueber den Kandidaten hinaus
            f_try = f_k(point_on_line(x, x_candidate, a_try), target_class)
            if np.sign(f_try) != np.sign(f_hi):
                a_lo, f_lo = a_try, f_try
                found = True
                break
        if not found:
            # Notfall: Punkt mit kleinstem |f| auf einem groben Raster.
            grid = np.linspace(-15.0, 1.0, 200)
            vals = [abs(f_k(point_on_line(x, x_candidate, a), target_class)) for a in grid]
            return point_on_line(x, x_candidate, grid[int(np.argmin(vals))])

    # Bisektion zwischen a_lo und a_hi
    for _ in range(iters):
        a_mid = 0.5 * (a_lo + a_hi)
        f_mid = f_k(point_on_line(x, x_candidate, a_mid), target_class)
        if np.sign(f_mid) == np.sign(f_lo):
            a_lo, f_lo = a_mid, f_mid
        else:
            a_hi, f_hi = a_mid, f_mid

    return point_on_line(x, x_candidate, 0.5 * (a_lo + a_hi))


def nearest_neighbour_of_class(x, target_digit):
    """Naechstes Trainingsbild der Klasse target_digit (euklidische Distanz)."""
    candidates = x_train_n[y_train == target_digit]
    d = np.linalg.norm(candidates.reshape(len(candidates), -1) - x.ravel(), axis=1)
    return candidates[int(np.argmin(d))]


# ===========================================================================
# 6. Taylor-Decomposition nach Gl. (53):  R_d = (x - x0)_d * df/dx_d (x0)
# ===========================================================================
def taylor_decomposition(x, x0, target_class):
    grad = gradient_at(x0, target_class)
    R = (x - x0) * grad                      # elementweises Produkt
    return R, grad


def normalize_for_display(R):
    """Gl. (35): Normierung auf [-1, +1] fuer das Farbmapping."""
    m = np.abs(R).max()
    return R / m if m > 0 else R


# ===========================================================================
# 7. Die beiden Gruppen aus dem Bildausschnitt bauen.
# ===========================================================================

# --- Gruppe A: Eingabe "5", x0 = leere Kachel, Zielklasse 3 -----------------
#
# Das Paper beschreibt x0 als "blank tile". Eine voellig leere Kachel ist aber
# nicht automatisch eine Nullstelle von f_3. Wir nutzen die leere Kachel daher
# als KANDIDATENPUNKT x' und suchen per Liniensuche den Punkt auf der Strecke
# x <-> x', an dem f_3 = 0 gilt (Paper, Abschnitt "Taylor-type decomposition").
# Das Ergebnis ist ein sehr blasses, fast leeres Bild -- genau wie im Paper.
idx_5 = np.where(y_test == 5)[0][0]
x_A = x_test_n[idx_5]
blank = np.full((H, W), normalize(0.0), dtype=np.float32)   # leere Kachel
target_A = 3
x0_A = find_root_on_segment(x_A, blank, target_class=target_A)
R_A, grad_A = taylor_decomposition(x_A, x0_A, target_A)

# --- Gruppe B: Eingabe "2", x0 = naechste "6", Zielklasse 2 ----------------
#
# Hier ist der Kandidatenpunkt keine leere Kachel, sondern das euklidisch
# naechstgelegene Trainingsbild der Klasse 6. Auf der Verbindungsstrecke
# wird wieder die Nullstelle gesucht.
idx_2 = np.where(y_test == 2)[0][0]
x_B = x_test_n[idx_2]
nn_6 = nearest_neighbour_of_class(x_B, 6)             # naechster Nachbar aus Klasse 6
target_B = 2
x0_B = find_root_on_segment(x_B, nn_6, target_class=target_B)
R_B, grad_B = taylor_decomposition(x_B, x0_B, target_B)

print(f"\nGruppe A: f_3(x0) = {f_k(x0_A, 3):+.3f}   (Ziel: nahe 0)")
print(f"Gruppe B: f_2(x0) = {f_k(x0_B, 2):+.3f}   (Ziel: nahe 0)")


# ===========================================================================
# 8. Plot: zwei Vierergruppen nebeneinander, wie im Paper.
# ===========================================================================
fig, axes = plt.subplots(1, 8, figsize=(18, 2.8))

def show(ax, img, title, cmap="gray", vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

# Gruppe A
show(axes[0], x_A,  "Eingabe (5)",              cmap="gray_r")
show(axes[1], x0_A, "$x_0$ (leere Kachel)",      cmap="gray_r",
     vmin=x_A.min(), vmax=x_A.max())
show(axes[2], normalize_for_display(grad_A), "$Df_3(x_0)$",
     cmap="jet", vmin=-1, vmax=1)
show(axes[3], normalize_for_display(R_A), "$R_d$ (Gl. 53), Klasse 3",
     cmap="jet", vmin=-1, vmax=1)

# Gruppe B
show(axes[4], x_B,  "Eingabe (2)",              cmap="gray_r")
show(axes[5], x0_B, "$x_0$ (naechste 6)",        cmap="gray_r")
show(axes[6], normalize_for_display(grad_B), "$Df_2(x_0)$",
     cmap="jet", vmin=-1, vmax=1)
im = show(axes[7], normalize_for_display(R_B), "$R_d$ (Gl. 53), Klasse 2",
          cmap="jet", vmin=-1, vmax=1)

fig.suptitle("Reproduktion von Fig 11: Taylor-Decomposition (Gl. 53)  |  "
             "jet: blau = negative, gruen = 0, rot = positive Relevanz",
             fontsize=12)
plt.tight_layout()
#plt.savefig("fig11_reproduction.png", dpi=150, bbox_inches="tight")
print("\ngespeichert: fig11_reproduction.png")

# %%
