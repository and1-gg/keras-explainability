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
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: dementia_xai_gpu_py-3.10_tf-2.17.1_cuda-12.3
#     language: python
#     name: dementia_xai_gpu_py-3.10_tf-2.17.1_cuda-12.3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.19
# ---

# %% [markdown]
# # Layerwise Relevance Propagation (LRP) für MNIST
#
# Dieses Notebook demonstriert **Explainable AI (XAI)** mit der LRP-Methode auf dem MNIST-Datensatz.
# Wir trainieren ein einfaches CNN und erklären anschließend, welche Pixel zur Vorhersage beigetragen haben.
#
# ## Was ist LRP?
#
# LRP (*Layerwise Relevance Propagation*, Bach et al. 2015) erklärt eine Netzvorhersage,
# indem es die Relevanz vom Output rückwärts durch alle Schichten propagiert —
# bis hin zu den einzelnen Eingabepixeln.
#
# **Kernidee:** Die Relevanz ist *konservativ*. Was am Ausgang eines Layers ankommt,
# wird vollständig auf seine Eingaben verteilt:
#
# $$\sum_i R_i^{(l)} = \sum_j R_j^{(l+1)} = f(\mathbf{x})$$
#
# wobei $f(\mathbf{x})$ der Ausgabewert (z.B. Klassenaktivierung) ist.

# %% [markdown]
# ## LRP-Regeln
#
# ### ε-Regel (für Dense-Layer)
#
# Die gebräuchlichste Regel für vollverbundene Schichten:
#
# $$R_i^{(l)} = \sum_j \frac{x_i \cdot w_{ij}}{z_j + \varepsilon \cdot \text{sign}(z_j)} \cdot R_j^{(l+1)}$$
#
# mit der Vorwärtsaktivierung $z_j = \sum_i x_i w_{ij} + b_j$.
#
# Das $\varepsilon > 0$ stabilisiert den Nenner und verhindert Division durch Null.
# Üblich: $\varepsilon = 10^{-6}$.
#
# ### z⁺-Regel (für Conv-Layer)
#
# Bei Faltungsschichten werden positive und negative Gewichte getrennt behandelt,
# um numerische Stabilität bei ReLU-Aktivierungen zu verbessern:
#
# $$R_i^{(l)} = x_i \cdot \left(
#     \frac{\partial}{\partial x_i}
#     \sum_j \frac{z_j^+ + z_j^-}{z_j^+ + z_j^- + \varepsilon \cdot \text{sign}(\cdot)}
#     \cdot R_j^{(l+1)}
# \right)$$
#
# Praktisch implementiert als transponierte Faltung ("`conv_transpose`"):
#
# $$s_j = \frac{R_j^{(l+1)}}{z_j + \varepsilon}, \qquad
#   R_i^{(l)} = x_i \cdot \left[ (s \star W^+)_i + (s \star W^-)_i \right]$$
#
# wobei $W^+ = \max(W, 0)$, $W^- = \min(W, 0)$ und $\star$ die transponierte Faltung ist.
#
# ### MaxPooling: Gradient als exaktes Max-Unpooling
#
# Der Gradient von MaxPooling ist binär: Er ist 1 für das Maximum-Neuron, 0 für alle anderen.
# Damit verhält sich der Gradient wie ein exaktes **Max-Unpooling**:
#
# $$R_i^{(l)} = \begin{cases} R_j^{(l+1)} & \text{wenn } x_i = \max(\text{Pool-Fenster}) \\ 0 & \text{sonst} \end{cases}$$
#
# In TensorFlow lässt sich das elegant über `GradientTape` realisieren.

# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# %% [markdown]
# ## 1. Daten laden und vorverarbeiten

# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0
x_train = x_train[..., np.newaxis]   # (60000, 28, 28, 1)
x_test  = x_test[..., np.newaxis]    # (10000, 28, 28, 1)

y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test,  10)

print(f"Trainingsbilder: {x_train.shape}, Testbilder: {x_test.shape}")


# %% [markdown]
# ## 2. CNN-Modell
#
# Einfaches CNN: zwei Faltungsblöcke (Conv2D + MaxPooling), dann zwei Dense-Layer.
#
# ```
# Input (28×28×1)
#   → Conv2D(16, 3×3, ReLU) → MaxPool(2×2)   → (14×14×16)
#   → Conv2D(32, 3×3, ReLU) → MaxPool(2×2)   → (7×7×32)
#   → Flatten                                  → (1568,)
#   → Dense(128, ReLU)                         → (128,)
#   → Dense(10, Softmax)                       → (10,)
# ```

# %%
def build_model():
    inp = keras.Input(shape=(28, 28, 1), name="input")
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu", name="conv1")(inp)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    out = layers.Dense(10, activation="softmax", name="output")(x)
    return keras.Model(inputs=inp, outputs=out, name="SimpleCNN")

model = build_model()
model.summary()


# %% [markdown]
# ## 3. Training

# %%
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print("\nTraining...")
model.fit(x_train, y_train, epochs=5, batch_size=128,
          validation_split=0.1, verbose=1)

_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest-Accuracy: {test_acc:.4f}")


# %% [markdown]
# ## 4. LRP-Implementierung
#
# ### Algorithmus (Überblick)
#
# ```
# 1. Forward-Pass:  alle Schicht-Aktivierungen speichern
# 2. Startrelevanz: R[target_class] = 1.0, Rest = 0
# 3. Backward-Pass: für jeden Layer (rückwärts):
#      Dense      → ε-Regel
#      Conv2D     → z⁺-Regel (transponierte Faltung)
#      Flatten    → Reshape auf ursprüngliche räumliche Form
#      MaxPooling → Gradient (= Max-Unpooling)
# 4. Ergebnis:      R hat Form des Netz-Inputs (28×28×1)
# ```

# %%
class LRPExplainer:

    def __init__(self, model, epsilon=1e-6):
        self.model   = model
        self.epsilon = epsilon
        self.all_layers = [l for l in model.layers
                           if not isinstance(l, keras.layers.InputLayer)]

    def explain(self, x_input, target_class=None):
        """
        x_input:      numpy (1, 28, 28, 1)
        target_class: int oder None (dann: vorhergesagte Klasse)
        Rückgabe:     heatmap (28, 28), target_class (int)
        """
        x = tf.constant(x_input, dtype=tf.float32)

        # Forward-Pass: alle Aktivierungen sammeln
        activations = [x_input.astype(np.float32)]
        for layer in self.all_layers:
            x = layer(x)
            activations.append(x.numpy())

        pred = activations[-1]
        if target_class is None:
            target_class = int(np.argmax(pred[0]))

        # Startrelevanz: 1.0 für Zielklasse
        R = np.zeros_like(pred)
        R[0, target_class] = 1.0

        # Backward-Pass durch alle Layer
        for i in range(len(self.all_layers) - 1, -1, -1):
            layer = self.all_layers[i]
            x_in  = activations[i]

            if isinstance(layer, layers.Dense):
                R = self._lrp_dense(layer, x_in, R)
            elif isinstance(layer, layers.Conv2D):
                R = self._lrp_conv(layer, x_in, R)
            elif isinstance(layer, layers.Flatten):
                R = R.reshape(x_in.shape)
            elif isinstance(layer, layers.MaxPooling2D):
                R = self._lrp_maxpool(layer, x_in, R)

        heatmap = R.squeeze()   # (28, 28)
        return heatmap, target_class

    # ── MaxPooling: Gradient = exaktes Max-Unpooling ─────────────────────────
    # Formel: R_i^(l) = R_j^(l+1) falls x_i = max(Pool-Fenster), sonst 0

    def _lrp_maxpool(self, layer, x_in, R_upper):
        x_t = tf.constant(x_in, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_t)
            y = layer(x_t)
        grads = tape.gradient(
            y, x_t,
            output_gradients=tf.constant(R_upper, dtype=tf.float32)
        )
        return grads.numpy()

    # ── ε-Regel für Dense ─────────────────────────────────────────────────────
    # R_i^(l) = sum_j [ x_i * w_ij / (z_j + ε·sign(z_j)) ] * R_j^(l+1)

    def _lrp_dense(self, layer, x_in, R_upper):
        W = layer.get_weights()[0].astype(np.float32)   # (in_dim, out_dim)
        b = (layer.get_weights()[1].astype(np.float32)
             if layer.use_bias else np.zeros(W.shape[1], np.float32))

        x      = x_in.astype(np.float32)
        z      = x @ W + b
        sign   = np.where(z >= 0, 1.0, -1.0)
        z_stab = z + self.epsilon * sign

        s       = R_upper / z_stab          # (1, out_dim)
        R_lower = x * (s @ W.T)             # (1, in_dim)
        return R_lower

    # ── z⁺-Regel für Conv2D ──────────────────────────────────────────────────
    # s_j = R_j^(l+1) / z_j,  R_i^(l) = x_i · [(s ⋆ W⁺)_i + (s ⋆ W⁻)_i]

    def _lrp_conv(self, layer, x_in, R_upper):
        W     = layer.get_weights()[0].astype(np.float32)
        W_pos = np.maximum(W, 0.0)
        W_neg = np.minimum(W, 0.0)

        x    = x_in.astype(np.float32)
        R_up = R_upper.astype(np.float32)

        strides = [1, layer.strides[0], layer.strides[1], 1]
        padding = layer.padding.upper()

        z_pos  = self._conv_fwd(x, W_pos, strides, padding)
        z_neg  = self._conv_fwd(x, W_neg, strides, padding)
        z      = z_pos + z_neg
        sign   = np.where(z >= 0, 1.0, -1.0)
        z_stab = z + self.epsilon * sign

        s     = R_up / z_stab
        R_pos = self._conv_bwd(s, W_pos, x.shape, strides, padding)
        R_neg = self._conv_bwd(s, W_neg, x.shape, strides, padding)
        return x * (R_pos + R_neg)

    def _conv_fwd(self, x, W, strides, padding):
        return tf.nn.conv2d(
            tf.constant(x, dtype=tf.float32),
            tf.constant(W, dtype=tf.float32),
            strides=strides, padding=padding
        ).numpy()

    def _conv_bwd(self, grad, W, input_shape, strides, padding):
        return tf.nn.conv2d_transpose(
            tf.constant(grad, dtype=tf.float32),
            tf.constant(W,    dtype=tf.float32),
            output_shape=input_shape,
            strides=strides, padding=padding
        ).numpy()


# %% [markdown]
# ## 5. Visualisierung
#
# Für jedes Beispiel wird angezeigt:
# - **Links**: Original-MNIST-Bild (Graustufen)
# - **Rechts**: LRP-Heatmap (`hot` colormap — hell = hohe Relevanz für die vorhergesagte Klasse)
#
# Titelfarbe: **grün** = korrekte Vorhersage, **rot** = Fehler.

# %%
def plot_lrp(model, x_test, y_test, n_samples=5,
             save_path="lrp_mnist_result.png"):
    explainer = LRPExplainer(model, epsilon=1e-6)

    fig = plt.figure(figsize=(6, 3 * n_samples))
    fig.suptitle(
        "Layerwise Relevance Propagation — MNIST\n"
        "Links: Original  |  Rechts: LRP-Heatmap (hell = hohe Relevanz)",
        fontsize=12, y=1.01
    )
    gs = gridspec.GridSpec(n_samples, 2, hspace=0.45, wspace=0.05)

    for i in range(n_samples):
        x_single   = x_test[i:i+1]
        true_label = int(np.argmax(y_test[i]))

        pred_probs = model.predict(x_single, verbose=0)
        pred_label = int(np.argmax(pred_probs[0]))
        confidence = float(pred_probs[0, pred_label])

        heatmap, target_cls = explainer.explain(x_single, target_class=pred_label)

        # Nur positive Relevanzen, normalisiert auf [0, 1]
        heatmap_vis = np.maximum(heatmap, 0.0)
        vmax = heatmap_vis.max()
        if vmax > 0:
            heatmap_vis /= vmax

        correct = (true_label == pred_label)

        ax_img = fig.add_subplot(gs[i, 0])
        ax_img.imshow(x_single.squeeze(), cmap="gray", vmin=0, vmax=1)
        ax_img.set_title(
            f"Label: {true_label}  Pred: {pred_label}  ({confidence:.0%})",
            fontsize=9, color="green" if correct else "red"
        )
        ax_img.axis("off")

        ax_hm = fig.add_subplot(gs[i, 1])
        im = ax_hm.imshow(heatmap_vis, cmap="hot", vmin=0, vmax=1)
        ax_hm.set_title(f"LRP — Klasse {target_cls}", fontsize=9)
        ax_hm.axis("off")

        if i == n_samples - 1:
            cbar = plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
            cbar.set_label("Relevanz", fontsize=8)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot gespeichert: {save_path}")
    plt.show()


# %% [markdown]
# ## 6. LRP ausführen

# %%
if __name__ == "__main__":
    print("\nLRP-Erklärungen berechnen...")
    plot_lrp(model, x_test, y_test, n_samples=5,
             save_path="lrp_mnist_result.png")

# %%
