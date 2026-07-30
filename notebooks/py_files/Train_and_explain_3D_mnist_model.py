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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
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

# %%
import sys
import os
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

import h5py
import matplotlib.pyplot as plt
import numpy as np

data_path = os.path.join(os.path.expanduser('~/git-repos/keras-explainability'), 'data', '3d-mnist', 'full_dataset_vectors.h5')

assert os.path.isfile(data_path), \
    'Download the 3d-mnist data from https://www.kaggle.com/daavoo/3d-mnist'

with h5py.File(data_path, 'r') as f:
    train_X = np.reshape(f["X_train"][:], (-1, 16, 16, 16, 1))
    train_y = f["y_train"][:]    
    test_X = np.reshape(f["X_test"][:]  , (-1, 16, 16, 16, 1))
    test_y = f["y_test"][:]

train_X = train_X[:,::-1,:,:]
test_X = test_X[:,::-1,:,:]

def onehot(values: np.ndarray) -> np.ndarray:
    encoded = np.zeros((len(values), 10))

    for i in range(len(values)):
        encoded[i,values[i]] = 1

    return encoded

train_y = onehot(train_y)
test_y = onehot(test_y)

for i in range(5):
    fig, ax = plt.subplots(4, 4)
    fig.suptitle(str(np.argmax(train_y[i])))
    ax = ax.ravel()

    for j in range(16):
        ax[j].imshow(train_X[i,:,:,j], cmap='Greys')
        ax[j].axis('off')

    plt.show()

# %%
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Conv3D, Dense, \
                                    Dropout, Flatten, \
                                    GlobalAveragePooling3D, Input, \
                                    MaxPooling3D
from tensorflow.keras.regularizers import l2

inputs = Input((16, 16, 16, 1), name='inputs')

x = inputs

kernel = (3, 3, 3)
dropout = 0.3
weight_decay = 1e-3
regularizer = l2(weight_decay)

x = Conv3D(32, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv1')(x)
x = BatchNormalization(name='norm1')(x)
x = Activation('relu')(x)
x = Conv3D(32, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv2')(x)
x = BatchNormalization(name='norm2')(x)
x = Activation('relu')(x)
x = MaxPooling3D((2, 2, 2), name='pool1')(x)
x = Conv3D(64, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv3')(x)
x = BatchNormalization(name='norm3')(x)
x = Activation('relu')(x)
x = Conv3D(64, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv4')(x)
x = BatchNormalization(name='norm4')(x)
x = Activation('relu')(x)
x = MaxPooling3D((2, 2, 2), name='pool2')(x)
x = Conv3D(128, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv5')(x)
x = BatchNormalization(name='norm5')(x)
x = Activation('relu')(x)
x = Conv3D(128, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv6')(x)
x = BatchNormalization(name='norm6')(x)
x = MaxPooling3D((2, 2, 2), name='pool3')(x)
x = Conv3D(256, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv7')(x)
x = BatchNormalization(name='norm7')(x)
x = Activation('relu')(x)
x = Conv3D(256, kernel, padding='SAME', activation=None, 
           kernel_regularizer=regularizer, name='conv8')(x)
x = BatchNormalization(name='norm8')(x)
x = Activation('relu')(x)
x = GlobalAveragePooling3D(name='pool4')(x)
x = Dropout(dropout)(x)
x = Dense(10, kernel_regularizer=regularizer, activation=None, name='dense')(x)
x = BatchNormalization(name='norm9')(x)
x = Activation('relu')(x)
x = Dropout(dropout)(x)
x = Dense(10, activation='softmax', name='preds')(x)

model = Model(inputs, x)
model.summary()


# %%
import numpy as np
import matplotlib.pyplot as plt

# 1. Beispieldaten erstellen (z. B. Werte von 0 bis 100)
data = np.linspace(0, 100, 100).reshape((10, 10))

# 2. Daten mit der 'jet'-Farbpalette darstellen
plt.figure(figsize=(6, 5))
img = plt.imshow(data, cmap='jet')

# 3. Colorbar hinzufügen (Zahlenwerte werden automatisch angezeigt)
cbar = plt.colorbar(img)
cbar.set_label('Zahlenwerte', rotation=270, labelpad=15)

plt.title("Beispiel für Colorbar mit 'jet'")
plt.tight_layout()
plt.show()

# %%
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-3),
              metrics=['accuracy'])

callbacks = [
    ReduceLROnPlateau(
        monitor="loss",
        factor=0.1,
        patience=5,
        verbose=1
    )
]

"""
model.fit(train_X, train_y, 
          validation_data=(test_X, test_y), 
          #epochs=2, 
          epochs=100, 
          batch_size=32, 
          shuffle=True,
          callbacks=callbacks)
"""

# %%
from tensorflow.keras.models import load_model

MODEL_DIR = repo_root / "trainings_runs" / "3d_mnist" / "100_epochs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "3d_mnist_cnn.keras"

#model.save(MODEL_PATH)
#print(f"Model gespeichert unter: {MODEL_PATH}")

model = load_model(MODEL_PATH)
print(f"Model geladen von: {MODEL_PATH}")

# %%
from explainability import LRP, LRPStrategy


alpha = 2
beta = 1

strategy = LRPStrategy(
    layers=[
        {'b': True, 'alpha': 1, 'beta': 0},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'epsilon': 0.25},
        {'epsilon': 0.25}
    ]
)

image_idx = 654

explanations = np.zeros((10, 16, 16, 16, 1))
predictions = model.predict(train_X[image_idx:image_idx + 1])
print(f'Predictions: {predictions[0]}')

for i in range(10):
    explainer = LRP(model, layer=len(model.layers)-1, idx=i, strategy=strategy)
    explanations[i] = explainer(train_X[image_idx:image_idx + 1])
    print(f'Sum evidence for {i}: {np.sum(explanations[i])}')

# %%
import matplotlib.pyplot as plt
import numpy as np

num_explanations = len(explanations)

# 1. Erstelle EINE große Figur vor der Schleife
# Zeilen = 2 * Anzahl der Erklärungen, Spalten = 16
fig, ax = plt.subplots(2 * num_explanations, 16, figsize=(30, 5 * num_explanations))

for i in range(num_explanations):
    explanations[i] = explanations[i] / np.amax(np.abs(explanations[i]))
    explanations[i] -= explanations[i, 0, 0, 0, 0]
    
    # Berechne die korrekten Zeilen-Indizes für diese Erklärung
    row_orig = 2 * i
    row_expl = 2 * i + 1
    
    # Optional: Ein Titel pro Block links am Rand (da suptitle sonst alles überschreibt)
    ax[row_orig][0].set_ylabel(f'ID {i} ({round(predictions[0,i], 2)})', 
                               fontsize=16, rotation=0, labelpad=40, va='center')
    
    for j in range(16):
        # Originalbilder in die obere Zeile des Blocks
        ax[row_orig][j].imshow(train_X[image_idx, :, :, j])
        ax[row_orig][j].set_xticks([])
        ax[row_orig][j].set_yticks([])
        
        # Erklärungen in die untere Zeile des Blocks
        ax[row_expl][j].imshow(explanations[i, :, :, j], cmap='seismic', clim=(-1, 1))
        ax[row_expl][j].set_xticks([])
        ax[row_expl][j].set_yticks([])
        
        # Rahmen ausschalten (außer für das erste Bild wegen des Labels links)
        if j > 0:
            ax[row_orig][j].axis('off')
        ax[row_expl][j].axis('off')

# 2. Layout optimieren, damit sich die Zeilen nicht überschneiden
plt.tight_layout()

# 3. Als eine einzige, große Datei speichern
fig.savefig(MODEL_DIR / "all_slices_combined.png", bbox_inches='tight', dpi=150)

# 4. Am Ende einmal anzeigen und Speicher freigeben
plt.show()
plt.close(fig)

# %%
fig, ax = plt.subplots(10, 10, figsize=(20, 20))

for i in range(10):
    for j in range(10):
        ax[i][j].axis('off')
        ax[i][j].imshow(explanations[i,:,:,5] - explanations[j,:,:,5], cmap='seismic', clim=(-1, 1))
        
plt.show()

# %% [markdown]
# # 3d mnist plotten

# %%
import plotly.graph_objects as go

def plot_digit_3d(vol, title="", threshold=0.1):
    vol = np.asarray(vol[..., 0] if vol.ndim == 4 else vol)
    z, y, x = np.where(vol > threshold)
    fig = go.Figure(go.Scatter3d(
        x=x, y=y, z=z, mode="markers",
        marker=dict(size=4, opacity=0.85, color=vol[z, y, x], colorscale="Viridis"),
    ))
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="cube",
                   xaxis=dict(range=[0, 16]), yaxis=dict(range=[0, 16]), zaxis=dict(range=[0, 16])),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.show()

idx = image_idx
plot_digit_3d(train_X[idx], title=f"Label: {np.argmax(train_y[idx])}")

# %% [markdown]
# # Eine 3D-Zahl rein → Prediction

# %%
idx = image_idx
sample = train_X[idx:idx + 1]          # Shape (1, 16, 16, 16, 1)
probs = model.predict(sample, verbose=0)[0]
pred = int(np.argmax(probs))
true = int(np.argmax(train_y[idx]))

print(f"True: {true} | Pred: {pred} | Confidence: {probs[pred]:.4f}")
print(probs)

# %% [markdown]
# # Input + LRP-Explanation in 3D

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from explainability import LRP, LRPStrategy

idx = image_idx
sample = train_X[idx:idx + 1]
true_label = int(np.argmax(train_y[idx]))
probs = model.predict(sample, verbose=0)[0]
pred = int(np.argmax(probs))

strategy = LRPStrategy(layers=[
    {'b': True, 'alpha': 1, 'beta': 0},
    *[{'alpha': 2, 'beta': 1}] * 7,
    {'epsilon': 0.25},
    {'epsilon': 0.25},
])
explainer = LRP(model, layer=len(model.layers) - 1, idx=pred, strategy=strategy)
R = np.array(explainer(sample))[0, ..., 0]
R = R / (np.max(np.abs(R)) + 1e-12)

vol = sample[0, ..., 0]
z, y, x = np.where(vol > 0.1)

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "scene"}, {"type": "scene"}]],
    subplot_titles=(
        f"Input (true={true_label})",
        f"LRP for class {pred} (p={probs[pred]:.2f})",
    ),
)
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z, mode="markers",
    marker=dict(size=4, opacity=0.85, color=vol[z, y, x], colorscale="Viridis"),
    showlegend=False,
), row=1, col=1)
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z, mode="markers",
    marker=dict(
        size=4, opacity=0.9,
        color=R[z, y, x],
        colorscale="RdBu_r",
        cmin=-1, cmax=1,
        showscale=True,
        colorbar=dict(title="Relevance", x=1.02),
    ),
    showlegend=False,
), row=1, col=2)
fig.update_layout(
    title=f"Sample #{idx}",
    scene=dict(aspectmode="cube"),
    scene2=dict(aspectmode="cube"),
    margin=dict(l=0, r=80, t=60, b=0),
)
fig.show()

# %%
data_path = os.path.join(os.path.expanduser('~/git-repos/keras-explainability'), 'data', '3d-mnist', 'full_dataset_vectors.h5')

assert os.path.isfile(data_path), \
    'Download the 3d-mnist data from https://www.kaggle.com/daavoo/3d-mnist'

with h5py.File(data_path, 'r') as f:
    dtrain_x = (f["X_train"][:], (-1, 16, 16, 16, 1))
    dtrain_y = f["y_train"][:]    
    draw_x = f["X_train"][:]
    #dtrain_x = f["X_train"][:]  
    dtest_X = np.reshape(f["X_test"][:]  , (-1, 16, 16, 16, 1))
    dtest_y = f["y_test"][:]

# %%
dtrain_x

# %%
dtrain_y.shape

# %%
draw_x.shape

# %%
16**3

# %%
