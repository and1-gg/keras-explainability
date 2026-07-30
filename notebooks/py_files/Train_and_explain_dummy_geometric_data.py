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
#     display_name: py-uv_keras_xai (uv)
#     language: python
#     name: py-uv_keras_xai
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

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple

import ipynbname

# 2. Notebook-Namen holen
# Unter nbconvert scheitert ipynbname oft (IndexError am Kernel-Connection-File)
try:
    notebook_name = ipynbname.name()
except Exception:
    notebook_name = "Train_and_explain_dummy_geometric_data"

# 3. Pfad zusammensetzen: (root-dir-des-repos) / output/notebooks / notebook_name
target_dir = repo_root / "output/notebooks" / notebook_name

# 4. Ordner erstellen
target_dir.mkdir(parents=True, exist_ok=True)

print(f"Zielordner ist: {target_dir}")

# für wiederholung
np.random.seed(42)


# %%
def generate_square(shape: Tuple[int] = (16, 16, 16)) -> np.ndarray:
    img = np.zeros(shape)
    corner = np.asarray(shape) // 4 + np.random.randint(0, shape[0] // 2, 3)
    side = np.random.randint(shape[1] / 3, shape[1] * 2 / 3)
    
    idx = np.asarray(np.meshgrid(*[np.arange(side) for _ in range(3)])).T.reshape(-1, 3)
    idx += corner
    maxes = np.amax(idx, axis=-1)
    idx = idx[maxes < shape[0]]
    
    img[tuple(idx.T)] = 1
    
    return img

def generate_circle(shape: Tuple[int] = (16, 16, 16)) -> np.ndarray:
    img = np.zeros(shape)
    center = np.random.randint(0, shape[0], 3)
    radius = np.random.randint(shape[0] // 4, shape[0] // 2)

    idx = np.asarray(np.meshgrid(*[np.arange(x) for x in shape])).T.reshape(-1, 3)
    distances = euclidean_distances(idx, center.reshape(1, -1))[:,0]
    inside = distances <= radius
    
    img[tuple(idx[inside].T)] = 1
    
    return img

def generate_noise(shape: Tuple[int] = (16, 16, 16)) -> np.ndarray:
    return np.random.uniform(0, 1, shape)

n = 200
shape = 16
squares = np.asarray([generate_square(shape=(shape, shape, shape)) for _ in range(n)])
circles = np.asarray([generate_circle(shape=(shape, shape, shape)) for _ in range(n)])
noise = np.asarray([generate_noise(shape=(shape, shape, shape)) for _ in range(n)])
X = np.concatenate([squares, circles, noise], axis=0)
y = np.asarray((['square'] * n) + (['circle'] * n) + (['noise'] * n))
idx = np.random.permutation(np.arange(len(X)))
X = np.reshape(X, (-1, shape, shape, shape, 1))
X = X[idx]
y = y[idx]

print(f'X.shape: {X.shape}')
print(f'y.shape: {y.shape}')

n = 10
fig, ax = plt.subplots(n, shape, figsize=(15, 2 * n))

for i in range(n):
    ax[i][shape // 2].set_title(str(y[i]))
    for j in range(shape):
        ax[i][j].imshow(X[i, j], cmap='Greys_r')
        ax[i][j].axis('off')

fig.savefig(target_dir / '1_geometric_samples_overview.png', bbox_inches='tight', dpi=150)
plt.show()
plt.close(fig)

encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1, 1)).toarray()

train_X = X[:300]
train_y = y[:300]
test_X = X[:300]
test_y = y[:300]

# %%
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Conv3D, Dense, Dropout, \
                                    Flatten, GlobalAveragePooling3D, Input, MaxPooling3D, Reshape
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

MODEL_DIR = repo_root / "output" / "notebooks" / notebook_name / "100_epochs"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "geometric_3d_cnn.keras"

input = Input((shape, shape, shape, 1))
x = input

x = Conv3D(8, (3, 3, 3), padding='SAME', activation=None, kernel_regularizer=l2(1e-3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling3D((2, 2, 2))(x)

x = Conv3D(16, (3, 3, 3), padding='SAME', activation=None, kernel_regularizer=l2(1e-3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling3D((2, 2, 2))(x)

x = Conv3D(32, (3, 3, 3), padding='SAME', activation=None, kernel_regularizer=l2(1e-3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling3D((2, 2, 2))(x)

x = Conv3D(32, (3, 3, 3), padding='SAME', activation=None, kernel_regularizer=l2(1e-3))(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling3D((2, 2, 2))(x)

x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(3, activation='softmax')(x)

model = Model(input, x)

model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])

existing_model_path = MODEL_PATH if MODEL_PATH.exists() else next(MODEL_DIR.glob("*.model"), None)
if existing_model_path is not None:
    model = load_model(existing_model_path)
    print(f"Model geladen von: {existing_model_path}")
else:
    model.fit(
        train_X,
        train_y,
        validation_data=(test_X, test_y),
        batch_size=32,
        epochs=100,
    )

    model.save(MODEL_PATH)
    print(f"Model gespeichert unter: {MODEL_PATH}")

model = load_model(MODEL_PATH)
print(f"Model geladen von: {MODEL_PATH}")

# %%
print(len(model.layers))

# %%
from explainability import LRP, LRPStrategy


alpha = 2
beta = 1

strategy = LRPStrategy(
    layers=[
        {'b': True},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'epsilon': 0.5}
    ]
)
#N_layers = 9
#N_layers = 32
N_layers = len(model.layers)-1

for layer in model.layers:
    if "dropout" in layer.name.lower():
        layer.rate = 0.0

explainers = {
    encoder.categories_[0][i]: LRP(model, layer=N_layers, idx=i, strategy=strategy) \
    for i in range(3)
}

# %%
encoder.categories_[0]

# %%
for i in range(10):
    img = test_X[i]
    label = encoder.categories_[0][np.argmax(test_y[i])]

    fig, ax = plt.subplots(1, shape, figsize=(15, 2))
    fig.suptitle(f'Image {label}')
    for i in range(shape):
        ax[i].imshow(img[i], cmap='Greys_r')
        ax[i].axis('off')
    plt.show()

    for classname in encoder.categories_[0]:
        explanation = explainers[classname].predict(np.expand_dims(img, axis=0))
        explanation = explanation / np.amax(np.abs(explanation))

        fig, ax = plt.subplots(1, shape, figsize=(15, 2))
        fig.suptitle(f'{classname} explanation')
        for j in range(shape):
            im = ax[j].imshow(explanation[0,j], cmap='seismic', clim=(-1, 1))
            ax[j].axis('off')
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        plt.show()

fig.savefig(target_dir / '2_geometric_samples_explanations.png', bbox_inches='tight', dpi=150)
plt.show()
plt.close(fig)        


# %%
from scipy.spatial.distance import euclidean


labels = encoder.categories_[0]

square = np.zeros((16, 16, 16, 1))
square[4:12,4:12,4:12,0] = 1

fig, ax = plt.subplots(1, shape, figsize=(15, 2))
fig.suptitle('Square')
for j in range(shape):
    ax[j].imshow(square[j], cmap='Greys_r')
    ax[j].axis('off')
plt.show()

circle = np.zeros((16, 16, 16, 1))
center = (8, 8, 8)
radius = 4

for i in range(16):
    for j in range(16):
        for k in range(16):
            if euclidean((i, j, k), center) <= radius:
                circle[i,j,k,0] = 1
                
fig, ax = plt.subplots(1, shape, figsize=(15, 2))
fig.suptitle('Circle')
for j in range(shape):
    ax[j].imshow(circle[j], cmap='Greys_r')
    ax[j].axis('off')
plt.show()

combinations = [
    np.concatenate([square[:8], circle[8:]], axis=0),
    np.concatenate([circle[:8], square[8:]], axis=0),
    np.concatenate([square[:,:8], circle[:,8:]], axis=1),
    np.concatenate([circle[:,:8], square[:,8:]], axis=1),
    np.concatenate([square[:,:,:8], circle[:,:,8:]], axis=2),
    np.concatenate([circle[:,:,:8], square[:,:,8:]], axis=2),
]

for i in range(len(combinations)):
    fig, ax = plt.subplots(1, shape, figsize=(15, 2))
    prediction = model.predict(np.expand_dims(combinations[i], axis=0))[0]
    fig.suptitle(' '.join([f'{labels[i]}: {prediction[i]:.2f}' for i in range(len(labels))]))
    
    for j in range(shape):
        ax[j].imshow(combinations[i][j], cmap='Greys_r')
        ax[j].axis('off')
        
    plt.show()
        
    for j in range(3):
        fig, ax = plt.subplots(1, shape, figsize=(15, 2))
        classname = encoder.categories_[0][j]
        explanation = explainers[classname].predict(np.expand_dims(combinations[i], axis=0))[0]
        explanation = explanation / np.amax(np.abs(explanation))
        fig.suptitle(f'{classname} explanation')
        
        for k in range(shape):
            im = ax[k].imshow(explanation[k], cmap='seismic', clim=(-1, 1))
            ax[k].axis('off')
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        plt.show()
        
    square_explanation = explainers['square'].predict(np.expand_dims(combinations[i], axis=0))[0]
    square_explanation = square_explanation - np.amin(square_explanation)
    square_explanation = square_explanation / np.amax(square_explanation)
    circle_explanation = explainers['circle'].predict(np.expand_dims(combinations[i], axis=0))[0]
    circle_explanation = circle_explanation - np.amin(circle_explanation)
    circle_explanation = circle_explanation / np.amax(circle_explanation)
    absolute_difference = square_explanation - circle_explanation
    
    fig, ax = plt.subplots(1, shape, figsize=(15, 2))
    fig.suptitle('Square explanation - circle explanation')

    for j in range(shape):
        im = ax[j].imshow(absolute_difference[j], cmap='seismic', clim=(-1, 1))
        ax[j].axis('off')
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.show()


# %%
from scipy.spatial.distance import euclidean

labels = encoder.categories_[0]

square = np.zeros((16, 16, 16, 1))
square[4:12, 4:12, 4:12, 0] = 1

circle = np.zeros((16, 16, 16, 1))
center = (8, 8, 8)
radius = 4
for i in range(16):
    for j in range(16):
        for k in range(16):
            if euclidean((i, j, k), center) <= radius:
                circle[i, j, k, 0] = 1

combinations = [
    np.concatenate([square[:8], circle[8:]], axis=0),
    np.concatenate([circle[:8], square[8:]], axis=0),
    np.concatenate([square[:, :8], circle[:, 8:]], axis=1),
    np.concatenate([circle[:, :8], square[:, 8:]], axis=1),
    np.concatenate([square[:, :, :8], circle[:, :, 8:]], axis=2),
    np.concatenate([circle[:, :, :8], square[:, :, 8:]], axis=2),
]

# Square + Circle + je Combination: Input + 3 Erklärungen + Differenz
n_rows = 2 + len(combinations) * (1 + 3 + 1)
fig, ax = plt.subplots(n_rows, shape, figsize=(16, 2 * n_rows))
row = 0

def plot_row(data, title, cmap='Greys_r', clim=None, colorbar=False):
    global row
    ax[row][shape // 2].set_title(title)
    im = None
    for j in range(shape):
        kwargs = {'cmap': cmap}
        if clim is not None:
            kwargs['clim'] = clim
        im = ax[row][j].imshow(data[j], **kwargs)
        ax[row][j].axis('off')
    if colorbar:
        fig.colorbar(im, ax=ax[row], fraction=0.015, pad=0.01)
    row += 1

plot_row(square, 'Square')
plot_row(circle, 'Circle')

for i, combo in enumerate(combinations):
    prediction = model.predict(np.expand_dims(combo, axis=0))[0]
    title = ' '.join([f'{labels[c]}: {prediction[c]:.2f}' for c in range(len(labels))])
    plot_row(combo, title)

    for classname in labels:
        explanation = explainers[classname].predict(np.expand_dims(combo, axis=0))[0]
        explanation = explanation / np.amax(np.abs(explanation))
        plot_row(explanation, f'{classname} explanation',
                 cmap='seismic', clim=(-1, 1), colorbar=True)

    square_explanation = explainers['square'].predict(np.expand_dims(combo, axis=0))[0]
    square_explanation = square_explanation - np.amin(square_explanation)
    square_explanation = square_explanation / np.amax(square_explanation)
    circle_explanation = explainers['circle'].predict(np.expand_dims(combo, axis=0))[0]
    circle_explanation = circle_explanation - np.amin(circle_explanation)
    circle_explanation = circle_explanation / np.amax(circle_explanation)
    absolute_difference = square_explanation - circle_explanation

    plot_row(absolute_difference, 'Square explanation - circle explanation',
             cmap='seismic', clim=(-1, 1), colorbar=True)

fig.savefig(target_dir / '3_geometric_combinations_explanations.png',
            bbox_inches='tight', dpi=150)
plt.show()
plt.close(fig)

# %% [markdown]
# # 3d plot

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def plot_volume_plotly(vol: np.ndarray, title: str = "", threshold: float = 0.5):
    vol = np.asarray(vol)
    if vol.ndim == 4:
        vol = vol[..., 0]
    z, y, x = np.where(vol > threshold)
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=4, opacity=0.8, color=vol[z, y, x], colorscale="Viridis"),
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x", yaxis_title="y", zaxis_title="z",
            aspectmode="cube",
            xaxis=dict(range=[0, vol.shape[2]]),
            yaxis=dict(range=[0, vol.shape[1]]),
            zaxis=dict(range=[0, vol.shape[0]]),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_random_volumes_plotly(
    X: np.ndarray,
    y: np.ndarray,
    n: int = 10,
    threshold: float = 0.5,
    seed: int = 42,
    class_names=None,
):
    """n zufällige 3D-Volumen als eine interaktive Plotly-Figur."""
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(X), size=min(n, len(X)), replace=False)

    titles = []
    for i in idxs:
        if y.ndim == 2:
            cls = class_names[np.argmax(y[i])] if class_names is not None else int(np.argmax(y[i]))
        else:
            cls = y[i]
        titles.append(f"#{i}: {cls}")

    n_plots = len(idxs)
    n_cols = 5
    n_rows = int(np.ceil(n_plots / n_cols))
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"type": "scene"} for _ in range(n_cols)] for _ in range(n_rows)],
        subplot_titles=titles,
    )

    for k, i in enumerate(idxs):
        row, col = divmod(k, n_cols)
        vol = np.asarray(X[i])
        if vol.ndim == 4:
            vol = vol[..., 0]
        z, yy, x = np.where(vol > threshold)
        fig.add_trace(
            go.Scatter3d(
                x=x, y=yy, z=z,
                mode="markers",
                marker=dict(size=3, opacity=0.85, color=vol[z, yy, x], colorscale="Viridis"),
                showlegend=False,
                name=titles[k],
            ),
            row=row + 1,
            col=col + 1,
        )

    d, h, w = X.shape[1], X.shape[2], X.shape[3]
    scene_layout = dict(
        aspectmode="cube",
        xaxis=dict(range=[0, w], title="x"),
        yaxis=dict(range=[0, h], title="y"),
        zaxis=dict(range=[0, d], title="z"),
    )
    layout_scenes = {f"scene{k+1}" if k else "scene": scene_layout for k in range(n_plots)}
    fig.update_layout(
        title=f"{n_plots} zufällige 3D-Samples",
        height=400 * n_rows,
        width=1200,
        margin=dict(l=0, r=0, t=60, b=0),
        **layout_scenes,
    )
    return fig


fig = plot_random_volumes_plotly(
    X, y, n=10, seed=42, class_names=encoder.categories_[0]
)
fig.write_html(target_dir / "4_geometric_3d_samples.html")
#fig.write_image(target_dir / "4_geometric_3d_samples.png", scale=2)
fig.show(renderer="notebook")
