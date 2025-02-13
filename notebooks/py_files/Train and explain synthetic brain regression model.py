# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
import os

# Get the current working directory
notebook_dir = os.getcwd()

# Construct the path to the 'src' directory
src_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '..'))

# Add the 'src' directory to the Python path
if src_dir not in sys.path:
    sys.path.append(src_dir) 

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from typing import Any

from explainability import LRP, LRPStrategy


np.random.seed(42)

IMAGE_SIZE = 32
NUM_TUNNELS = 6
MAX_RADIUS = 10
#N = 10
N = 1000

def key(x: Any):
    if isinstance(x, tuple):
        return f'{int(x[0][0])}-{int(x[1][0])}-{int(x[2][0])}'
    else:
        return f'{int(x[0])}-{int(x[1])}-{int(x[2])}'

def drill(brain: np.ndarray, surface: np.ndarray, center: np.ndarray, width: float, 
          inside_keys: set, idx: np.ndarray) -> np.ndarray:
    current_idx = np.random.choice(np.arange(len(surface)))
    current = surface[current_idx]
    direction = center - current
    direction = direction / np.sum(np.abs(direction))
    current_idx = tuple(np.expand_dims(current+direction, -1).astype(int))
    
    while key(current_idx) in inside_keys:
        vertex_radius = np.random.uniform(width // 2, 1)
        vertex_distances = euclidean_distances(idx, np.asarray(current_idx).reshape(1, 3))[:,0]
        pocket = vertex_distances <= vertex_radius
        brain[tuple(idx[pocket].T)] = 0

        next = current + direction
        direction = next - current
        direction[0] = np.random.normal(direction[0], np.abs(direction[0] / 3))
        direction[1] = np.random.normal(direction[0], np.abs(direction[1] / 3))
        direction[2] = np.random.normal(direction[0], np.abs(direction[2] / 3))
        direction = direction / np.sum(np.abs(direction))
        current = next
        current_idx = tuple(np.expand_dims(current, -1).astype(int))
        
    return brain

def create_brain(size: int, width: int, num_tunnels: int = 1):
    brain = np.zeros((size, size, size, 1))
    
    center = np.random.randint(7 * size//16, 9*size//16, 3)
    radius = np.random.randint(size//2-6, size//2-2)
    
    idx = np.asarray(np.meshgrid(*[np.arange(size) for _ in range(3)])).T.reshape(-1, 3)
    distances = euclidean_distances(idx, center.reshape(1, -1))[:,0]
    inside = distances <= radius
    surface = np.isclose(distances, radius, atol=1e-1)
    surface = idx[surface]
    
    brain[tuple(idx[inside].T)] = np.random.uniform(0.25, 1, (len(idx[inside]), 1))
    brain[tuple(idx[surface].T)] = np.random.uniform(0.25, 1, (len(idx[surface]), 1))
    
    inside_keys = set([key(x) for x in idx[inside]]) | set([key(x) for x in surface])
    
    for _ in range(num_tunnels):
        drill(brain, surface, center, width, inside_keys, idx)
    
    return brain

X = []
y = np.random.randint(1, MAX_RADIUS + 1, N)
print("y: ", y)

for i in range(len(y)):
    X.append(create_brain(IMAGE_SIZE, width=y[i], num_tunnels=NUM_TUNNELS))
    print(f'{i+1}/{N}')

    
fig, ax = plt.subplots(10, 8, figsize=(15, 15))

for i in range(1, MAX_RADIUS + 1):
    print("i: ", i)
    idx = np.where(y == i)[0][0]
    #idx = np.where(y == i)[0]
    print("idx: ", idx)
    
    for j in range(8):
        ax[i-1][j].imshow(X[idx][12+j], cmap='Greys_r')
        ax[i-1][j].axis('off')
        
plt.show()

# %%
from plotly.figure_factory import create_distplot

X = np.asarray(X)
y = np.asarray(y).reshape((-1, 1))
train_X = X[:int(0.6*len(X))]
train_y = y[:int(0.6*len(X))]

val_X = X[int(0.6*len(X)):int(0.8*len(X))]
val_y = y[int(0.6*len(X)):int(0.8*len(X))]

test_X = X[int(0.8*len(X)):]
test_y = y[int(0.8*len(X)):]

# %%
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Conv3D, Dense, Dropout, Input, \
                                    GlobalAveragePooling3D, MaxPooling3D
from tensorflow.keras.regularizers import l2


np.random.seed(42)
tf.random.set_seed(42)

regularizer = l2(1e-3)
depths = [32, 64, 128, 256, 256, 64]
activation='relu'
dropout=0.5

inputs = Input((IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
x = inputs

for i in range(3):
    x = Conv3D(depths[i], (3, 3, 3), padding='SAME',
               activation=None, kernel_regularizer=regularizer,
               bias_regularizer=regularizer)(x)

    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPooling3D((2, 2, 2))(x)

x = Conv3D(depths[-1], (1, 1, 1), padding='SAME', activation=None,
           kernel_regularizer=regularizer)(x)

x = BatchNormalization()(x)

x = Activation(activation)(x)
x = GlobalAveragePooling3D()(x)

x = Dense(32, activation=None)(x)
x = Activation('relu')(x)

x = Dropout(dropout)(x)
x = Dense(1, activation=None)(x)

model = Model(inputs, x)

model.summary()

# %%
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss='mse', optimizer=Adam(1e-3), metrics=['mae'])

callbacks = [
    ReduceLROnPlateau(
        monitor="loss",
        factor=0.1,
        patience=10,
        min_lr=1e-5
    ),
    EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=50,
        restore_best_weights=True
    )
]

history = model.fit(train_X, train_y, 
                    validation_data=(val_X, val_y), 
                    batch_size=32,
                    #epochs=500,
                    epochs=2,
                    callbacks=callbacks)


traces = [
    go.Scatter(
        x=np.arange(len(history.history['loss'])),
        y=history.history['loss'],
        name='Training loss'
    ),
    go.Scatter(
        x=np.arange(len(history.history['loss'])),
        y=history.history['val_loss'],
        name='Validation loss'
    )
]

iplot(go.Figure(traces))

# %%
from plotly.subplots import make_subplots


train_predictions = model.predict(train_X)
val_predictions = model.predict(val_X)
test_predictions = model.predict(test_X)

fig = make_subplots(1, 3)

fig.add_trace(
    go.Scatter(
        x=train_y.squeeze(),
        y=train_predictions.squeeze(),
        mode='markers',
        showlegend=False
    )
, row=1, col=1)
fig.add_trace(
    go.Scatter(
        x=[0, 11],
        y=[0, 11],
        mode='lines',
        showlegend=False
    )
, row=1, col=1)

fig.add_trace(
    go.Scatter(
        x=val_y.squeeze(),
        y=val_predictions.squeeze(),
        mode='markers',
        showlegend=False
    )
, row=1, col=2)
fig.add_trace(
    go.Scatter(
        x=[0, 11],
        y=[0, 11],
        mode='lines',
        showlegend=False
    )
, row=1, col=2)

fig.add_trace(
    go.Scatter(
        x=test_y.squeeze(),
        y=test_predictions.squeeze(),
        mode='markers',
        showlegend=False
    )
, row=1, col=3)
fig.add_trace(
    go.Scatter(
        x=[0, 11],
        y=[0, 11],
        mode='lines',
        showlegend=False
    )
, row=1, col=3)

# %%
from explainability import LayerwiseRelevancePropagator, LRPStrategy


strategy = LRPStrategy(
    layers=[
        {'flat': True},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'epsilon': 0.25},
    ]
)

explainer = LayerwiseRelevancePropagator(model, layer=-1, idx=0, strategy=strategy)

for i in range(1, MAX_RADIUS + 1):
    fig, ax = plt.subplots(2, 8, figsize=(15, 3))
    idx = np.where(test_y == i)[0][0]
    explanations = explainer(test_X[idx:(idx + 1)])
    explanations = explanations / np.amax(np.abs(explanations))
    
    for j in range(8):
        ax[0][j].imshow(test_X[idx,12+j], cmap='Greys_r')
        ax[0][j].axis('off')
        ax[1][j].imshow(explanations[0,12+j], cmap='seismic', clim=(-1, 1))
        ax[1][j].axis('off')
        
plt.show()

# %%
from plotly.colors import DEFAULT_PLOTLY_COLORS


np.random.seed(42)

brain = np.zeros((IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))

center = np.asarray([IMAGE_SIZE // 2 for _ in range(3)])
radius = IMAGE_SIZE // 2-2

idx = np.asarray(np.meshgrid(*[np.arange(IMAGE_SIZE) for _ in range(3)])).T.reshape(-1, 3)
distances = euclidean_distances(idx, center.reshape(1, -1))[:,0]
inside = distances <= radius
surface = np.isclose(distances, radius, atol=1e-1)
surface = idx[surface]

brain[tuple(idx[inside].T)] = np.random.uniform(0.25, 1, (len(idx[inside]), 1))
brain[tuple(idx[surface].T)] = np.random.uniform(0.25, 1, (len(idx[surface]), 1))

inside_keys = set([key(x) for x in idx[inside]])

predictions = []

for _ in range(1, 2 * NUM_TUNNELS + 1):
    brain = drill(brain, surface, center, 5, inside_keys, idx)
    predictions.append(model.predict(np.expand_dims(brain, 0))[0,0])
    
fig, ax = plt.subplots(1, 8, figsize=(15, 2))

for i in range(8):
    ax[i].imshow(brain[12+i], cmap='Greys_r')
    ax[i].axis('off')

plt.show()
    
traces = [
    go.Scatter(
        x=np.arange(1, 41),
        y=predictions,
        mode='markers+lines',
        showlegend=False,
        marker={
            'color': DEFAULT_PLOTLY_COLORS[0]
        },
        line={
            'color': DEFAULT_PLOTLY_COLORS[0]
        }
    ),
    go.Scatter(
        x=[1, 2*NUM_TUNNELS],
        y=[5, 5],
        mode='lines',
        showlegend=False,
        line={
            'color': DEFAULT_PLOTLY_COLORS[2],
            'dash': 'dash'
        }
    )
]

layout = go.Layout(
    title={
        'x': 0.5,
        'text': 'Prediction as a function of number of tunnels'
    },
    xaxis={
        'title': 'Number of tunnels'
    },
    yaxis={
        'title': 'Prediction'
    }
)

iplot(go.Figure(traces, layout))

# %%
np.random.seed(42)

brain = np.zeros((IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))

center = np.asarray([IMAGE_SIZE // 2 for _ in range(3)])
radius = IMAGE_SIZE // 2-2

idx = np.asarray(np.meshgrid(*[np.arange(IMAGE_SIZE) for _ in range(3)])).T.reshape(-1, 3)
distances = euclidean_distances(idx, center.reshape(1, -1))[:,0]
inside = distances <= radius
surface = np.isclose(distances, radius, atol=1e-1)
surface = idx[surface]

brain[tuple(idx[inside].T)] = np.random.uniform(0.25, 1, (len(idx[inside]), 1))
brain[tuple(idx[surface].T)] = np.random.uniform(0.25, 1, (len(idx[surface]), 1))

inside_keys = set([key(x) for x in idx[inside]])

predictions = []

for i in range(1, NUM_TUNNELS + 1):
    width = 2 + (6 * (i % 2))
    brain = drill(brain, surface, center, width, inside_keys, idx)
    predictions.append(model.predict(np.expand_dims(brain, 0))[0,0])
    
fig, ax = plt.subplots(1, 8, figsize=(15, 2))

for i in range(8):
    ax[i].imshow(brain[12+i], cmap='Greys_r')
    ax[i].axis('off')

plt.show()

colours = [DEFAULT_PLOTLY_COLORS[(i+1) % 2] for i in range(len(predictions))]
    
traces = [
    go.Scatter(
        x=np.arange(1, NUM_TUNNELS + 1),
        y=predictions,
        mode='markers+lines',
        showlegend=False,
        marker={
            'color': colours
        },
        line={
            'color': DEFAULT_PLOTLY_COLORS[0]
        },
    ),
    go.Scatter(
        x=[1, NUM_TUNNELS],
        y=[5, 5],
        mode='lines',
        showlegend=False,
        line={
            'color': DEFAULT_PLOTLY_COLORS[2],
            'dash': 'dash'
        }
    )
]

layout = go.Layout(
    title={
        'x': 0.5,
        'text': 'Prediction as a function of number of tunnels'
    },
    xaxis={
        'title': 'Number of tunnels'
    },
    yaxis={
        'title': 'Prediction'
    }
)

iplot(go.Figure(traces, layout))

# %%
explainer = LayerwiseRelevancePropagator(model, layer=20, idx=0, strategy=strategy)
explanations = explainer.predict(np.expand_dims(brain, 0))[0]
explanations = explanations / np.amax(np.abs(explanations))

fig, ax = plt.subplots(8, 8, figsize=(15, 8))

for i in range(0, 8, 2):
    for j in range(8):
        idx = ((i // 2) * 8)+ j
        
        ax[i][j].imshow(brain[idx], cmap='Greys_r')
        ax[i][j].axis('off')
        ax[i+1][j].imshow(explanations[idx], cmap='seismic', clim=(-1, 1))
        ax[i+1][j].axis('off')
        
plt.savefig('standard.png')

plt.show()

# %%
encoder = Model(model.input, model.layers[17].output)
encodings = encoder.predict(test_X)
group_idx = np.where(test_y == 5)[0]
group_encodings = encodings[group_idx]
mean_encoding = np.mean(group_encodings, axis=0)
encoding_stddev = np.std(group_encodings, axis=0)

# %%
from explainability import RestructuredLRP
    
restructured_lrp = RestructuredLRP(model, layer=20, idx=0, bottleneck=17, strategy=strategy)
restructured_explanations = restructured_lrp.predict([np.expand_dims(brain, 0), 
                                                      np.expand_dims(mean_encoding, 0)])[0]
restructured_explanations = restructured_explanations / np.amax(np.abs(restructured_explanations))

fig, ax = plt.subplots(4, 8, figsize=(15, 4))


for i in range(8):
    ax[0][i].imshow(brain[12+i], cmap='Greys_r')
    ax[0][i].axis('off')
    ax[1][i].imshow(explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[1][i].axis('off')
    ax[2][i].imshow(restructured_explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[2][i].axis('off')
    ax[3][i].imshow(restructured_explanations[12+i] - explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[3][i].axis('off')

plt.show()

# %%
from explainability import RestructuredLRP
    
restructured_lrp = RestructuredLRP(model, layer=20, idx=0, bottleneck=17, strategy=strategy, threshold=True)
restructured_explanations = restructured_lrp.predict([np.expand_dims(brain, 0), 
                                                      np.expand_dims(mean_encoding, 0),
                                                      np.expand_dims(encoding_stddev, 0)])[0]
restructured_explanations = restructured_explanations / np.amax(np.abs(restructured_explanations))

fig, ax = plt.subplots(4, 8, figsize=(15, 4))


for i in range(8):
    ax[0][i].imshow(brain[12+i], cmap='Greys_r')
    ax[0][i].axis('off')
    ax[1][i].imshow(explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[1][i].axis('off')
    ax[2][i].imshow(restructured_explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[2][i].axis('off')
    ax[3][i].imshow(restructured_explanations[12+i] - explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[3][i].axis('off')

plt.show()

# %%
tmp = Model(model.input, model.layers[17].output)

strategy = LRPStrategy(
    layers=[
        {'flat': True},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1}
    ]
)


fig, ax = plt.subplots(1, 8, figsize=(15, 3))

for i in range(8):
    ax[i].imshow(brain[12+i], cmap='Greys_r')
    ax[i].axis('off')
    
plt.show()

for i in range(32):
    explainer = LayerwiseRelevancePropagator(tmp, layer=17, idx=i, strategy=strategy)
    explanations = explainer.predict(np.expand_dims(brain, 0))[0]

    if np.sum(explanations) == 0:
        continue
    
    explanations = explanations / np.amax(np.abs(explanations))

    fig, ax = plt.subplots(1, 8, figsize=(15, 8))

    for j in range(8):
        ax[j].imshow(explanations[12+j], cmap='seismic', clim=(-1, 1))
        ax[j].axis('off')

    plt.show()

# %%
train_encodings = encoder.predict(train_X)

correlations = [[np.corrcoef(train_encodings[:,i], train_encodings[:,j])[0,1] \
                 for i in range(32)] for j in range(32)]

plt.figure(figsize=(10, 10))
heatmap = plt.imshow(correlations, clim=(0, 1))
plt.colorbar(heatmap)
plt.show()

# %%
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import GlobalMaxPooling3D, Reshape


np.random.seed(42)
tf.random.set_seed(42)

regularizer = l2(1e-3)
depths = [32, 64, 128, 256, 256, 64]
activation='relu'
dropout=0.5

inputs = Input((IMAGE_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
x = inputs

for i in range(3):
    x = Conv3D(depths[i], (3, 3, 3), padding='SAME',
               activation=None, kernel_regularizer=regularizer,
               bias_regularizer=regularizer)(x)

    x = BatchNormalization()(x)
    x = Activation(activation)(x)
    x = MaxPooling3D((2, 2, 2))(x)

x = BatchNormalization()(x)

x = Activation(activation)(x)
x = Reshape((-1,))(x)
x = Dropout(dropout)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Activation(activation)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Activation(activation)(x)

x = Dense(32, activation=None)(x)
x = Activation('relu')(x)

x = Dense(1, activation=None, bias_initializer=Constant([5.]), 
          bias_constraint=MinMaxNorm(min_value=5.0, max_value=5.0))(x)

model = Model(inputs, x)

model.summary()

# %%
np.random.seed(42)
tf.random.set_seed(42)

model.compile(loss='mse', optimizer=Adam(1e-3), metrics=['mae'])

callbacks = [
    ReduceLROnPlateau(
        monitor="loss",
        factor=0.1,
        patience=10,
        min_lr=1e-5
    ),
    EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=50,
        restore_best_weights=True
    )
]

history = model.fit(train_X, train_y, 
                    validation_data=(val_X, val_y), 
                    batch_size=32,
                    epochs=2,
                    #epochs=500,
                    callbacks=callbacks)


traces = [
    go.Scatter(
        x=np.arange(len(history.history['loss'])),
        y=history.history['loss'],
        name='Training loss'
    ),
    go.Scatter(
        x=np.arange(len(history.history['loss'])),
        y=history.history['val_loss'],
        name='Validation loss'
    )
]

iplot(go.Figure(traces))

# %%
from explainability import LayerwiseRelevancePropagator

strategy = LRPStrategy(
    layers=[
        {'flat': True},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'epsilon': 0.25},
        {'epsilon': 0.25}
    ]
)

explainer = LayerwiseRelevancePropagator(model, layer=25, idx=0, strategy=strategy)
explanations = explainer.predict(np.expand_dims(brain, 0))[0]
explanations = explanations / np.amax(np.abs(explanations))

fig, ax = plt.subplots(8, 8, figsize=(15, 8))

for i in range(0, 8, 2):
    for j in range(8):
        idx = ((i // 2) * 8)+ j
        
        ax[i][j].imshow(brain[idx], cmap='Greys_r')
        ax[i][j].axis('off')
        ax[i+1][j].imshow(explanations[idx], cmap='seismic', clim=(-1, 1))
        ax[i+1][j].axis('off')

plt.show()

# %%
tmp = Model(model.input, model.layers[-2].output)

strategy = LRPStrategy(
    layers=[
        {'flat': True},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'epsilon': 0.25}
    ]
)

fig, ax = plt.subplots(1, 8, figsize=(15, 3))

for i in range(8):
    ax[i].imshow(brain[12+i], cmap='Greys_r')
    ax[i].axis('off')
    
plt.show()

for i in range(32):
    explainer = LayerwiseRelevancePropagator(tmp, layer=len(tmp.layers)-1, idx=i, strategy=strategy)
    explanations = explainer.predict(np.expand_dims(brain, 0))[0]

    if np.sum(explanations) == 0:
        continue
    
    explanations = explanations / np.amax(np.abs(explanations))

    fig, ax = plt.subplots(1, 8, figsize=(15, 8))

    for j in range(8):
        ax[j].imshow(explanations[12+j], cmap='seismic', clim=(-1, 1))
        ax[j].axis('off')

    plt.show()

# %%
encoder = Model(model.input, model.layers[-2].output)

train_encodings = encoder.predict(train_X)

correlations = [[np.corrcoef(train_encodings[:,i], train_encodings[:,j])[0,1] \
                 for i in range(32)] for j in range(32)]

plt.figure(figsize=(10, 10))
heatmap = plt.imshow(correlations, clim=(0, 1))
plt.colorbar(heatmap)
plt.show()

# %%
strategy = LRPStrategy(
    layers=[
        {'flat': True},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'alpha': 2, 'beta': 1},
        {'epsilon': 0.25},
        {'epsilon': 0.25}
    ]
)

explainer = LayerwiseRelevancePropagator(model, layer=len(model.layers) - 1, idx=0, strategy=strategy)
explanations = explainer.predict(np.expand_dims(brain, 0))[0]
explanations = explanations / np.amax(np.abs(explanations))

fig, ax = plt.subplots(8, 8, figsize=(15, 8))

for i in range(0, 8, 2):
    for j in range(8):
        idx = ((i // 2) * 8)+ j
        
        ax[i][j].imshow(brain[idx], cmap='Greys_r')
        ax[i][j].axis('off')
        ax[i+1][j].imshow(explanations[idx], cmap='seismic', clim=(-1, 1))
        ax[i+1][j].axis('off')

plt.show()

# %%
from explainability import RestructuredLRP
    
restructured_lrp = RestructuredLRP(model, layer=25, idx=0, bottleneck=23, strategy=strategy)
restructured_explanations = restructured_lrp.predict([np.expand_dims(brain, 0), 
                                                      np.expand_dims(mean_encoding, 0)])[0]
restructured_explanations = restructured_explanations / np.amax(np.abs(restructured_explanations))

fig, ax = plt.subplots(4, 8, figsize=(15, 4))


for i in range(8):
    ax[0][i].imshow(brain[12+i], cmap='Greys_r')
    ax[0][i].axis('off')
    ax[1][i].imshow(explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[1][i].axis('off')
    ax[2][i].imshow(restructured_explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[2][i].axis('off')
    ax[3][i].imshow(restructured_explanations[12+i] - explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[3][i].axis('off')

plt.show()

# %%
encoder = Model(model.input, model.layers[-3].output)
encodings = encoder.predict(test_X)
group_idx = np.where(test_y == 5)[0]
group_encodings = encodings[group_idx]
mean_encoding = np.mean(group_encodings, axis=0)
encoding_stddev = np.std(group_encodings, axis=0)

# %%
from explainability import RestructuredLRP
    
restructured_lrp = RestructuredLRP(model, layer=25, idx=0, bottleneck=23, strategy=strategy, threshold=True)
restructured_explanations = restructured_lrp.predict([np.expand_dims(brain, 0), 
                                                      np.expand_dims(mean_encoding, 0),
                                                      np.expand_dims(encoding_stddev * 8, 0)])[0]
restructured_explanations = restructured_explanations / np.amax(np.abs(restructured_explanations))

fig, ax = plt.subplots(4, 8, figsize=(15, 4))


for i in range(8):
    ax[0][i].imshow(brain[12+i], cmap='Greys_r')
    ax[0][i].axis('off')
    ax[1][i].imshow(explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[1][i].axis('off')
    ax[2][i].imshow(restructured_explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[2][i].axis('off')
    ax[3][i].imshow(restructured_explanations[12+i] - explanations[12+i], cmap='seismic', clim=(-1, 1))
    ax[3][i].axis('off')

plt.show()

# %%

# %%
