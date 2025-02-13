# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: brainage-explainability-1UiqnLSG-py3.10
#     language: python
#     name: brainage-explainability-1uiqnlsg-py3.10
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
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple


np.random.seed(42)

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

for i in range(10):
    fig, ax = plt.subplots(1, shape, figsize=(15, 2))
    fig.suptitle(y[i])
    for j in range(shape):
        ax[j].imshow(X[i,j], cmap='Greys_r')
        ax[j].axis('off')
    plt.show()

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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

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

model.fit(train_X, train_y, validation_data=(test_X, test_y), batch_size=32, 
          epochs=10)
          #epochs=100)

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
#layer_idx = 9
#layer_idx = 32
layer_idx = 19
explainers = {
    encoder.categories_[0][i]: LRP(model, layer=layer_idx, idx=i, strategy=strategy) \
    for i in range(3)
}

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
            ax[j].imshow(explanation[0,j], cmap='seismic', clim=(-1, 1))
            ax[j].axis('off')
        plt.show()


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
            ax[k].imshow(explanation[k], cmap='seismic', clim=(-1, 1))
            ax[k].axis('off')
            
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
        ax[j].imshow(absolute_difference[j], cmap='seismic', clim=(-1, 1))
        ax[j].axis('off')

    plt.show()


# %%
print(len(model.layers))
