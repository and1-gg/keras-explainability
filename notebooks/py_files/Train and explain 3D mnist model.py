# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: brainage-explainability-1UiqnLSG-py3.10
#     language: python
#     name: brainage-explainability-1uiqnlsg-py3.10
# ---

# %%
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np

data_path = os.path.join(os.path.expanduser('/home/and1/git-repos/keras-explainability'), 'data', '3d-mnist', 
                         'full_dataset_vectors.h5')

#data_path = "/home/and1/git-repos/keras-explainability/data/3d-mnist/full_dataset_vectors.h5"

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

model.fit(train_X, train_y, 
          validation_data=(test_X, test_y), 
          epochs=2, 
          #epochs=100, 
          batch_size=32, 
          shuffle=True,
          callbacks=callbacks)

# %%
import sys
import os

# Get the current working directory
notebook_dir = os.getcwd()

# Construct the path to the 'src' directory
#src_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), '..'))
src_dir = os.path.abspath(os.path.join(os.path.dirname(os.getcwd())))

# Add the 'src' directory to the Python path
if src_dir not in sys.path:
    sys.path.append(src_dir) 

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

image_idx = 0

explanations = np.zeros((10, 16, 16, 16, 1))
predictions = model.predict(train_X[image_idx:image_idx + 1])
print(f'Predictions: {predictions[0]}')

for i in range(10):
    explainer = LRP(model, layer=33, idx=i, strategy=strategy)
    explanations[i] = explainer(train_X[image_idx:image_idx + 1])
    print(f'Sum evidence for {i}: {np.sum(explanations[i])}')

# %%
for i in range(len(explanations)):
    explanations[i] = explanations[i] / np.amax(np.abs(explanations[i]))
    explanations[i] -= explanations[i,0,0,0,0]
    
    fig, ax = plt.subplots(2, 16, figsize=(30, 5))
    fig.suptitle(f'{i} ({round(predictions[0,i], 2)})')
    
    for j in range(16):
        ax[0][j].imshow(train_X[image_idx,:,:,j])
        ax[1][j].imshow(explanations[i,:,:,j], cmap='seismic', clim=(-1, 1))
        ax[0][j].axis('off')
        ax[1][j].axis('off')
        
    plt.show()

# %%
fig, ax = plt.subplots(10, 10, figsize=(20, 20))

for i in range(10):
    for j in range(10):
        ax[i][j].axis('off')
        ax[i][j].imshow(explanations[i,:,:,5] - explanations[j,:,:,5], cmap='seismic', clim=(-1, 1))
        
plt.show()

# %%
