# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from skimage.io import imread
from skimage.transform import resize
from explainability import LRP, LRPStrategy


data = os.path.join(os.pardir, 'tests', 'data')
model = VGG19(weights='imagenet')

image = np.load(os.path.join(data, 'preprocessed_cat.npy'))
original_image = np.load(os.path.join(data, 'original_cat.npy'))

predictions = model.predict(np.expand_dims(image, 0))
print(f'Predictions: {decode_predictions(predictions, 5)}')

lrp = LRP(model, layer=len(model.layers) - 1, idx=281, epsilon=1e-15)
explanations = lrp(np.expand_dims(image, 0))
explanations = np.sum(explanations, axis=-1)
explanations = explanations / np.amax(np.abs(explanations))

innvestigate = np.load(os.path.join(data, 'cat_explanations_none.npy'))
innvestigate = np.sum(innvestigate, axis=-1)
innvestigate = innvestigate / np.amax(np.abs(innvestigate))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Epsilon=1e-15')

ax[0].imshow(explanations[0], cmap='seismic', clim=(-1, 1))
ax[0].axis('off')
ax[0].set_title('Ours')
ax[1].imshow(innvestigate, cmap='seismic', clim=(-1, 1))
ax[1].axis('off')
ax[1].set_title('Innvestigate')
plt.show()

lrp = LRP(model, layer=len(model.layers) - 1, idx=281, epsilon=0.25)
explanations = lrp(np.expand_dims(image, 0))
explanations = np.sum(explanations, axis=-1)
explanations = explanations / np.amax(np.abs(explanations))

innvestigate = np.load(os.path.join(data, 'cat_explanations_eps.npy'))
innvestigate = np.sum(innvestigate, axis=-1)
innvestigate = innvestigate / np.amax(np.abs(innvestigate))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('epsilon=0.25')

ax[0].imshow(explanations[0], cmap='seismic', clim=(-1, 1))
ax[0].axis('off')
ax[0].set_title('Ours')
ax[1].imshow(innvestigate, cmap='seismic', clim=(-1, 1))
ax[1].axis('off')
ax[1].set_title('Innvestigate')
plt.show()

lrp = LRP(model, layer=len(model.layers) - 1, idx=281, alpha=1, beta=0)
explanations = lrp(np.expand_dims(image, 0))
explanations = np.sum(explanations, axis=-1)
explanations = explanations / np.amax(np.abs(explanations))

innvestigate = np.load(os.path.join(data, 'cat_explanations_a1b0.npy'))
innvestigate = np.sum(innvestigate, axis=-1)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('alpha=1, beta=0')

ax[0].imshow(explanations[0], cmap='seismic', clim=(-1, 1))
ax[0].axis('off')
ax[0].set_title('Ours')
ax[1].imshow(innvestigate, cmap='seismic', clim=(-1, 1))
ax[1].axis('off')
ax[1].set_title('Innvestigate')
plt.show()

lrp = LRP(model, layer=len(model.layers) - 1, idx=281, alpha=2, beta=1)
explanations = lrp(np.expand_dims(image, 0))
explanations = np.sum(explanations, axis=-1)
explanations = explanations / np.amax(np.abs(explanations))

innvestigate = np.load(os.path.join(data, 'cat_explanations_a2b1.npy'))
innvestigate = np.sum(innvestigate, axis=-1)
innvestigate = innvestigate / np.amax(np.abs(innvestigate))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('alpha=2, beta=1')

ax[0].imshow(explanations[0], cmap='seismic', clim=(-1, 1))
ax[0].axis('off')
ax[0].set_title('Ours')
ax[1].imshow(innvestigate, cmap='seismic', clim=(-1, 1))
ax[1].axis('off')
ax[1].set_title('Innvestigate')
plt.show()

alpha=1.5
beta=0.5

strategy = LRPStrategy(
    layers=[
        {'flat': True},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'epsilon': 0.25},
        {'epsilon': 0.25},
        {'epsilon': 0.25},
        {'epsilon': 0.25},
    ]
)

lrp = LRP(model, layer=len(model.layers) - 1, idx=281, strategy=strategy)
explanations = lrp(np.expand_dims(image, 0) + np.amin(image))
explanations = np.sum(explanations, axis=-1)
explanations = explanations / np.amax(np.abs(explanations))

innvestigate = np.load(os.path.join(data, 'cat_explanations_best.npy'))
innvestigate = np.sum(innvestigate, axis=-1)
innvestigate = innvestigate / np.amax(np.abs(innvestigate))

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Best')

ax[0].imshow(original_image)
ax[0].axis('off')
ax[0].set_title('Original image')
ax[1].imshow(explanations[0], cmap='seismic', clim=(-1, 1))
ax[1].axis('off')
ax[1].set_title('Ours')
ax[2].imshow(innvestigate, cmap='seismic', clim=(-1, 1))
ax[2].axis('off')
ax[2].set_title('Innvestigate')
plt.show()

# %%
import requests
import matplotlib.pyplot as plt
import numpy as np

from skimage.io import imread
from skimage.transform import resize


urls = {
    'cat': 'https://cdn.pixabay.com/photo/2014/11/30/14/11/cat-551554__340.jpg',
    'dog': ('https://static.wikia.nocookie.net/naturerules1/images/f/f9/Border-collie-1.jpg/'
            'revision/latest?cb=20210403210149'),
    'bird': 'https://static.independent.co.uk/2021/04/29/22/newFile-3.jpg?quality=75&width=1200&auto=webp',
    'fish': 'https://m.media-amazon.com/images/I/61QN8NWuNlL._AC_SX679_.jpg'
}

images = {}
original_images = {}

for key in urls:
    req = requests.get(urls[key])

    with open(f'/tmp/{key}.jpg', 'wb') as f:
        f.write(req.content)

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(key)
    img = imread(f'/tmp/{key}.jpg')
    img = resize(img, (224, 224), preserve_range=True)
    img = img.astype(np.uint8)
    original_images[key] = img
    plt.imshow(img)
    plt.gca().axis('off')
    plt.show()
    
    img = preprocess_input(img)
    images[key] = img

# %%
for key in images:
    prediction = model.predict(np.expand_dims(images[key], axis=0))
    print(f'Actual class: {key}')
    print('Predictions:')
    print(decode_predictions(prediction, 10))

# %%
from explainability import LRP

idx = [
    ('fish', 1),
    ('dog', 232),
    ('cat', 281),
    ('bird', 94)
]

explainers = {
    p[0]: LRP(model, layer=25, idx=p[1], strategy=strategy) \
    for p in idx
}

fig, ax = plt.subplots(5, 4, figsize=(15, 15))

explanations = np.zeros((4, 4), dtype=object)

keys = [p[0] for p in idx]

ax[0][0].axis('off')

for i in range(4):
    ax[0][i].imshow(original_images[keys[i]])
    ax[0][i].axis('off')

for i in range(len(keys)):
    for j in range(len(keys)):
        explanations[i][j] = explainers[keys[i]](np.expand_dims(images[keys[j]], axis=0))

for i in range(len(explanations)):
    for j in range(len(explanations[i])):
        explanation = np.sum(explanations[i][j][0], axis=-1)
        explanation = explanation / np.amax(np.abs(explanation))
        ax[i+1][j].imshow(explanation, cmap='seismic', clim=(-1, 1))
        ax[i+1][j].axis('off')
        ax[i+1][j].set_ylabel(keys[i])
        
        
plt.show()
