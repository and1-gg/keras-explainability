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
import sys


sys.path.append('/home/esten/repos/FastSurfer/FastSurferCNN')

# %%
from pyment.data import NiftiDataset, AsyncNiftiGenerator
from pyment.data.preprocessors import NiftiPreprocessor

ixi_folder = os.path.join(os.path.expanduser('~'), 'data', 'IXI')
image_folder = os.path.join(ixi_folder, 'cropped')
project_folder = os.path.join(os.path.expanduser('~'), 'projects', '')
dataset = NiftiDataset.from_folder(image_folder, target='age')
preprocessor = NiftiPreprocessor(sigma=255.)
generator = AsyncNiftiGenerator(
    dataset=dataset,
    preprocessor=preprocessor,
    batch_size=4,
    threads=8
)

# %%
from pyment.models import RegressionSFCN

model = RegressionSFCN(weights='brain-age')

predictions = model.predict(generator)

# %%
from pyment.models import Model

encoder = Model(model.input, model.layers[25].output)

encodings = encoder.predict(generator)

# %%
import numpy as np


ages = dataset.y
predictions = predictions.squeeze()
predictions = predictions[np.where(~np.isnan(ages))]
ages = ages[np.where(~np.isnan(ages))]
delta = predictions - ages
print(f'Brain age delta: {round(np.mean(np.abs(delta)), 2)}')

# %%
from plotly.figure_factory import create_distplot

import matplotlib.pyplot as plt

from explainability import LRP, LRPStrategy

alpha=2
beta=1

strategy = LRPStrategy(
    layers=[
        {'flat': True},
        {'flat': True},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'alpha': alpha, 'beta': beta},
        {'epsilon': 0.25}
    ])

lrp = LRP(model, layer=len(model.layers)-1, idx=0, strategy=strategy)

from pyment.models import Model
import numpy as np
m1 = Model(lrp.input, lrp.layers[57].output)
m2 = Model(lrp.input, lrp.layers[56].output)
m3 = Model(lrp.input, lrp.layers[57].input)

for X, y in generator:
    preds = model.predict(X)
    print(preds[0])
    
    for i in range(29, len(lrp.layers)):
        m = Model(lrp.input, lrp.layers[i].output)
        print(f'{i}: {lrp.layers[i]}')
        e = m.predict(X[:1])[0]
        print(np.sum(e))
        
    explanations = lrp(X[:1])[0].numpy()
    mask = np.zeros(X[0].shape)
    mask[np.where(X[0] != 0)] = 1
    explanations = explanations * mask
    explanations = explanations / np.amax(np.abs(explanations))
    idx = np.argmax(np.abs(explanations))
    idx = np.unravel_index(idx, explanations.shape)
    
    fig, ax = plt.subplots(6, 8, figsize=(15, 15))
    
    for i in range(-4, 4):
        ax[0][i+4].imshow(np.rot90(X[0,idx[0]+i]), cmap='Greys_r')
        ax[0][i+4].axis('off')
        ax[1][i+4].imshow(np.rot90(explanations[idx[0]+i]), cmap='seismic', clim=(-1, 1))
        ax[1][i+4].axis('off')
        ax[2][i+4].imshow(np.rot90(X[0,:,idx[1]+i]), cmap='Greys_r')
        ax[2][i+4].axis('off')
        ax[3][i+4].imshow(np.rot90(explanations[:,idx[1]+i]), cmap='seismic', clim=(-1, 1))
        ax[3][i+4].axis('off')
        ax[4][i+4].imshow(X[0,:,:,idx[2]+i], cmap='Greys_r')
        ax[4][i+4].axis('off')
        ax[5][i+4].imshow(explanations[:,:,idx[2]+i], cmap='seismic', clim=(-1, 1))
        ax[5][i+4].axis('off')

    break

    plt.show()

# %%
m1 = Model(lrp.input, lrp.layers[56].output)
m2 = Model(lrp.input, lrp.layers[57].output)
m3 = Model(lrp.input, lrp.layers[57].input)

a, R = m3.predict(X[:1])
a = a[0]
R2 = R2[0]

print(len(np.where(a.flatten() <= 0)[0]))
print(len(np.where(R.flatten() == 0)[0]))

# %%
from tqdm import tqdm


images = []
labels = []
predictions = []
all_explanations = np.zeros((len(generator),) +  X[0].shape)

generator.reset()

for i, (X, y) in tqdm(enumerate(generator), total=generator.batches):
    for j in range(len(X)):
        image = X[j]
        labels.append(y[j])
        predictions.append(model.predict(np.expand_dims(image, 0))[0])
        expl = lrp.predict(np.expand_dims(image, 0))[0]
        mask = np.zeros(image.shape)
        mask[np.where(image != 0)] = 1
        expl = expl * mask
        all_explanations[i*4 + j] = expl
        images.append(image)

# %%
mean_explanation = np.mean(all_explanations, axis=0)
mean_explanation = mean_explanation / np.amax(np.abs(mean_explanation))
idx = (np.asarray(mean_explanation.shape) / 2).astype(int)

fig, ax = plt.subplots(6, 8, figsize=(15, 15))

for i in range(-4, 4):
    ax[0][i+4].imshow(np.rot90(X[0,idx[0]+i]), cmap='Greys_r')
    ax[0][i+4].axis('off')
    ax[1][i+4].imshow(np.rot90(mean_explanation[idx[0]+i]), cmap='seismic', clim=(-1, 1))
    ax[1][i+4].axis('off')
    ax[2][i+4].imshow(np.rot90(X[0,:,idx[1]+i]), cmap='Greys_r')
    ax[2][i+4].axis('off')
    ax[3][i+4].imshow(np.rot90(mean_explanation[:,idx[1]+i]), cmap='seismic', clim=(-1, 1))
    ax[3][i+4].axis('off')
    ax[4][i+4].imshow(X[0,:,:,idx[2]+i], cmap='Greys_r')
    ax[4][i+4].axis('off')
    ax[5][i+4].imshow(mean_explanation[:,:,idx[2]+i], cmap='seismic', clim=(-1, 1))
    ax[5][i+4].axis('off')

plt.show()

# %%
im1 = all_explanations[0][80]
im1 = im1 / np.amax(np.abs(im1))
im1 = np.rot90(im1)
plt.imshow(im1, cmap='seismic', clim=(-1, 1))
plt.show()

im2 = mean_explanation[80]
im2 = im2 / np.amax(np.abs(im2))
im2 = np.rot90(im2)
plt.imshow(im2, cmap='seismic', clim=(-1, 1))
plt.show()

im3 = im1 - im2
plt.imshow(im3, cmap='seismic', clim=(-1, 1))
plt.show()

brain = images[0][80]
brain = np.rot90(brain)
plt.imshow(brain, cmap='Greys_r', clim=(0, 1))
plt.show()

# %%
import nibabel as nib

from collections import Counter
from copy import copy

from data_loader.conform import conform


fastsurfer_folder = os.path.join(ixi_folder, 'fastsurfer')

regions = {}
sizes = {}
totals = {}

def colorize(mask: np.ndarray):
    colours = np.unique(mask.flatten())
    colours = [col for col in colours if col != 0.]
    cmap = plt.cm.get_cmap('gist_rainbow', len(colours))
    colourized = np.zeros(mask.shape + (4,))

    for colour in colours:
        colourized[np.where(mask == colour)] = cmap(colour)
    
    return colourized

for i in tqdm(range(len(all_explanations))):
    id = dataset.ids[i]
    
    if not (os.path.isfile(os.path.join(image_folder, 'images', f'{id}.nii.gz')) and \
            os.path.isfile(os.path.join(fastsurfer_folder, id, 'mri', 'aparc.DKTatlas+aseg.deep.mgz'))):
        continue
    
    image = nib.load(os.path.join(image_folder, 'images', f'{id}.nii.gz'))
    affine = copy(image.affine)
    header = copy(image.header)
    original_image = copy(image.get_fdata())
    image = conform(image)
    
    explanation = all_explanations[i]
    original_explanation = copy(explanation)
    min_value = np.amin(explanation)
    explanation = explanation - min_value
    max_value = np.amax(explanation)
    explanation = explanation / max_value
    explanation = explanation * 255
    explanation = explanation.astype(np.uint8)
    explanation = nib.Nifti1Image(explanation, affine=affine, header=header)
    explanation = conform(explanation)
    
    mask = nib.load(os.path.join(fastsurfer_folder, id, 'mri', 'aparc.DKTatlas+aseg.deep.mgz'))
    
    image = image.get_fdata()
    explanation = explanation.get_fdata()
    mask = mask.get_fdata()
    
    explanation = explanation / 255.
    explanation = explanation * max_value
    explanation = explanation + min_value
    
    totals[id] = np.sum(explanation)
    
    for region in np.unique(mask.flatten()):
        voxels = np.where(mask == region)
        
        if not region in regions:
            regions[region] = {}
            
        regions[region][id] = np.sum(explanation[voxels])
        
        if not region in sizes:
            sizes[region] = {}
            
        sizes[region][id] = len(voxels[0])

# %%
import pandas as pd

from functools import reduce


fastsurfer_labels = pd.read_csv('~/data/IXI/fastsurfer_labels.csv')

ids = dataset.ids
ages = dataset.y
idx = np.argsort(ages)
sorted_ids = np.asarray(ids)[idx]
sorted_ages = np.asarray(ages)[idx]
ages = {sorted_ids[i]: sorted_ages[i] for i in range(len(sorted_ids))}

for key in regions:
    names = fastsurfer_labels.loc[fastsurfer_labels['id'] == key, 'name'].values
    
    if len(names) == 0:
        if key == 0.0:
            name = 'Background'
        elif key == 2.0:
            name = 'WM'
        else:
            name = key
    else:
        name = names[0]
        
    region_ids = [id for id in sorted_ids if id in regions[key] and id != 'IXI237-Guys-1049-T1']
    region_ages = np.asarray([ages[id] for id in region_ids])
    region_relevance = np.asarray([regions[key][id] for id in region_ids])
    region_sizes = np.asarray([sizes[key][id] for id in region_ids])
    region_totals = np.asarray([totals[id] for id in region_ids])
    age_normalized_relevance = region_relevance / np.abs(region_totals)
    size_normalized_relevance = age_normalized_relevance / region_sizes
    
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(region_ages, region_relevance)
    ax[1].scatter(region_ages, age_normalized_relevance)
    ax[2].scatter(region_ages, size_normalized_relevance)
    fig.suptitle(name)
    
    plt.show()

# %%
import matplotlib.pyplot as plt

from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple


def pad_to_size(image, size: int = 212, value: Tuple = 0):
    vertical = size - image.shape[0]
    top = int(np.ceil(vertical / 2))
    bottom = vertical - top
    
    horizontal = size - image.shape[1]
    left = int(np.ceil(horizontal / 2))
    right = horizontal - left
    
    return np.pad(image, ((top, bottom), (left, right)), constant_values=value)

def concat_horizontal(i1, i2, color=(0, 0, 0)):
    dst = Image.new('RGB', (i1.width + i2.width, i1.height))
    dst.paste(i1, (0, 0))
    dst.paste(i2, (i1.width, 0))
    return dst

def concat_vertical(i1, i2):
    dst = Image.new('RGB', (i1.width, i1.height + i2.height))
    dst.paste(i1, (0, 0))
    dst.paste(i2, (0, i1.height))
    return dst

idx = np.argsort([pred[0] for pred in predictions])
sorted_labels = [labels[i] for i in idx]
sorted_predictions = [predictions[i] for i in idx]
sorted_images = [images[i] for i in idx]
sorted_explanations = [all_explanations[i] for i in idx]

sorted_bitmaps = []

for i in tqdm(range(len(images))):
    expl = sorted_explanations[i]
    expl = expl / np.amax(np.abs(expl))
    expl = expl + 0.5
    
    
    saggital_image = sorted_images[i][84]
    saggital_image = np.rot90(saggital_image)
    saggital_image = pad_to_size(saggital_image)
    saggital_explanations = expl[84]
    saggital_explanations = np.rot90(saggital_explanations)
    saggital_explanations = pad_to_size(saggital_explanations, value=0.5)
    saggital_image = Image.fromarray(np.uint8(cm.Greys_r(saggital_image)*255))
    saggital_explanations = Image.fromarray(np.uint8(cm.seismic(saggital_explanations)*255))
    
    coronal_image = sorted_images[i][:,106]
    coronal_image = np.rot90(coronal_image)
    coronal_image = pad_to_size(coronal_image)
    coronal_explanations = expl[:,106]
    coronal_explanations = np.rot90(coronal_explanations)
    coronal_explanations = pad_to_size(coronal_explanations, value=0.5)
    coronal_image = Image.fromarray(np.uint8(cm.Greys_r(coronal_image)*255))
    coronal_explanations = Image.fromarray(np.uint8(cm.seismic(coronal_explanations)*255))
    
    axial_image = sorted_images[i][:,:,80]
    axial_image = np.rot90(axial_image)
    axial_image = pad_to_size(axial_image)
    axial_explanations = expl[:,:,80]
    axial_explanations = np.rot90(axial_explanations)
    axial_explanations = pad_to_size(axial_explanations, value=0.5)
    axial_image = Image.fromarray(np.uint8(cm.Greys_r(axial_image)*255))
    axial_explanations = Image.fromarray(np.uint8(cm.seismic(axial_explanations)*255))
    
    brain_bitmap = concat_horizontal(concat_horizontal(saggital_image, coronal_image), axial_image)
    explanations_bitmap = concat_horizontal(concat_horizontal(saggital_explanations, coronal_explanations),
                                            axial_explanations)
    bitmap = concat_vertical(brain_bitmap, explanations_bitmap)
    
    draw = ImageDraw.Draw(bitmap)
    font = ImageFont.truetype('arial.ttf', 20)
    draw.text((180, 180),f'Age={sorted_labels[i]:.2f}, brain age {sorted_predictions[i][0]:.2f}', 
              (255,255,255), font=font)
    
    sorted_bitmaps.append(bitmap)
    
sorted_bitmaps[0].save('/home/esten/demo.gif',
               save_all=True, append_images=sorted_bitmaps[1:], optimize=False, duration=40, loop=0)

# %%
import tensorflow as tf
import time

def correlate(a, b):
    numerator = np.sum(a * b)
    sums = np.sum(a ** 2) * np.sum(b ** 2)
    denominator = np.sqrt(sums)
    
    return numerator / denominator

correlations = np.zeros((len(all_explanations), len(all_explanations)))

start = time.time()

for i in tqdm(range(len(all_explanations))):
    for j in range(len(all_explanations)):
        correlations[i,j] = correlate(
            all_explanations[i] / np.abs(np.amax(all_explanations[i])),
            all_explanations[j] / np.abs(np.amax(all_explanations[j]))
        )

# %%
idx = np.argsort(labels)
sorted_correlations = correlations[idx][:,idx]

fig = plt.figure(figsize=(15, 15))
heatmap = plt.imshow(sorted_correlations, cmap='YlGnBu', clim=(0, 1))
plt.colorbar(heatmap)
plt.xticks(np.arange(0, 600, 100), [round(labels[idx[i]], 2) for i in np.arange(0, 600, 100)])
plt.xlabel('Chronological age')
plt.yticks(np.arange(0, 600, 100), [round(labels[idx[i]], 2) for i in np.arange(0, 600, 100)])
plt.ylabel('Chronological age')
plt.savefig('/home/esten/sorted_correlations.png')
plt.show()

# %%
for x in all_explanations:
    print(np.sum(x))

# %%
subject_regions = [regions[region]['IXI012-HH-1211-T1'] for region in regions \
                   if 'IXI012-HH-1211-T1' in regions[region]]
print(np.sum(all_explanations[0]))

# %%

# %%
