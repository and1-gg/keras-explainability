#!/usr/bin/env python
"""Permutiert alle Hirnvoxel ausserhalb der Thalamusmaske.

Gehirnform (Nicht-Null-Voxel) und Grauwerthistogramm bleiben exakt erhalten;
zerstoert wird ausschliesslich die raeumliche Struktur.

    python shuffle_T1.py mni152.nii.gz thalamus_mask.nii.gz out.nii.gz [seed]
"""
import sys
import nibabel as nib
import numpy as np

t1_file, mask_file, out_file = sys.argv[1:4]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0

t1_img = nib.load(t1_file)
mask_img = nib.load(mask_file)
t1 = t1_img.get_fdata()
mask = mask_img.get_fdata()

if t1.shape != mask.shape:
    raise ValueError(f"Formen passen nicht: {t1.shape} vs {mask.shape}. "
                     "Wurde derselbe Crop auf Bild und Maske angewendet?")

# Nur Hirnvoxel (!= 0) ausserhalb des Thalamus permutieren
shuffle_mask = (mask <= 0) & (t1 != 0)
values = t1[shuffle_mask].copy()
np.random.default_rng(seed).shuffle(values)

shuffled = t1.copy()
shuffled[shuffle_mask] = values

assert np.array_equal(shuffled[mask > 0], t1[mask > 0]), "Thalamus wurde veraendert!"
assert np.array_equal(np.sort(shuffled.ravel()), np.sort(t1.ravel())), "Histogramm geaendert!"

nib.save(nib.Nifti1Image(shuffled, t1_img.affine, t1_img.header), out_file)
print(f"{shuffle_mask.sum():,} Voxel permutiert -> {out_file}")
