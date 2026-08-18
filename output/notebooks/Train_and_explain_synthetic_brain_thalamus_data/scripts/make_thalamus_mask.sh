#!/usr/bin/env bash
# Thalamusmaske im MNI152-Raum erzeugen.
# Voraussetzung: FreeSurfer/FastSurfer-Lauf liegt vor (aseg + brainmask).
set -euo pipefail

SUBJECT="${1:?Usage: make_thalamus_mask.sh <subject-id> <recon-dir> <out-dir>}"
RECON_DIR="${2:?}"
OUT_DIR="${3:?}"
MNI="${FSLDIR}/data/standard/MNI152_T1_1mm.nii.gz"

mkdir -p "${OUT_DIR}"

# 1) .mgz -> .nii.gz, denn FSL liest kein MGZ
mri_convert "${RECON_DIR}/${SUBJECT}/mri/brainmask.mgz" "${OUT_DIR}/brainmask.nii.gz"
mri_convert "${RECON_DIR}/${SUBJECT}/mri/aseg.mgz"      "${OUT_DIR}/aseg.nii.gz"

# 2) einheitliche Achsenkonvention
fslreorient2std "${OUT_DIR}/brainmask.nii.gz" "${OUT_DIR}/brainmask_reoriented.nii.gz"
fslreorient2std "${OUT_DIR}/aseg.nii.gz"      "${OUT_DIR}/aseg_reoriented.nii.gz"

# 3) T1 auf MNI152 registrieren. -dof 6 = nur Rotation/Translation, KEINE Skalierung,
#    damit die Volumina erhalten bleiben. -omat speichert die Matrix - das ist der Kern.
flirt -in  "${OUT_DIR}/brainmask_reoriented.nii.gz" \
      -ref "${MNI}" \
      -out "${OUT_DIR}/mni152.nii.gz" \
      -dof 6 \
      -omat "${OUT_DIR}/T1_to_mni152.mat"

# 4) Segmentierung mit DERSELBEN Matrix transformieren.
#    nearestneighbour ist zwingend: Labelnummern darf man nicht interpolieren.
flirt -in  "${OUT_DIR}/aseg_reoriented.nii.gz" \
      -ref "${MNI}" \
      -out "${OUT_DIR}/aseg_mni152.nii.gz" \
      -applyxfm -init "${OUT_DIR}/T1_to_mni152.mat" \
      -interp nearestneighbour

# 5) Thalamus extrahieren: 10 = links, 49 = rechts
fslmaths "${OUT_DIR}/aseg_mni152.nii.gz" -thr 10 -uthr 10 -bin "${OUT_DIR}/thal_left.nii.gz"
fslmaths "${OUT_DIR}/aseg_mni152.nii.gz" -thr 49 -uthr 49 -bin "${OUT_DIR}/thal_right.nii.gz"

# 6) beide Seiten zu einer Maske vereinen
fslmaths "${OUT_DIR}/thal_left.nii.gz" -add "${OUT_DIR}/thal_right.nii.gz" -bin \
         "${OUT_DIR}/thalamus_mask.nii.gz"

echo "Volumen des Thalamus (mm^3):"
fslstats "${OUT_DIR}/thalamus_mask.nii.gz" -V

# 7) Immer visuell gegenpruefen!
echo "Pruefen mit: fsleyes ${OUT_DIR}/mni152.nii.gz ${OUT_DIR}/thalamus_mask.nii.gz -cm red"
