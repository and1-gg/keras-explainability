#!/usr/bin/env bash
#
# prepare_ixi_subset.sh
#
# Wählt N zufällige Subjects aus der IXI-Datenquelle aus, kopiert für jedes
# Subject die cropped-MRI und die FastSurfer-Segmentierung in eine neue
# Zielstruktur und schreibt eine gefilterte labels.csv.
#
# Zielstruktur:
#   ixi/
#   ├── cropped/
#   │   ├── images/<id>.nii.gz
#   │   └── labels.csv
#   └── fastsurfer/<id>/mri/aparc.DKTatlas+aseg.deep.mgz
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Konfiguration (bei Bedarf anpassen)
# ---------------------------------------------------------------------------
SRC_ROOT="/mnt/ceph/data/ixi/recon"
SRC_LABELS_CSV="${HOME}/data/IXI/cropped/labels.csv"

DEST_ROOT="${HOME}/git-repos/keras-explainability/data/mri/ixi"
DEST_IMAGES_DIR="${DEST_ROOT}/cropped/images"
DEST_LABELS_CSV="${DEST_ROOT}/cropped/labels.csv"
DEST_FASTSURFER_DIR="${DEST_ROOT}/fastsurfer"

N_SUBJECTS=10
RANDOM_SEED=""   # z.B. "42" setzen fuer reproduzierbare Auswahl, leer lassen fuer echten Zufall

CROPPED_FILENAME="cropped.nii.gz"
SEG_SRC_FILENAME="aparc.DKTatlas+aseg.mgz"
SEG_DEST_FILENAME="aparc.DKTatlas+aseg.deep.mgz"

# ---------------------------------------------------------------------------
# Vorbereitung
# ---------------------------------------------------------------------------
if [[ ! -d "${SRC_ROOT}" ]]; then
    echo "FEHLER: Quellverzeichnis nicht gefunden: ${SRC_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${SRC_LABELS_CSV}" ]]; then
    echo "FEHLER: labels.csv nicht gefunden: ${SRC_LABELS_CSV}" >&2
    exit 1
fi

mkdir -p "${DEST_IMAGES_DIR}"
mkdir -p "${DEST_FASTSURFER_DIR}"

# ---------------------------------------------------------------------------
# Kandidaten sammeln: nur Subjects, bei denen BEIDE benoetigten Dateien existieren
# ---------------------------------------------------------------------------
echo "Suche Subjects mit vollstaendigen Daten in ${SRC_ROOT} ..."

candidates=()
for subj_dir in "${SRC_ROOT}"/*/; do
    subj_id="$(basename "${subj_dir}")"
    cropped_path="${subj_dir}mri/${CROPPED_FILENAME}"
    seg_path="${subj_dir}mri/${SEG_SRC_FILENAME}"

    if [[ -f "${cropped_path}" && -f "${seg_path}" ]]; then
        candidates+=("${subj_id}")
    fi
done

n_candidates=${#candidates[@]}
echo "Gefundene vollstaendige Subjects: ${n_candidates}"

if (( n_candidates < N_SUBJECTS )); then
    echo "FEHLER: Nur ${n_candidates} vollstaendige Subjects gefunden, benoetigt werden ${N_SUBJECTS}." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# N zufaellige Subjects auswaehlen
# ---------------------------------------------------------------------------
if [[ -n "${RANDOM_SEED}" ]]; then
    selected=($(printf "%s\n" "${candidates[@]}" | shuf --random-source=<(yes "${RANDOM_SEED}") -n "${N_SUBJECTS}"))
else
    selected=($(printf "%s\n" "${candidates[@]}" | shuf -n "${N_SUBJECTS}"))
fi

echo "Ausgewaehlte Subjects:"
printf '  - %s\n' "${selected[@]}"

# ---------------------------------------------------------------------------
# Dateien kopieren + labels.csv aufbauen
# ---------------------------------------------------------------------------
csv_header="$(head -n 1 "${SRC_LABELS_CSV}")"
echo "${csv_header}" > "${DEST_LABELS_CSV}"

# Spaltenindex von "id" in der CSV ermitteln (1-basiert, komma-separiert)
id_col_idx="$(echo "${csv_header}" | tr ',' '\n' | grep -nx "id" | cut -d: -f1 || true)"
if [[ -z "${id_col_idx}" ]]; then
    echo "FEHLER: Konnte Spalte 'id' im Header von ${SRC_LABELS_CSV} nicht finden: ${csv_header}" >&2
    exit 1
fi

failed=()

for subj_id in "${selected[@]}"; do
    echo "----"
    echo "Verarbeite ${subj_id} ..."

    src_cropped="${SRC_ROOT}/${subj_id}/mri/${CROPPED_FILENAME}"
    src_seg="${SRC_ROOT}/${subj_id}/mri/${SEG_SRC_FILENAME}"

    dest_cropped="${DEST_IMAGES_DIR}/${subj_id}.nii.gz"
    dest_seg_dir="${DEST_FASTSURFER_DIR}/${subj_id}/mri"
    dest_seg="${dest_seg_dir}/${SEG_DEST_FILENAME}"

    mkdir -p "${dest_seg_dir}"

    if ! cp "${src_cropped}" "${dest_cropped}"; then
        echo "  FEHLER beim Kopieren von ${src_cropped}" >&2
        failed+=("${subj_id}")
        continue
    fi

    if ! cp "${src_seg}" "${dest_seg}"; then
        echo "  FEHLER beim Kopieren von ${src_seg}" >&2
        failed+=("${subj_id}")
        continue
    fi

    # Passende Zeile(n) aus labels.csv extrahieren (exakter Match auf id-Spalte)
    row="$(awk -F',' -v col="${id_col_idx}" -v id="${subj_id}" \
        'NR>1 && $col==id {print; found=1} END{exit !found}' "${SRC_LABELS_CSV}")" \
        || { echo "  WARNUNG: Kein labels.csv-Eintrag fuer ${subj_id} gefunden" >&2; failed+=("${subj_id}"); continue; }

    echo "${row}" >> "${DEST_LABELS_CSV}"
    echo "  OK: ${dest_cropped}"
    echo "  OK: ${dest_seg}"
    echo "  OK: labels.csv Eintrag uebernommen"
done

echo "===================================================="
echo "Fertig. Zielverzeichnis: ${DEST_ROOT}"
echo "Neue labels.csv: ${DEST_LABELS_CSV} ($(($(wc -l < "${DEST_LABELS_CSV}") - 1)) Eintraege)"

if (( ${#failed[@]} > 0 )); then
    echo "ACHTUNG: Fehler bei folgenden Subjects:"
    printf '  - %s\n' "${failed[@]}"
    exit 1
fi