#!/usr/bin/env bash
# Legt .venv-tf215/ an und installiert die TF-2.15-Abhängigkeiten darein.
# Die Pakete sind in requirements-tf215.txt definiert (getrennt von pyproject.toml,
# da TF 2.15 und TF 2.18 inkompatibel sind und uv keine gemeinsame Lösung findet).
# Danach Kernel für Jupyter registrieren.
#
# Aufruf (einmalig, vom Repo-Root):
#   bash scripts/setup_tf215_venv.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv-tf215"
KERNEL_NAME="keras-xai-tf215"

echo "==> Erstelle venv unter $VENV_DIR (Python 3.10) ..."
uv venv "$VENV_DIR" --python 3.10

echo "==> Installiere TF-2.15-Abhängigkeiten aus requirements-tf215.txt ..."
uv pip install \
  --python "$VENV_DIR/bin/python" \
  -r "$REPO_ROOT/requirements-tf215.txt"

echo "==> Registriere Jupyter-Kernel '$KERNEL_NAME' ..."
"$VENV_DIR/bin/python" -m ipykernel install \
  --user \
  --name "$KERNEL_NAME" \
  --display-name "keras-xai-tf215 (TF 2.15 / Keras 2)"

echo ""
echo "Fertig. Kernel '$KERNEL_NAME' ist jetzt in Jupyter verfügbar."
echo "Notebook-Kernel wechseln auf: keras-xai-tf215 (TF 2.15 / Keras 2)"
