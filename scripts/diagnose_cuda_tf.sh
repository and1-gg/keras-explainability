#!/usr/bin/env bash
# Ferndiagnose: CUDA-Treiber vs. TF/PyTorch-Wheels (keras-explainability)
# Nutzung (im Repo-Root):
#   bash scripts/diagnose_cuda_tf.sh
# Optional: Ausgabe speichern
#   bash scripts/diagnose_cuda_tf.sh 2>&1 | tee /tmp/cuda_tf_diag.txt

set -u
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
VENV="${ROOT}/.venv"
PY="${VENV}/bin/python"
KERNEL_JSON="${HOME}/.local/share/jupyter/kernels/py-uv_keras-xai/kernel.json"
SC="${VENV}/lib/python3.10/site-packages/sitecustomize.py"

hr() { printf '\n======== %s ========\n' "$*"; }
ok() { printf '[OK] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*"; }

hr "1) System / Treiber"
uname -a || true
echo "HOSTNAME=$(hostname 2>/dev/null || true)"
if command -v nvidia-smi >/dev/null 2>&1; then
  ok "nvidia-smi gefunden: $(command -v nvidia-smi)"
  nvidia-smi || fail "nvidia-smi fehlgeschlagen"
else
  fail "nvidia-smi NICHT gefunden (Treiber/Utils fehlen oder nicht im PATH)"
fi
if [[ -r /proc/driver/nvidia/version ]]; then
  echo "--- /proc/driver/nvidia/version ---"
  cat /proc/driver/nvidia/version
else
  warn "/proc/driver/nvidia/version fehlt"
fi
echo "--- /dev/nvidia* ---"
ls -la /dev/nvidia* 2>&1 | head -20 || true

hr "2) uv / venv"
if [[ -x "$PY" ]]; then
  ok "venv python: $PY"
  "$PY" -V
else
  fail "venv fehlt ($PY) — erst: make setup / uv sync"
  exit 1
fi
command -v uv >/dev/null 2>&1 && uv --version || warn "uv nicht im PATH"

hr "3) Installierte nvidia-* / TF / Torch Pakete"
"$PY" - <<'PY'
import importlib.metadata as md
pkgs = sorted({d.metadata["Name"] for d in md.distributions()
               if d.metadata["Name"].lower().startswith(("nvidia-", "tensorflow", "torch", "keras"))})
for n in pkgs:
    try:
        print(f"  {n}=={md.version(n)}")
    except Exception as e:
        print(f"  {n}: {e}")
PY

hr "4) LD_LIBRARY_PATH Kandidaten (pip nvidia wheels)"
if [[ -x scripts/nvidia_cuda_path.py ]]; then
  LD_PIP="$("$PY" scripts/nvidia_cuda_path.py --print-ld 2>/dev/null || true)"
  echo "LD_PIP=${LD_PIP}"
  echo "--- enthält cu13? ---"
  if echo "${LD_PIP}" | tr ':' '\n' | grep -q '/cu13/'; then
    warn "cu13 (typisch PyTorch CUDA 13) ist im LD-Pfad — kann TF stören"
    echo "${LD_PIP}" | tr ':' '\n' | grep '/cu13/' || true
  else
    ok "kein cu13 im LD-Pfad"
  fi
else
  warn "scripts/nvidia_cuda_path.py fehlt"
  LD_PIP=""
fi

hr "5) Jupyter-Kernel env"
if [[ -f "$KERNEL_JSON" ]]; then
  ok "kernel.json: $KERNEL_JSON"
  KERNEL_JSON="$KERNEL_JSON" "$PY" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["KERNEL_JSON"])
data = json.loads(p.read_text())
env = data.get("env") or {}
ld = env.get("LD_LIBRARY_PATH", "")
print("kernel argv:", data.get("argv"))
print("kernel LD_LIBRARY_PATH set:", bool(ld))
print("kernel LD starts with:", ":".join(ld.split(":")[:3]))
print("kernel LD has cu13:", "/cu13/" in ld)
PY
else
  warn "kernel.json fehlt — make setup ausführen"
fi
if [[ -f "$SC" ]]; then
  ok "sitecustomize.py vorhanden: $SC"
  wc -l "$SC"
else
  warn "sitecustomize.py fehlt (make setup / nvidia_cuda_path.py --install)"
fi

hr "6) TensorFlow Device-Check (verschiedene Envs)"
run_tf() {
  local label="$1"; shift
  echo "--- $label ---"
  env "$@" "$PY" - <<'PY' 2>&1 | tail -40
import os, traceback
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("LD_LIBRARY_PATH set=", bool(os.environ.get("LD_LIBRARY_PATH")))
ld = os.environ.get("LD_LIBRARY_PATH") or ""
print("LD has cu13=", "/cu13/" in ld)
try:
    import tensorflow as tf
    print("tf", tf.__version__)
    print("GPUs", tf.config.list_physical_devices("GPU"))
    print("devices", tf.config.list_physical_devices())
except Exception:
    traceback.print_exc()
    raise SystemExit(2)
PY
}

# A: sauber CPU erzwingen, kein LD
run_tf "A: CUDA_VISIBLE_DEVICES=-1, ohne LD_LIBRARY_PATH" \
  -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=-1

# B: wie Kernel (mit LD inkl. ggf. cu13)
if [[ -n "${LD_PIP}" ]]; then
  run_tf "B: mit pip-LD (wie Kernel), ohne CUDA_VISIBLE" \
    LD_LIBRARY_PATH="$LD_PIP"
  LD_NO_CU13="$(echo "$LD_PIP" | tr ':' '\n' | grep -v '/cu13/' | paste -sd: -)"
  run_tf "C: pip-LD ohne cu13" \
    LD_LIBRARY_PATH="$LD_NO_CU13"
fi

hr "7) Mini-Modell (zeigt den Notebook-Crash)"
run_vgg() {
  local label="$1"; shift
  echo "--- $label ---"
  env "$@" "$PY" - <<'PY' 2>&1 | tail -50
import os, traceback
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    from tensorflow.keras.applications import VGG19
    m = VGG19(weights=None)  # ohne Download
    print("VGG19 OK:", m.name)
except Exception:
    traceback.print_exc()
    raise SystemExit(2)
PY
}

run_vgg "VGG A: CPU erzwingen" -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=-1
if [[ -n "${LD_PIP}" ]]; then
  run_vgg "VGG B: mit pip-LD (Kernel-ähnlich)" LD_LIBRARY_PATH="$LD_PIP"
  LD_NO_CU13="$(echo "$LD_PIP" | tr ':' '\n' | grep -v '/cu13/' | paste -sd: -)"
  run_vgg "VGG C: pip-LD ohne cu13" LD_LIBRARY_PATH="$LD_NO_CU13"
fi

hr "8) Kurzfazit (automatisch)"
"$PY" - <<'PY'
import shutil, os
from pathlib import Path
smi = shutil.which("nvidia-smi")
proc = Path("/proc/driver/nvidia/version").is_file()
print("nvidia-smi:", smi or "FEHLT")
print("nvidia proc driver:", "ja" if proc else "nein")
if not smi and not proc:
    print("FAZIT: Kein nutzbarer NVIDIA-Treiber → TF mit CUDA-Wheels kann crashen.")
    print("       Sofort-Workaround: CUDA_VISIBLE_DEVICES=-1 VOR tensorflow-Import.")
    print("       uv-Neuinstallation hilft NICHT.")
elif smi:
    print("FAZIT: Treiber vorhanden — wenn VGG B fehlschlägt und C/A ok: cu13/LD-Konflikt.")
    print("       Wenn alles fehlschlägt: Treiber zu alt für die CUDA-Runtime der Wheels.")
print("Ausgabe bitte vollständig zurückschicken (oder /tmp/cuda_tf_diag.txt).")
PY

hr "fertig"
echo "Gesamtes Log speichern:  bash scripts/diagnose_cuda_tf.sh 2>&1 | tee /tmp/cuda_tf_diag.txt"
