# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python (pybrainmetrics)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Analyse: Was steckt in `model.keras`?
#
# Dieses Notebook erklärt am Beispiel eines 3D-MNIST-Training-Runs:
#
# 1. **Woher kommt die CNN-Struktur?** (YAML-Config → Python-Code → Keras-Graph)
# 2. **Was landet in der `.keras`-Datei?** (Architektur-JSON + Gewichte)
# 3. **Wie baue ich das Modell wieder auf?** (zwei Wege)
# 4. **Wie prüfe ich, dass es korrekt zusammengesetzt wurde?**
#
# **Start:** Repo-Root, Kernel der Pixi-Umgebung (`pixi run jupyter notebook`).

# %% [markdown]
# ## 0. Setup und Pfade

# %%
from pathlib import Path
import json
import sys
import zipfile

import numpy as np
import tensorflow as tf
from omegaconf import OmegaConf, open_dict

_repo_root = Path.cwd()
if not (_repo_root / "pixi.toml").exists():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "pixi.toml").exists():
            _repo_root = p
            break
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from pybrainmetrics.modeling.train import build_model

RUN_DIR = Path.home() / (
    "data/nn-trainings/3dmnist/number/training_run_17h27m18s_16apr2026"
)
RUN_DIR = RUN_DIR.expanduser().resolve()
MODEL_PATH = RUN_DIR / "model.keras"
CONFIG_PATH = RUN_DIR / "config.yaml"

print(f"Run:    {RUN_DIR}")
print(f"Model:  {MODEL_PATH.exists()} → {MODEL_PATH}")
print(f"Config: {CONFIG_PATH.exists()} → {CONFIG_PATH}")
assert MODEL_PATH.is_file(), f"model.keras fehlt: {MODEL_PATH}"
assert CONFIG_PATH.is_file(), f"config.yaml fehlt: {CONFIG_PATH}"

# %% [markdown]
# ## 1. Die CNN-Struktur entsteht *nicht* erst in der `.keras`-Datei
#
# In diesem Projekt fließen Architektur-Informationen so:
#
# ```text
# configs/model/sfcn_reg_3dmnist.yaml   (Filter, Kernel, Pooling, …)
#         │
#         ▼  Hydra merged → config.yaml im Run-Ordner
# build_model(cfg)  →  RegressionSFCN(...)  →  Conv3D / BN / Pool / Dense
#         │
#         ▼  nach dem Training
# model.save("model.keras")   # speichert Graph + trainierte Gewichte
# ```
#
# Die YAML beschreibt also die Struktur; Keras serialisiert sie beim Speichern
# zusätzlich in die `.keras`-Datei. Zum erneuten Laden braucht man deshalb
# **entweder** die gespeicherte Architektur in der Datei **oder** dieselbe YAML
# + denselben Python-Code (`RegressionSFCN`).

# %%
cfg = OmegaConf.load(CONFIG_PATH)

print("Architektur-Name:", cfg.model.architecture)
print("Modell-Name:     ", cfg.model.name)
print("Input (cropped): ", list(cfg.model.input.shape_cropped))
print("Aktivierung:     ", cfg.model.activation)
print("Pooling:         ", cfg.model.pooling)
print("\nHidden blocks (aus Config = Quelle der CNN-Struktur):")
for i, layer in enumerate(cfg.model.layers.hidden):
    print(
        f"  block-{i}: filters={layer.filters}, "
        f"kernel={list(layer.kernel_size)}, pool={list(layer.pool_size)}"
    )
top = cfg.model.layers.top
print(f"  top:      filters={top.filters}, kernel={list(top.kernel_size)}")

# Prediction-Range (wird in train.build_model aus prediction_ranges gesetzt)
pred_var = cfg.data.prediction_variable
ranges = cfg.data.prediction_ranges[pred_var]
print(f"\nZielvariable '{pred_var}': Range [{ranges.min}, {ranges.max}]")

# %% [markdown]
# ## 2. Was steckt physisch in `model.keras`?
#
# Ab TensorFlow/Keras 3 bzw. dem `.keras`-Format ist die Datei ein **ZIP-Archiv**
# mit u. a.:
#
# | Eintrag | Inhalt |
# |---|---|
# | `config.json` | Layer-Typen, Namen, Parameter, Verbindungen (= Architektur) |
# | `metadata.json` | Keras-/TF-Version |
# | `model.weights.h5` / `variables/` | trainierte Gewichte (Kernel, Bias, BN-Stats, …) |
#
# Die CNN-Struktur steckt also als **serialisierte Layer-Configs** in `config.json`.
# Die Zahlenwerte der Filter liegen in den Weight-Dateien.

# %%
print(f"Dateigröße: {MODEL_PATH.stat().st_size / 1e6:.2f} MB")
print(f"Ist ZIP?    {zipfile.is_zipfile(MODEL_PATH)}\n")

with zipfile.ZipFile(MODEL_PATH, "r") as zf:
    names = zf.namelist()
    print("Inhalt von model.keras:")
    for n in sorted(names):
        info = zf.getinfo(n)
        print(f"  {n:40s}  {info.file_size:10d} Bytes")

    # Architektur-JSON laden (Pfad kann je nach TF-Version leicht variieren)
    config_members = [n for n in names if n.endswith("config.json")]
    meta_members = [n for n in names if n.endswith("metadata.json")]

    if meta_members:
        meta = json.loads(zf.read(meta_members[0]))
        print("\nmetadata.json:")
        print(json.dumps(meta, indent=2)[:800])

    if config_members:
        arch = json.loads(zf.read(config_members[0]))
        print("\nconfig.json – oberste Keys:", list(arch.keys()) if isinstance(arch, dict) else type(arch))
        # Kompakte Übersicht: Layer-Klassen und Namen
        cfg_inner = arch.get("config", arch) if isinstance(arch, dict) else {}
        layers = cfg_inner.get("layers") or arch.get("layers") or []
        print(f"\nAnzahl Layer-Einträge in config.json: {len(layers)}")
        print("Erste Layer (class_name / name):")
        for layer in layers[:12]:
            if isinstance(layer, dict):
                cname = layer.get("class_name") or layer.get("module")
                lcfg = layer.get("config", {})
                lname = lcfg.get("name", "?") if isinstance(lcfg, dict) else "?"
                print(f"  {cname:30s}  name={lname}")
        if len(layers) > 12:
            print(f"  … (+{len(layers) - 12} weitere)")

# %% [markdown]
# ## 3. Zwei Wege, aus der Datei wieder ein Modell zu machen
#
# ### Weg A – `tf.keras.models.load_model` (Architektur aus der Datei)
#
# Keras liest `config.json`, baut die Layer neu und lädt die Gewichte.
# Bei **Custom Models** (hier: `RegressionSFCN`) braucht man oft
# `custom_objects=...` oder die Klasse muss `@keras.saving.register_keras_serializable`
# sein. In diesem Repo ist das **nicht** der bevorzugte Weg.
#
# ### Weg B – Config + `build_model` + `load_weights` (**empfohlen hier**)
#
# 1. Architektur aus `config.yaml` + Python (`RegressionSFCN`) neu bauen  
# 2. Nur die Gewichte aus `model.keras` laden  
#
# Vorteil: derselbe Code wie beim Training; `postprocess()` bleibt erhalten;
# Layer-Namen müssen zur gespeicherten Datei passen.

# %% [markdown]
# ### 3a. Weg A versuchen (`load_model`)

# %%
loaded_via_keras = None
load_error = None
try:
    loaded_via_keras = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    print("load_model OK")
    print(f"  Typ:   {type(loaded_via_keras)}")
    print(f"  Name:  {loaded_via_keras.name}")
    print(f"  Layers:{len(loaded_via_keras.layers)}")
except Exception as exc:  # noqa: BLE001 – Analyse: Fehler soll sichtbar sein
    load_error = exc
    print("load_model fehlgeschlagen (häufig bei Custom Model ohne Registrierung):")
    print(f"  {type(exc).__name__}: {exc}")

# %% [markdown]
# ### 3b. Weg B: Architektur aus Config, Gewichte aus `.keras`

# %%
model = build_model(cfg)
print(f"Frisch gebaut: {model.name}  ({type(model).__name__})")
print(f"Input shape:   {model.input_shape}")
print(f"Output shape:  {model.output_shape}")
print(f"Parameter:     {model.count_params():,}")

# Zufallsgewichte merken, um zu prüfen, dass load_weights wirklich greift
weights_before = [w.copy() for w in model.get_weights()]

model.load_weights(str(MODEL_PATH))
weights_after = model.get_weights()

delta_first = float(np.mean(np.abs(weights_after[0] - weights_before[0])))
print(f"\nMittleres |Δ| der ersten Weight-Matrix: {delta_first:.6g}")
if delta_first < 1e-9:
    raise RuntimeError(
        "load_weights hat nichts verändert – Layer-Namen/Shapes passen nicht zur Datei."
    )
print("→ Gewichte wurden erfolgreich aus model.keras übernommen.")

# %% [markdown]
# ## 4. Verifikation: Ist das Modell „richtig“ zusammengesetzt?
#
# Checkliste:
#
# 1. **Gewichts-Delta** nach `load_weights` (oben) – muss ≫ 0 sein  
# 2. **Layer-Namen** frisch gebaut vs. in der Datei / vs. `load_model`  
# 3. **Weight-Shapes** müssen 1:1 übereinstimmen  
# 4. **Forward-Pass**: gleicher Dummy-Input → gleiche Ausgabe (numerisch)  
# 5. Optional: Vergleich mit `ds_summary.nc` / Scatter-Notebook

# %% [markdown]
# ### 4.1 Layer-Übersicht des per Config gebauten Modells

# %%
print(f"{'Layer':40s} {'Output shape':28s} {'Params':>10s}")
print("-" * 82)
for layer in model.layers:
    shape = getattr(layer, "output_shape", None)
    n_params = layer.count_params()
    print(f"{layer.name:40s} {str(shape):28s} {n_params:10,d}")

# %% [markdown]
# ### 4.2 Weight-Shapes und – falls `load_model` ging – numerischer Vergleich

# %%
print("Weight-Tensoren im gebauten Modell:")
for i, w in enumerate(model.weights):
    print(f"  [{i:2d}] {w.name:50s}  shape={tuple(w.shape)}")

if loaded_via_keras is not None:
    names_build = [l.name for l in model.layers]
    names_load = [l.name for l in loaded_via_keras.layers]
    print("\nLayer-Namen identisch?", names_build == names_load)
    if names_build != names_load:
        only_build = set(names_build) - set(names_load)
        only_load = set(names_load) - set(names_build)
        print("  nur in build_model:", sorted(only_build)[:20])
        print("  nur in load_model: ", sorted(only_load)[:20])

    wb = model.get_weights()
    wl = loaded_via_keras.get_weights()
    print(f"Anzahl Weight-Arrays: build={len(wb)}  load_model={len(wl)}")
    if len(wb) == len(wl):
        max_abs = max(float(np.max(np.abs(a - b))) for a, b in zip(wb, wl))
        print(f"Max |build − load_model| über alle Weights: {max_abs:.6g}")
        print("OK" if max_abs < 1e-5 else "ABWEICHUNG – genauer prüfen")
    else:
        print("Unterschiedliche Anzahl Weight-Arrays – Shapes/Layer prüfen.")
else:
    print("\nKein load_model-Vergleich möglich (siehe Fehler oben).")
    print("Weg B (build_model + load_weights) ist der Projekt-Standard.")

# %% [markdown]
# ### 4.3 Forward-Pass-Check (Dummy-Volume)
#
# Gleiches Input → gleiche Prediction. Wenn `load_model` fehlgeschlagen ist,
# vergleichen wir nur Stabilität (zweimal `predict` auf dem geladenen Modell).

# %%
rng = np.random.default_rng(0)
# Input-Shape ohne Batch: z. B. (16, 16, 16, 1)
in_shape = tuple(s if s is not None else 1 for s in model.input_shape[1:])
x = rng.random((2, *in_shape), dtype=np.float32)

y1 = model.predict(x, verbose=0)
y2 = model.predict(x, verbose=0)
print(f"Input batch shape: {x.shape}")
print(f"Output shape:      {y1.shape}")
print(f"Predictions:       {np.squeeze(y1)}")
print(f"Deterministisch?   max|y1−y2| = {float(np.max(np.abs(y1 - y2))):.3e}")

if loaded_via_keras is not None:
    y_load = loaded_via_keras.predict(x, verbose=0)
    diff = float(np.max(np.abs(np.squeeze(y1) - np.squeeze(y_load))))
    print(f"max|build_model − load_model| Output: {diff:.6g}")
    print("Forward-Pass übereinstimmend." if diff < 1e-5 else "Abweichung im Forward-Pass!")

# postprocess gehört zur Custom-Klasse (Weg B)
if hasattr(model, "postprocess"):
    print("postprocess(y1):", model.postprocess(y1))

# %% [markdown]
# ### 4.4 Kurzfassung der Checks

# %%
checks = {
    "model.keras existiert": MODEL_PATH.is_file(),
    "config.yaml existiert": CONFIG_PATH.is_file(),
    "build_model + load_weights ändert Gewichte": delta_first > 1e-9,
    "Forward-Pass deterministisch": float(np.max(np.abs(y1 - y2))) < 1e-6,
}
if loaded_via_keras is not None:
    checks["load_model erfolgreich"] = True
    checks["Weights build ≈ load_model"] = max_abs < 1e-5
else:
    checks["load_model erfolgreich"] = False

print("Verifikations-Checkliste:")
for name, ok in checks.items():
    print(f"  [{'OK' if ok else '—'}] {name}")

print(
    "\nFazit: Die CNN-Struktur kommt aus der YAML/dem Python-Code; "
    "model.keras speichert sie zusätzlich als config.json + Gewichte. "
    "In pybrainmetrics: Architektur neu bauen (build_model), dann load_weights."
)

# %% [markdown]
# ## Anhang: Wo im Code gespeichert wird
#
# In `pybrainmetrics.main` nach dem Training:
#
# ```python
# model.save(os.path.join(run_dir, "model.keras"))
# ```
#
# Beim erneuten Nutzen (Prediction-Scripts / Scatter-Notebook):
#
# ```python
# model = build_model(cfg)
# model.load_weights(str(run_dir / "model.keras"))
# ```
#
# Die Layer-Namen (`sfcn-reg-3dmnist_block-0_conv`, …) müssen zwischen Speichern
# und Laden identisch sein – sie werden aus `cfg.model.name` und dem
# Block-Index in `SFCN.__init__` erzeugt.
