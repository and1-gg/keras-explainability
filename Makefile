KERNEL_NAME := py-uv_keras-xai

PY_DIR    := notebooks/py_files
NB_DIR    := notebooks/ipynb_files
HTML_DIR  := notebooks/html_files

.PHONY: help setup kernel test-gpu test-gpu-tensorflow test-gpu-pytorch test-gpu-xgboost \
        notebooks-from-py notebook-from-single-py \
        py-from-notebooks py-from-single-notebook sync-py-and-ipynb \
        run-notebooks html-from-notebooks html-from-single-notebook \
        html-from-notebook-with-quarto clean-notebooks

.DEFAULT_GOAL := help

# Erlaubt: make … Foo.ipynb / Foo.py  (ohne "No rule to make target")
%.ipynb:
	@:

%.py:
	@:

help: ## Zeigt alle verfügbaren Targets mit Beschreibung
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"} {printf "\033[36m%-32s\033[0m %s\n", $$1, $$2}'

setup: ## uv sync + Jupyter-Kernel registrieren
	uv sync
	uv run python -m ipykernel install --user --name $(KERNEL_NAME) --display-name "$(KERNEL_NAME) (uv)"
	uv run python scripts/nvidia_cuda_path.py --install

test-gpu: test-gpu-tensorflow test-gpu-pytorch test-gpu-xgboost ## Prüft GPU-Erkennung für TensorFlow, PyTorch und XGBoost

# pip-nvidia-Wheels liegen unter site-packages/nvidia/*/lib — TF braucht sie in LD_LIBRARY_PATH
# (Pfad erst hier berechnen, nicht beim Parsen des Makefiles — sonst hängt z.B. make help)
test-gpu-tensorflow:
	LD_LIBRARY_PATH="$$(uv run python scripts/nvidia_cuda_path.py --print-ld 2>/dev/null):$${LD_LIBRARY_PATH}" \
	uv run python -c "import tensorflow as tf; print(\"tensor-flow-version: \", tf.__version__); print(\"Num GPUs Available: \", len(tf.config.list_physical_devices('GPU')))"

test-gpu-pytorch:
	uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

test-gpu-xgboost:
	uv run python -c "import numpy as np, xgboost as xgb; print('xgboost-version:', xgb.__version__); print('USE_CUDA:', xgb.build_info()['USE_CUDA']); X=np.random.rand(50, 5); y=np.random.randint(0, 2, 50); xgb.train({'tree_method': 'hist', 'device': 'cuda'}, xgb.DMatrix(X, label=y), num_boost_round=1); print('GPU training: ok')"

notebooks-from-py: ## py_files -> ipynb_files (z.B. nach frischem Clone)
	@mkdir -p "$(NB_DIR)"
	@for f in "$(PY_DIR)"/*.py; do \
		[ -e "$$f" ] || continue; \
		name=$$(basename "$$f" .py); \
		uv run jupytext --to notebook --set-kernel $(KERNEL_NAME) \
			--set-formats "notebooks/ipynb_files//ipynb,notebooks/py_files//py:percent" \
			"$$f" -o "$(NB_DIR)/$$name.ipynb"; \
	done

notebook-from-single-py: ## Einzelnes .py -> .ipynb (z.B. make notebook-from-single-py Foo.py)
	@py_args="$(filter %.py,$(MAKECMDGOALS))"; \
	count=$$(echo "$$py_args" | wc -w); \
	if [ -n "$(PY)" ]; then \
		if [ "$$count" -gt 0 ]; then \
			echo "Bitte nur PY=… oder genau ein *.py-Argument angeben, nicht beides."; \
			exit 1; \
		fi; \
		py="$(PY)"; \
	elif [ "$$count" -eq 0 ]; then \
		echo "Usage: make notebook-from-single-py <name>.py"; \
		echo "   or: make notebook-from-single-py PY=<name>.py"; \
		exit 1; \
	elif [ "$$count" -gt 1 ]; then \
		echo "Bitte genau ein *.py-File angeben (erhalten: $$count)."; \
		exit 1; \
	else \
		py="$$(echo "$$py_args" | awk '{print $$1}')"; \
	fi; \
	case "$$py" in \
		*.py) ;; \
		*) echo "Datei muss auf .py enden: $$py"; exit 1 ;; \
	esac; \
	case "$$py" in \
		*/*) ;; \
		*) py="$(PY_DIR)/$$py" ;; \
	esac; \
	if [ ! -f "$$py" ]; then \
		echo "Python-Datei nicht gefunden: $$py"; \
		exit 1; \
	fi; \
	mkdir -p "$(NB_DIR)"; \
	name=$$(basename "$$py" .py); \
	echo "Convert: $$py -> $(NB_DIR)/$$name.ipynb"; \
	uv run jupytext --to notebook --set-kernel $(KERNEL_NAME) \
		--set-formats "notebooks/ipynb_files//ipynb,notebooks/py_files//py:percent" \
		"$$py" -o "$(NB_DIR)/$$name.ipynb"

py-from-notebooks: ## ipynb_files -> py_files (falls py_files verlorengingen)
	@mkdir -p "$(PY_DIR)"
	@for f in "$(NB_DIR)"/*.ipynb; do \
		[ -e "$$f" ] || continue; \
		name=$$(basename "$$f" .ipynb); \
		uv run jupytext --to py:percent --set-kernel $(KERNEL_NAME) \
			--set-formats "notebooks/ipynb_files//ipynb,notebooks/py_files//py:percent" \
			"$$f" -o "$(PY_DIR)/$$name.py"; \
	done

py-from-single-notebook: ## Einzelnes .ipynb -> .py (z.B. make py-from-single-notebook Foo.ipynb)
	@nb_args="$(filter %.ipynb,$(MAKECMDGOALS))"; \
	count=$$(echo "$$nb_args" | wc -w); \
	if [ -n "$(NB)" ]; then \
		if [ "$$count" -gt 0 ]; then \
			echo "Bitte nur NB=… oder genau ein *.ipynb-Argument angeben, nicht beides."; \
			exit 1; \
		fi; \
		nb="$(NB)"; \
	elif [ "$$count" -eq 0 ]; then \
		echo "Usage: make py-from-single-notebook <name>.ipynb"; \
		echo "   or: make py-from-single-notebook NB=<name>.ipynb"; \
		exit 1; \
	elif [ "$$count" -gt 1 ]; then \
		echo "Bitte genau ein *.ipynb-File angeben (erhalten: $$count)."; \
		exit 1; \
	else \
		nb="$$(echo "$$nb_args" | awk '{print $$1}')"; \
	fi; \
	case "$$nb" in \
		*.ipynb) ;; \
		*) echo "Datei muss auf .ipynb enden: $$nb"; exit 1 ;; \
	esac; \
	case "$$nb" in \
		*/*) ;; \
		*) nb="$(NB_DIR)/$$nb" ;; \
	esac; \
	if [ ! -f "$$nb" ]; then \
		echo "Notebook nicht gefunden: $$nb"; \
		exit 1; \
	fi; \
	mkdir -p "$(PY_DIR)"; \
	name=$$(basename "$$nb" .ipynb); \
	echo "Convert: $$nb -> $(PY_DIR)/$$name.py"; \
	uv run jupytext --to py:percent --set-kernel $(KERNEL_NAME) \
		--set-formats "notebooks/ipynb_files//ipynb,notebooks/py_files//py:percent" \
		"$$nb" -o "$(PY_DIR)/$$name.py"

sync-py-and-ipynb: ## py_files und ipynb_files bidirektional synchronisieren
	@for f in "$(PY_DIR)"/*.py; do \
		[ -e "$$f" ] || continue; \
		uv run jupytext --sync --set-kernel $(KERNEL_NAME) "$$f"; \
	done

run-notebooks: ## Alle Notebooks ausführen (mit Daten populieren)
	@for f in "$(NB_DIR)"/*.ipynb; do \
		[ -e "$$f" ] || continue; \
		uv run jupyter nbconvert --to notebook --execute --inplace "$$f"; \
	done

html-from-notebooks: run-notebooks ## Notebooks ausführen + als HTML exportieren (nbconvert)
	@mkdir -p "$(HTML_DIR)"
	@for f in "$(NB_DIR)"/*.ipynb; do \
		[ -e "$$f" ] || continue; \
		uv run jupyter nbconvert --to html "$$f" --output-dir "$(HTML_DIR)"; \
	done

html-from-single-notebook: ## Einzelnes Notebook ausführen + HTML (z.B. make html-from-single-notebook Foo.ipynb)
	@nb="$(or $(NB),$(firstword $(filter %.ipynb,$(MAKECMDGOALS))))"; \
	if [ -z "$$nb" ]; then \
		echo "Usage: make html-from-single-notebook <name>.ipynb"; \
		echo "   or: make html-from-single-notebook NB=<name>.ipynb"; \
		exit 1; \
	fi; \
	case "$$nb" in \
		*/*) ;; \
		*) nb="$(NB_DIR)/$$nb" ;; \
	esac; \
	if [ ! -f "$$nb" ]; then \
		echo "Notebook nicht gefunden: $$nb"; \
		exit 1; \
	fi; \
	mkdir -p "$(HTML_DIR)"; \
	echo "Execute: $$nb"; \
	uv run jupyter nbconvert --to notebook --execute --inplace "$$nb"; \
	echo "HTML -> $(HTML_DIR)/"; \
	uv run jupyter nbconvert --to html "$$nb" --output-dir "$(HTML_DIR)"

html-from-notebook-with-quarto: ## Notebooks via Quarto ausführen + als HTML exportieren
	@mkdir -p "$(HTML_DIR)"
	@for f in "$(NB_DIR)"/*.ipynb; do \
		[ -e "$$f" ] || continue; \
		quarto render "$$f" --to html --execute --output-dir "$(HTML_DIR)"; \
	done

clean-notebooks: ## ipynb_files und html_files löschen
	rm -f "$(NB_DIR)"/*.ipynb "$(HTML_DIR)"/*.html
