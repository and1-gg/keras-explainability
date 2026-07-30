KERNEL_NAME := py-uv_keras-xai

PY_DIR    := notebooks/py_files
NB_DIR    := notebooks/ipynb_files
HTML_DIR  := notebooks/html_files

.PHONY: help setup kernel test-gpu test-gpu-tensorflow test-gpu-pytorch test-gpu-xgboost \
        notebooks-from-py py-from-notebooks sync-py-and-ipynb \
        run-notebooks html-from-notebooks html-from-single-notebook \
        html-from-notebook-with-quarto clean-notebooks

# Erlaubt: make html-from-single-notebook Foo.ipynb  (ohne "No rule to make target")
%.ipynb:
	@:

help: ## Zeigt alle verfügbaren Targets mit Beschreibung
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"} {printf "\033[36m%-32s\033[0m %s\n", $$1, $$2}'

setup: ## uv sync + Jupyter-Kernel registrieren
	uv sync
	uv run python -m ipykernel install --user --name $(KERNEL_NAME) --display-name "$(KERNEL_NAME) (uv)"

test-gpu: test-gpu-tensorflow test-gpu-pytorch test-gpu-xgboost ## Prüft GPU-Erkennung für TensorFlow, PyTorch und XGBoost

test-gpu-tensorflow:
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

py-from-notebooks: ## ipynb_files -> py_files (falls py_files verlorengingen)
	@mkdir -p "$(PY_DIR)"
	@for f in "$(NB_DIR)"/*.ipynb; do \
		[ -e "$$f" ] || continue; \
		name=$$(basename "$$f" .ipynb); \
		uv run jupytext --to py:percent --set-kernel $(KERNEL_NAME) \
			--set-formats "notebooks/ipynb_files//ipynb,notebooks/py_files//py:percent" \
			"$$f" -o "$(PY_DIR)/$$name.py"; \
	done

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
