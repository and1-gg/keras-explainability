## License
The code and models in this repo is released under the [CC-BY-NC license](LICENSE.md) for <b>non-commercial</b> use.

## Install Keras-XAI on Denbi-Cloud with Poetry and Conda
```shell
# install pipx
pipx install poetry
# dass peotry die env mit hash namen erzeugt --> setze default config value auf null
poetry config virtualenvs.in-project
# venv installieren
poetry install
# check --> sollte namen mit hash anzeigen zb "brainage-explainability-HAjA2i2R-py3.10"
poetry env list
# notebook verwendung --> ggf resolven "poetry add ipykernel@^6.29.5 notebook@^7.4.7"
poetry add notebook ipykernel
poetry run python -m ipykernel install --user --name brainage-explainability-HAjA2i2R-py3.10
# copy data from local (after downloading it from kaggle)
rsync -aPhvz -e "ssh -i ~/.ssh/id_ed25519 -6" /home/and1/git-repos/keras-explainability/data andreasre@[2001:7c0:801:281:f816:3eff:fe6d:9246]:~/git-repos/keras-explainability/
```
Folgendes zur bashrc adden oer bash_profile
```shell
export PATH="$PATH:/home/and1/.local/bin"
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
```
## Install Keras-XAI on Denbi-Cloud with Pixi
* ...