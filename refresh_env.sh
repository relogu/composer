#!/usr/bin/env -i bash
eval "$(grep -vFx '[ -z "$PS1" ] && return' ~/.bashrc)"
LIBRARY_NAME="composer"
PYTHON_VERSION="3.11.12"
pyenv virtualenv-delete $LIBRARY_NAME
pyenv shell $PYTHON_VERSION
python -m pip install --upgrade pip
pyenv virtualenv $PYTHON_VERSION $LIBRARY_NAME
pyenv shell $LIBRARY_NAME
python -m pip install --upgrade pip
python -m pip install ".[dev]"
python -m pre_commit install
python -m pre_commit run --all-files
