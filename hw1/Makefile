SHELL := /bin/bash

# VENV variables
VENV_NAME ?= ./.venv
KERNAL_NAME ?= meta_learn
VENV_BIN_PATH ?= $(VENV_NAME)/bin
VENV_ACTIVATE = $(VENV_BIN_PATH)/activate
VENV_PYTHON = $(VENV_BIN_PATH)/python

# PYTHON
PYTHON = $(shell which python)
PYTHON_VERSION := $(shell ${PYTHON} --version | cut -d " " -f 2 | cut -d "." -f 1-2)

venv: ## Creates a new virtual environment using venv, with the latest version of pip.
	(\
		test -d $(VENV_NAME) || $(PYTHON) -m venv $(VENV_NAME) ;\
		. $(VENV_ACTIVATE) ;\
		pip install --upgrade pip ;\
		pip -V ;\
		which python ;\
	)

setup : venv ## Creates a new virtual environment and install the development dependencies on it.
	. $(VENV_ACTIVATE) && (\
		pip install -r ./requirements.txt ;\
		python -m ipykernel install --user --name $(KERNAL_NAME) --display-name "$(KERNAL_NAME)" ;\
		python -m ipykernel install --user --name meta_learn --display-name "meta_learn"
	)

clean: ## Removes virtual environment.
	rm -fr $(VENV_NAME)	

clean_kernel:
	yes | jupyter kernelspec uninstall -y $(KERNAL_NAME)

start_tensorboard:
    tensorboard --logdir="C:\Users\Robin\Documents\MOOC\CS330-2021-Meta-Learning\hw1\runs"
