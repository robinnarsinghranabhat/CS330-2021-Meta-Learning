# base shell
SHELL := /bin/bash

# export env variables
-include .env
export

venv:
	echo "Activating Virtual Environment";
	(\
		. $(VENV_ROOT_DIR); \
		which python; \
	)
	

test_hw1:
	. $(VENV_ROOT_DIR) && PYTHONPATH=$(hw1_module_path) python -m pytest hw1/tests/


test_hw2:
	. $(VENV_ROOT_DIR) && PYTHONPATH=$(hw2_module_path) python -m pytest hw2/tests/