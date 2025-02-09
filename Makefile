# Redefine OS dependent commands for Windows or Linux
# WARNING: Linux commands haven't been tested
VENV = $(call FixPath, ./venv/)
VENV_ACTIVATE = $(call FixPath, $(VENV)Scripts/activate)
PYTHON = $(call FixPath, $(VENV)Scripts/python)
PIP = $(call FixPath, $(VENV)Scripts/pip)
SRC = $(call FixPath, ./src/)
PYCACHE = $(call FixPath, $(SRC)__pycache__)
TESTS = $(call FixPath, ./tests/)
TEST_LOGS = $(call FixPath, $(wildcard $(TESTS)logs/*.log))
DOCS = $(call FixPath, ./docs/)
DOCS_SOURCE = $(call FixPath, $(DOCS)/source)
DOCS_BUILD = $(call FixPath, $(DOCS)/build)
DOCS_INDEX_HTML = $(call FixPath, $(DOCS_BUILD)/html/index.html)

PYTHON_SRC_FILES = $(addprefix $(SRC), \
	svg_generators.py \
	svg_parsers.py \
	svg_writers.py \
	free_cad_object_wrappers.py \
	svg_validators.py \
	svg_file.py \
	free_cad_object_wrappers.py \
	freecad_sketch_readers.py \
	trigonometry.py \
	freecad_sketcher_to_svg_translators.py \
	)

PYTHON_TEST_FILES = $(addprefix $(TESTS), \
	svg_d_attribute_parsing_test.py \
	test_svg_generators.py \
	test_svg_writers.py \
	test_svg_validators.py \
	test_svg_file.py \
	test_freecad_object_wrappers.py \
	test_freecad_sketch_readers.py \
	test_trigonometry.py \
	test_freecad_sketcher_to_svg_translators.py \
	test_freecad_svg_file.py \
	)


ifdef OS
	RMDIR = rd  /s /q
	RM = del /Q
	FixPath = $(subst /,\,$1)
else
	ifeq ($(shell uname), Linux)
		RMDIR = rm -rf
		RM = rm -f
		FixPath = $1
	endif
endif

all: docs

#$(TEST_LOGS)
test: 
	$(PYTHON) $(call FixPath, $(TESTS)svg_d_attribute_parsing_test.py)
	$(PYTHON) $(call FixPath, $(TESTS)test_svg_generators.py)
	$(PYTHON) $(call FixPath, $(TESTS)test_svg_validators.py)
	$(PYTHON) $(call FixPath, $(TESTS)test_svg_writers.py)
	$(PYTHON) $(call FixPath, $(TESTS)test_svg_file.py)
	$(PYTHON) $(call FixPath, $(TESTS)test_freecad_object_wrappers.py)
	$(PYTHON) $(call FixPath, $(TESTS)test_freecad_sketch_readers.py)
	$(PYTHON) $(call FixPath, $(TESTS)test_trigonometry.py)
	$(PYTHON) $(call FixPath, $(TESTS)test_freecad_sketcher_to_svg_translators.py)
	$(PYTHON) $(call FixPath, $(TESTS)test_freecad_svg_file.py)

#test: 
#	$(PYTHON) $(call FixPath, $(TESTS)/svg_d_attribute_parsing_test.py)
#	$(PYTHON) $(call FixPath, $(TESTS)/test_svg_generators.py)
#	$(PYTHON) $(call FixPath, $(TESTS)/test_svg_writers.py)
#	$(PYTHON) $(call FixPath, $(TESTS)/test_svg_validators.py)

$(TEST_LOGS): $(PYTHON_TEST_FILES) $(PYTHON_SRC_FILES) 
#	echo $?
#	echo $(PYTHON_TEST_FILES)

$(PYTHON_TEST_FILES): 
	echo $@
	$(PYTHON) $@


docs: $(DOCS_INDEX_HTML)

$(DOCS_INDEX_HTML): $(DOCS_SOURCE) $(PYTHON_SRC_FILES)
	make -f Makefile -C $(DOCS) html

setup: $(VENV_ACTIVATE)
	$(PIP) install -r requirements.txt

$(VENV_ACTIVATE): requirements.txt
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt


clean:
	make -f Makefile -C $(DOCS) clean
	$(RMDIR) $(PYCACHE)

pristine: clean
	$(RMDIR) $(VENV)