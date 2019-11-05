# See: https://github.com/cython/cython/blob/master/Demos/embed/Makefile
PYTHON := python
PYVERSION := $(shell $(PYTHON) -c "import sys; print(sys.version[:3])")
PYPREFIX := $(shell $(PYTHON) -c "import sys; print(sys.prefix)")

INCDIR := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_python_inc())")
PLATINCDIR := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_python_inc(plat_specific=True))")
LIBDIR1 := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
LIBDIR2 := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBPL'))")
PYLIB := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBRARY')[3:-2])")

CC := $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('CC'))")
LINKCC := $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('LINKCC'))")
LINKFORSHARED := $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('LINKFORSHARED'))")
LIBS := $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('LIBS'))")
SYSLIBS :=  $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('SYSLIBS'))")

CYTHON_OUTPUT_FILES := src/ccode/bitboard.c src/ccode/solver.c src/ccode/mcts_utils.c
PLAYER_CONFIG_DIR := src/player_configs
BUILD_DIR := build
PLAYER_C_DIR := $(BUILD_DIR)/_player_embeddings
PLAYER_OBJ_DIR := $(BUILD_DIR)/_player_objects
PLAYER_BINARY_DIR := $(BUILD_DIR)/players
PLAYER_CONFIGS := $(wildcard $(PLAYER_CONFIG_DIR)/*.py)
PLAYER_BINARY_FILES := $(patsubst $(PLAYER_CONFIG_DIR)/%.py, $(PLAYER_BINARY_DIR)/%, $(PLAYER_CONFIGS))

# Top-level targets
.PHONY: players
players: $(PLAYER_BINARY_FILES)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)/
	rm *.so
	rm $(CYTHON_OUTPUT_FILES)

# Player binaries and intermediate files
$(PLAYER_BINARY_DIR)/%: $(PLAYER_OBJ_DIR)/%.o
	@mkdir -p $(@D)
	$(LINKCC) -o $@ $^ -L$(LIBDIR1) -L$(LIBDIR2) -l$(PYLIB) $(LIBS) $(SYSLIBS) $(LINKFORSHARED)

$(PLAYER_OBJ_DIR)/%.o: $(PLAYER_C_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) -c $^ -I$(INCDIR) -I$(PLATINCDIR) -o $@

$(PLAYER_C_DIR)/%.c: $(PLAYER_CONFIG_DIR)/%.py $(CYTHON_OUTPUT_FILES)
	@mkdir -p $(@D)
	cython --embed -o $@ $<

# Cython files
$(CYTHON_OUTPUT_FILES): src/ccode/*.pyx src/ccode/*.pxd src/ccode/cbitboard.h src/ccode/cbitboard.c src/ccode/csolver.h src/ccode/csolver.c
	python3 setup.py build_ext --inplace
