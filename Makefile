CYTHON_FILES = src/ccode/bitboard.c src/ccode/solver.c src/ccode/mcts_utils.c

cython:
	python3 setup.py build_ext --inplace

clean:
	rm -rf build/
	rm *.so
	rm $(CYTHON_FILES)
