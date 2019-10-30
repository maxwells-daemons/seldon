CYTHON_FILES = src/ccode/backend.c

cython:
	python3 setup.py build_ext --inplace

clean:
	rm -rf build/
	rm *.so
	rm $(CYTHON_FILES)
