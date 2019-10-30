from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Distutils import build_ext

ext_modules = [
    Extension("backend", ["src/ccode/backend.pyx"], include_dirs=[numpy.get_include()])
]

setup(name="Seldon", cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)
