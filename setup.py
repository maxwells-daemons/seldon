from distutils.core import setup
from distutils.extension import Extension

import numpy  # type: ignore
from Cython.Distutils import build_ext  # type: ignore

ext_modules = [
    Extension(
        "bitboard", ["src/ccode/bitboard.pyx"], include_dirs=[numpy.get_include()]
    ),
    Extension("solver", ["src/ccode/solver.pyx"], include_dirs=[numpy.get_include()]),
    Extension("mcts_utils", ["src/ccode/mcts_utils.pyx"], include_dirs=[]),
]

setup(name="Seldon", cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)
