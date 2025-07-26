from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "spbae",
        ["spbae_search.pyx"],
        # Add these defines to disable intrinsics that cause problems
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            # ("__MINGW32__", "1"),
        ],
    )
]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,  # Turn off bounds-checking
            'wraparound': False,   # Turn off negative indexing
            'cdivision': True,     # Turn off division-by-zero checking
        }
    ),
    include_dirs=[numpy.get_include()]
)
