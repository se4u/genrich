"""Compile tag_rhmm model through cytohn
"""
__date__    = "12 December 2014"
__author__  = "Pushpendre Rastogi"
__contact__ = "pushpendre@jhu.edu"

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

setup(name = "tag_rhmm",
      ext_modules = cythonize([
            Extension(
                "tag_rhmm",
                ['tag_rhmm.pyx'],
                extra_compile_args=["-O3"]
                )
            ]),
      )
