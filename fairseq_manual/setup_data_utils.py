# IMPORTANT: To build this, run "python fairseq_manual/setup_data_utils.py build_ext --inplace" from root folder of the repo
# After that you can "import data_utils_fast" or "from data_utils_fast import ..." from anywhere

from setuptools import setup
from Cython.Build import cythonize
import numpy

from setuptools import Extension


class NumpyExtension(Extension):
    """Source: https://stackoverflow.com/a/54128391"""

    def __init__(self, *args, **kwargs):
        self.__include_dirs = []
        super().__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        import numpy

        return self.__include_dirs + [numpy.get_include()]

    @include_dirs.setter
    def include_dirs(self, dirs):
        self.__include_dirs = dirs


extensions = [
    NumpyExtension(
        "data_utils_fast",
        sources=["fairseq_manual/data_utils_fast.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11", "-O3"],
    ),
]

setup(
    name="data utils fairseq_manual",
    ext_modules=cythonize(extensions),
    zip_safe=False,
    include_dirs=[numpy.get_include()],
)