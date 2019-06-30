
from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
        name='ccp-criterion',
        version=0.1,
        author='Harish Ganesan',
        author_email='harishganesan96@gmail.com',
        ext_modules=cythonize('ccp.pyx'),
        include_dirs=[numpy.get_include()]
        )
