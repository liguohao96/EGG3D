import os
import os
import sys
from glob import glob

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

PKG_NAME = 'ext'
EXT_LIST = ['mesh']

ext_module_list = []
for ext in EXT_LIST:
    ext_module_list.append(
        Pybind11Extension(
            name=f'{ext}',
            include_dirs=[ext],
            sources=sorted(glob(os.path.join(ext, "*.c*"))),
            define_macros=[("EXTENSION_NAME", ext)],
            extra_compile_args=[
                "-std=c++11"
                # "-g"
                ]
        )
    )

setup(
    name=PKG_NAME,
    ext_modules=ext_module_list,
    cmdclass={
       "'build_ext": build_ext
    }
)