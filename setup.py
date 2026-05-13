import numpy as np
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "arlmet._pack",
            sources=["src/arlmet/_pack.c"],
            include_dirs=[np.get_include()],
        )
    ]
)
