from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the Cython extension
extensions = [
    Extension(
        name="spt.core",               # Name of the generated module
        sources=["spt/core.pyx"],      # Cython source file
        include_dirs=[np.get_include()] # Include NumPy headers
    )
]

# Define the setup
setup(
    name="spt",
    version="0.1",
    description="Single Particle Tracking Library",
    author="Francisco Palmero",
    author_email="franciscopalmeromoya@gmail.com",
    packages=["spt"],
    ext_modules=cythonize(extensions),
    install_requires=[
        'numpy',  # Example dependency
        # Add other dependencies
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
    ],
)
