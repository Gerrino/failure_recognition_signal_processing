from setuptools import setup, find_namespace_packages
import sys
__version__ = '1.0.1'

required = [
    "matplotlib",    
]

setup(
    name="failure_recognition_signal_processing",
    version=__version__,
    packages=['failure_recognition_signal_processing'],
    install_requires=required,
    extras_require={
        "dev": ["pylint", "black", "sphinx"],
    },
    #py_modules=['failure_recognition_signal_proccessing'],
    include_package_data=True
)

