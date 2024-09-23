from setuptools import setup

__version__ = '1.0.3'

required = [
    "tabulate",
    "scikit-learn",
    "click",
    "tqdm",
    "numpy<=1.22",
    "matplotlib",
    "pandas",
    "cuda-python",
    "tsfresh",
    "sqlalchemy"
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

