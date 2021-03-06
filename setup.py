from setuptools import setup, find_namespace_packages
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

requirements = [
    "numpy",
    "pandas",
    "matplotlib",
    "sympy",
    "h5py",
    "click",
    "numba",
    "rich",
    "sklearn",
    "slackclient",
    "dropbox",
    "pyinspect",
    "loguru",
    "pyrnn",
    "vedo",
    "fcutils",
    "celluloid",
]

setup(
    name="locoproj",
    version="0.0.1",
    description="Physical locomotion model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=requirements,
    python_requires=">=3.6, <3.8",
    packages=find_namespace_packages(
        exclude=("control", "models", "playground", "Screenshots")
    ),
    include_package_data=True,
    url="https://github.com/FedeClaudi/pysical_locomotion",
    author="Federico Claudi",
    zip_safe=False,
    entry_points={"console_scripts": []},
)
