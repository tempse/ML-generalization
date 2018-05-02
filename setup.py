import os
from setuptools import setup
from pipenv.project import Project
from pipenv.utils import convert_deps_to_pip

def read(fname):
    """Utility function to read the README file"""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def import_requires():
    """Utility function to read requirements from the Pipfile"""
    pfile = Project(chdir=False).parsed_pipfile
    return convert_deps_to_pip(pfile['packages'], r=False)

def import_requires_dev():
    """Utility function to read development requirements from the Pipfile"""
    pfile = Project(chdir=False).parsed_pipfile
    return convert_deps_to_pip(pfile['dev-packages'], r=False)


setup(
    name = "generalization",
    version = "0.1.2",
    author = "Mallinger, Ratzenboeck, Templ",
    author_email = "sebastian.templ@oeaw.ac.at",
    description = ("A machine learning framework for classifier " \
                   "generalization studies"),
    license = "MIT",
    keywords = "ML",
    url = "",
    tests_require =import_requires_dev(),
    install_requires=import_requires(),
    packages=['generalization'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
)
