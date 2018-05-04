from setuptools import setup, find_packages
import os

def read(fname):
    """Utility function to read the README"""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='generalization',
    version='0.1.4',
    author='Sebastian Templ, Sebastian Ratzenböck, Florian Mallinger',
    author_email = "sebastian.templ@oeaw.ac.at",
    description = ("A machine learning framework for classifier " \
                   "generalization studies"),
    long_description=read('README.md'),
    license = "MIT",
    keywords = "ML",
    package_dir={'': 'src'},
    packages=find_packages('src'),
    install_requires=[
        'pipenv',
    ],
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
