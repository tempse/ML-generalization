# A Machine Learning Framework for Classifier Generalization Studies
![](https://img.shields.io/badge/version-0.1.2-blue.svg) ![](https://img.shields.io/badge/python-3.5-blue.svg)

[![Build Status](https://travis-ci.org/tempse/ML-generalization.svg?branch=master)](https://travis-ci.org/tempse/ML-generalization) [![Documentation Status](https://readthedocs.org/projects/ml-generalization/badge/?version=latest)](http://ml-generalization.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/tempse/ML-generalization/branch/master/graph/badge.svg)](https://codecov.io/gh/tempse/ML-generalization)

[![codebeat badge](https://codebeat.co/badges/77d8ab35-0dce-48ac-a6a3-778297b0d823)](https://codebeat.co/projects/github-com-tempse-ml-generalization-master) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/4a9b32fd14934a95b27856582fa23991)](https://www.codacy.com/app/tempse/ML-generalization?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=tempse/ML-generalization&amp;utm_campaign=Badge_Grade) [![CodeFactor](https://www.codefactor.io/repository/github/tempse/ml-generalization/badge)](https://www.codefactor.io/repository/github/tempse/ml-generalization) [![Maintainability](https://api.codeclimate.com/v1/badges/ab83d3a90f0fd19ec405/maintainability)](https://codeclimate.com/github/tempse/ML-generalization/maintainability) [![BCH compliance](https://bettercodehub.com/edge/badge/tempse/ML-generalization?branch=master)](https://bettercodehub.com/)

[![Updates](https://pyup.io/repos/github/tempse/ML-generalization/shield.svg)](https://pyup.io/repos/github/tempse/ML-generalization/) [![Python 3](https://pyup.io/repos/github/tempse/ML-generalization/python-3-shield.svg)](https://pyup.io/repos/github/tempse/ML-generalization/) [![Requirements Status](https://requires.io/github/tempse/ML-generalization/requirements.svg?branch=master)](https://requires.io/github/tempse/ML-generalization/requirements/?branch=master)

> This framework has been developed in the course of the Machine Learning lecture 184.702 at TU Wien.

Documentation: [ml-generalization.readthedocs.io](http://ml-generalization.readthedocs.io/en/latest/)
        
## Installation

The software has been developed using Ubuntu 16.04. With the provided setup script, it should however be deployable to other platforms as well.

1) Install a virtual environment with Python 3 and activate it.
    > On Ubuntu-based systems, this can be done via
    >
    >`virtualenv -p python3 <myenv>`
    >     
    >(where `<myenv>` is the name of the environment) and
    >
    >`source <myenv>/bin/activate`
1) Install the framework and all its dependencies (numpy has to be installed first explicitly):
    
    ```
    pip install numpy
    python setup.py install
    ```

    Alternatively, one can run
    ```
    pip install numpy
    pip install -r requirements.txt
    ```

## Running the tests

To run the automated tests, execute `python -m unittest discover`.

## Run and control the software

Take a look at [the documentation](http://ml-generalization.readthedocs.io/en/latest/) for detailed information.

## Uninstall the software

To uninstall everything, deactivate the virtual environment (with `deactivate`) and just delete the following folders:
- `<myenv>` (the folder specified above during virtual environment setup)
- `build` (in the same directory as `setup.py`)
- `dist` (in the same directory as `setup.py`)
- `generalization.egg-info` (in the same directory as `setup.py`)
