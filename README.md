# A Machine Learning Framework for Classifier Generalization Studies
![](https://img.shields.io/badge/python-3.5-blue.svg) ![](https://img.shields.io/badge/version-0.1.0-yellow.svg)

    
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

## Running the tests

To run the automated tests, execute `python -m unittest discover generalization/tests/`.

## Run and control the software

Look at [these instructions](https://github.com/tempse/ML_184.702/wiki/ML-software-(third-assignment)), for the time being.

## Uninstall the software

To uninstall everything, deactivate the virtual environment (with `deactivate`) and just delete the following folders:
- `<myenv>` (the folder specified above during virtual environment setup)
- `build` (in the same directory as `setup.py`)
- `dist` (in the same directory as `setup.py`)
- `generalization.egg-info` (in the same directory as `setup.py`)