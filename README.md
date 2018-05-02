# A Machine Learning Framework for Classifier Generalization Studies
![](https://img.shields.io/badge/version-0.1.3-blue.svg) ![](https://img.shields.io/badge/python-3.5-blue.svg)

[![Build Status](https://travis-ci.org/tempse/ML-generalization.svg?branch=master)](https://travis-ci.org/tempse/ML-generalization) [![Documentation Status](https://readthedocs.org/projects/ml-generalization/badge/?version=latest)](http://ml-generalization.readthedocs.io/en/latest/?badge=latest) [![codecov](https://codecov.io/gh/tempse/ML-generalization/branch/master/graph/badge.svg)](https://codecov.io/gh/tempse/ML-generalization)

[![codebeat badge](https://codebeat.co/badges/77d8ab35-0dce-48ac-a6a3-778297b0d823)](https://codebeat.co/projects/github-com-tempse-ml-generalization-master) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/4a9b32fd14934a95b27856582fa23991)](https://www.codacy.com/app/tempse/ML-generalization?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=tempse/ML-generalization&amp;utm_campaign=Badge_Grade) [![CodeFactor](https://www.codefactor.io/repository/github/tempse/ml-generalization/badge)](https://www.codefactor.io/repository/github/tempse/ml-generalization) [![Maintainability](https://api.codeclimate.com/v1/badges/ab83d3a90f0fd19ec405/maintainability)](https://codeclimate.com/github/tempse/ML-generalization/maintainability) [![BCH compliance](https://bettercodehub.com/edge/badge/tempse/ML-generalization?branch=master)](https://bettercodehub.com/)

[![Updates](https://pyup.io/repos/github/tempse/ML-generalization/shield.svg)](https://pyup.io/repos/github/tempse/ML-generalization/) [![Python 3](https://pyup.io/repos/github/tempse/ML-generalization/python-3-shield.svg)](https://pyup.io/repos/github/tempse/ML-generalization/) [![Requirements Status](https://requires.io/github/tempse/ML-generalization/requirements.svg?branch=master)](https://requires.io/github/tempse/ML-generalization/requirements/?branch=master)

> This framework has been developed in the course of the Machine Learning lecture 184.702 at TU Wien.

Documentation: [ml-generalization.readthedocs.io](http://ml-generalization.readthedocs.io/en/latest/)
        
## Installation

> The software has been developed using Ubuntu 16.04.

### Requirements

Make sure to install these requirements first:
```
sudo apt install python-pip python-dev build-essential
sudo pip install --upgrade pip
sudo pip install --upgrade virtualenv
pip install pipenv
```

### Download the software:

Clone this repository by executing
```
git clone https://github.com/tempse/ML-generalization.git
```

    
### Install dependencies:

Change into the downloaded repository folder (probably located in `~/ML-generalization/`) and install all dependencies via `pipenv`:
```
cd ~/ML-generalization
pipenv install
```
This automatically creates a virtual environment and installs all dependencies into it.

### Run commands in the created virtual environment

There are two ways to execute commands from within the newly created virtual environment:
1) Activate the environment by
    ```
    pipenv shell
    ```
    (For this, you have to be in the same folder as the project's `Pipfile`.)

1) Invoke shell commands without explicitely activating the environment:
    ```
    pipenv run <command>
    ```
    (Example: `pipenv run python generalization.py` or `pipenv run pytest -v`)


## Run and control the software

Take a look at [the documentation](http://ml-generalization.readthedocs.io/en/latest/) for detailed information.


## Running the tests

In order to run the tests, install the development requirements first (this has to be done just once):
```
pipenv install --dev
```

Then, simply run `pytest` in the previously installed virtual environment. For example:
```
pipenv run pytest -v --cov=generalization
```


## Uninstall the software

To uninstall all installed dependencies, simply run
```
pipenv uninstall --all
```

In order to also remove the virtual environment that has been created by `pipenv`, remove the corresponding folder in `/home/<user>/.local/share/virtualenvs/`.
