************
Installation
************

The software has been developed using Ubuntu 16.04.

------------
Requirements
------------

Make sure to install these requirements first: ::

    sudo apt install python-pip python-dev build-essential
    sudo pip install --upgrade pip
    sudo pip install --upgrade virtualenv
    pip install pipenv

---------------------
Download the software
---------------------

Clone this repository by executing::

    git clone https://github.com/tempse/ML-generalization.git


--------------------    
Install dependencies
--------------------

Change into the downloaded repository folder (probably located in ``~/ML-generalization/``) and install all dependencies via ``pipenv``: ::

    cd ~/ML-generalization
    pipenv install

This automatically creates a virtual environment and installs all dependencies into it.

-----------------------------------------------
Run commands in the created virtual environment
-----------------------------------------------

There are two ways to execute commands from within the newly created virtual environment:

1) Activate the environment by::
    
       pipenv shell
   
  (For this, you have to be in the same folder as the project's ``Pipfile``.)

1) Invoke shell commands without explicitely activating the environment: ::
   
       pipenv run <command>
   
   (Example: ``pipenv run python generalization.py`` or ``pipenv run pytest -v``)

