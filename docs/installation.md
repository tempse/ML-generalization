# Installation

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