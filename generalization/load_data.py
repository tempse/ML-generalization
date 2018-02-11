import os
import sys

import pandas as pd


def load_data(data_params):
    """
    Simple wrapper for a pandas file import method.

    Takes a dictionary of data parameters.

    Required items in data_params dictionary:
    - data_path
    - data_read_func
    
    Optional items in data_params dictionary:
    - all valid keyword arguments for the selected Pandas import method

    Returns the data in form of a Pandas dataframe.
    """
    
    if not os.path.exists(data_params['data_path']):
        print('Error when loading the data. File {} is not a regular ' \
              'file.'.format(data_params['data_path']))
        sys.exit()

    read_func = data_params['data_read_func']

    # get all supported methods and all their available keywords
    valid_args = getattr(pd, read_func).__code__.co_varnames

    print('Reading file {}...'.format(data_params['data_path']))

    # dictionary holding all custom kwargs for the Pandas method
    kwargs_func = {}
    
    for key in data_params.keys():
        if key in valid_args:
            kwargs_func[key] = data_params[key]
    
    data = getattr(pd, read_func)(data_params['data_path'], **kwargs_func)

    print('Imported data: {} rows, {} columns'.format(data.shape[0],
                                                       data.shape[1]))
    print('Imported data: branches = {}'.format(list(data.columns.values)))

    return data
