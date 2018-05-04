import pytest
import os

from sklearn.datasets import make_classification


@pytest.fixture(scope='module', autouse=True)
def set_matplotlib_backend():
    if os.environ.get('DISPLAY') == '':
        print('No display name found. Using matplotlib \'Agg\' backend.')
        import matplotlib
        matplotlib.use('Agg')

@pytest.fixture()
def classification_data():
    return make_classification(n_features=5, random_state=1)

@pytest.fixture()
def data_config_required_keys():
    required_keys = ['data_identifier',
                     'data_path',
                     'data_read_func',
                     'data_target_col',
                     'data_target_positive_label',
                     'data_target_negative_label',
                     'data_preprocessing']
    return required_keys

@pytest.fixture()
def config_file_contents_simple():
    contents = '[section]\n' \
               'key_int: 42\n' \
               'key_float: 4.2\n' \
               'key_string: \'fortytwo\'\n' \
               'key_boolean: True\n'
    return contents
