import pytest
import pandas as pd

from generalization.load_data import load_data


def test_load_data(tmpdir, csv_file_contents_simple):
    contents = csv_file_contents_simple
    data_file = tmpdir.join('data_file.csv')
    data_file.write(contents)
    assert data_file.read() == contents
    
    data_params = {
        'data_path': str(tmpdir+'/data_file.csv'),
        'data_read_func': 'read_csv',
        'sep': ' '
    }

    data = load_data(data_params)

    assert isinstance(data, pd.DataFrame)
    assert data.shape == (2,3)
    assert data.ix[0,0] == 1.0
    assert data.ix[0,1] == 1.1
    assert data.ix[1,2] == 2.2
    assert data['featA'][1] == 2.0
    assert data['featC'].shape == (2,)

def test_load_data_fails_due_to_invalid_path():
    data_params = {
        'data_path': 'invalid path',
        'data_read_func': 'read_csv'
    }
    
    with pytest.raises(OSError):
        data = load_data(data_params)

def test_load_data_fails_due_to_invalid_read_function(tmpdir, csv_file_contents_simple):
    contents = csv_file_contents_simple
    data_file = tmpdir.join('data_file.csv')
    data_file.write(contents)
    assert data_file.read() == contents
    
    data_params = {
        'data_path': str(tmpdir+'/data_file.csv'),
        'data_read_func': 'invalid_read_func',
        'sep': ' '
    }

    with pytest.raises(AttributeError):
        data = load_data(data_params)
