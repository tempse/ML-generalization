import pytest
import os

try:
    import configparser
except:
    import ConfigParser as configparser

from generalization.control import fileparser, config_file_to_dict, \
    check_data_config_requirements


def test_fileparser(config_file_contents_simple, tmpdir):
    config_file = tmpdir.join('config_file_simple.conf')
    contents = config_file_contents_simple
    config_file.write(contents)

    parser = fileparser()
    parser.read(str(config_file))
    contents_dict = parser.as_dict()

    assert len(contents_dict) == 1
    assert len(contents_dict['section']) == 4

    assert isinstance(contents_dict['section']['key_int'], int)
    assert isinstance(contents_dict['section']['key_float'], float)
    assert isinstance(contents_dict['section']['key_string'], str)
    assert isinstance(contents_dict['section']['key_boolean'], bool)

    assert contents_dict['section']['key_int'] == 42
    assert contents_dict['section']['key_float'] == 4.2
    assert contents_dict['section']['key_string'] == 'fortytwo'
    assert contents_dict['section']['key_boolean'] == True


def test_config_file_to_dict(config_file_contents_simple, tmpdir):
    config_file = tmpdir.join('config_file_simple.conf')
    contents = config_file_contents_simple
    config_file.write(contents)

    contents_dict = config_file_to_dict(str(config_file))

    assert len(contents_dict) == 1
    assert len(contents_dict['section']) == 4

    assert isinstance(contents_dict['section']['key_int'], int)
    assert isinstance(contents_dict['section']['key_float'], float)
    assert isinstance(contents_dict['section']['key_string'], str)
    assert isinstance(contents_dict['section']['key_boolean'], bool)

    assert contents_dict['section']['key_int'] == 42
    assert contents_dict['section']['key_float'] == 4.2
    assert contents_dict['section']['key_string'] == 'fortytwo'
    assert contents_dict['section']['key_boolean'] == True


def test_config_file_to_dict_fails_due_to_invalid_path_type():
    with pytest.raises(ValueError):
        contents_dict = config_file_to_dict(False)


def test_config_file_to_dict_fails_due_to_invalid_path():
    with pytest.raises(IOError):
        contents_dict = config_file_to_dict('non-existing path')



def test_check_data_config_requirements(data_config_required_keys):
    required_keys = data_config_required_keys
    dict_all_keys = {k:0 for k in required_keys}
    assert check_data_config_requirements(dict_all_keys) == True

def test_check_data_config_requirements_fails_due_to_missing_keys(data_config_required_keys):
    required_keys = data_config_required_keys
    dict_one_key = {required_keys[0]: 'value1'}

    with pytest.raises(AttributeError):
        check_data_config_requirements(dict_one_key)

def test_check_data_config_requirements_fails_due_to_invalid_input():
    invalid_input = 'not a dictionary'
    with pytest.raises(ValueError):
        check_data_config_requirements(invalid_input)
