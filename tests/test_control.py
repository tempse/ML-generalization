import pytest

from generalization.control import check_data_config_requirements


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
