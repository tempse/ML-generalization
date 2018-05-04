import pytest
import os
import numpy as np
from numpy.testing import assert_almost_equal
import pandas as pd

from generalization.preprocessing import parse_target_labels, \
    PreprocessingManager


def test_parse_target_labels():
    pos_label = 'yes'
    neg_label = 'no'
    pos_label_invalid = 'yup'
    neg_label_invalid = 'nope'
    y = pd.DataFrame([pos_label, pos_label, neg_label])
    y_parsed = pd.DataFrame([1,1,0])
    y_invalid = pd.DataFrame(np.ones((2,2)))
    y_multiclass = pd.DataFrame([pos_label, neg_label, 'x'])

    with pytest.raises(ValueError):
        parse_target_labels(y_invalid, pos_label, neg_label)

    with pytest.raises(ValueError):
        parse_target_labels(y, pos_label_invalid, neg_label)

    with pytest.raises(ValueError):
        parse_target_labels(y, pos_label, neg_label_invalid)

    with pytest.raises(ValueError):
        parse_target_labels(y_multiclass, pos_label, neg_label)

    assert np.all(y_parsed == parse_target_labels(y, pos_label, neg_label))


def test_parse_target_labels_fails_due_to_invalid_data_format():
    with pytest.raises(ValueError):
        parse_target_labels('not a pandas data structure', 'yes', 'no')

def test_parse_target_labels_fails_due_to_equal_labels():
    y = pd.DataFrame(['yes', 'yes', 'no'])
    with pytest.raises(ValueError):
        parse_target_labels(y, 'yes', 'yes')


def test_PreprocessingManager(tmpdir):
    pm = PreprocessingManager(str(tmpdir))

    # TEST STANDARD SCALING
    X = pd.DataFrame([1,0,2])
    # X should have a mean of 1 and a std of 0.8165
    # the transformed should be roughly [ 0., -1.225,  1.225]
    target_X = pd.DataFrame([ 0., -1.224745, 1.224745])

    # if load_fitted_attributes is called before a file has been created
    # an IOError should be raised
    with pytest.raises(IOError):
        pm.standard_scale(X, True)

    X = pm.standard_scale(X, False)
    assert_almost_equal(X.as_matrix(), target_X.as_matrix())
    assert os.path.exists(str(tmpdir) + '/standard_scale_attributes.npy')

    X_new = pd.DataFrame([1,2,3])
    X_new = pm.standard_scale(X_new, True)
    target_X_new = pd.DataFrame([0.0, 1.2247449, 2.4494897])
    assert_almost_equal(X_new.as_matrix(), target_X_new.as_matrix())

    with pytest.raises(TypeError):
        pm.standard_scale('invalid type')

    # TEST OBJECT PARSING
    X = pd.DataFrame(['a', 's', 'd', np.nan])
    Y_nan = pd.DataFrame(['s', np.nan, 'a'])
    Y = pd.DataFrame(['s','d','a'])
    Y_new_val = pd.DataFrame(['d', 'a', 'x'])
    target_X = pd.DataFrame([0., 2., 1., np.nan])
    target_Y_nan = pd.DataFrame([2., np.nan, 0.])
    target_Y = pd.DataFrame([2, 1, 0])
    target_Y_new_val = pd.DataFrame([1,0,-2])

    with pytest.raises(IOError):
        pm.parse_object_columns(X, True)

    X = pm.parse_object_columns(X, False)
    assert_almost_equal(X.as_matrix(), target_X.as_matrix())
    assert os.path.exists(pm.get_output_path() + '/parse_object_columns_mapping.npy')
    Y = pm.parse_object_columns(Y, True)
    assert_almost_equal(Y.as_matrix(), target_Y.as_matrix())
    Y_nan = pm.parse_object_columns(Y_nan, True)
    assert_almost_equal(Y_nan.as_matrix(), target_Y_nan.as_matrix())
    Y_new_val = pm.parse_object_columns(Y_new_val, True)
    assert_almost_equal(Y_new_val.as_matrix(), target_Y_new_val.as_matrix())

    with pytest.raises(TypeError):
        pm.parse_object_columns('invalid type')

    # test case: nothing to parse
    X = pd.DataFrame([0])
    target_X = pd.DataFrame([0])
    assert_almost_equal(pm.parse_object_columns(X, False).as_matrix(), target_X.as_matrix())

    # TEST FILL NUMERICAL
    X = pd.DataFrame([1.5, np.nan, 2.5])
    target_X = pd.DataFrame([1.5, 2.0, 2.5])
    X = pm.fill_numerical(X)
    assert_almost_equal(X.as_matrix(), target_X.as_matrix())

    with pytest.raises(TypeError):
        pm.fill_numerical('invalid type')

    # TEST FILL CATEGORICAL
    d = {'col1': [1, np.nan, -2], 'col2': [3, 4, 2], 'col3': [1, 0, 3]}
    X = pd.DataFrame(d)
    X = pm.fill_categorical(X)
    n_nan_cols = pm.contains_nan(X)
    assert n_nan_cols == 0

    with pytest.raises(TypeError):
        pm.fill_categorical('invalid type')

    # TEST CONTAINS NAN
    X = pd.DataFrame([1, np.nan, 2], [1, 12, 2])
    n_nan_cols = pm.contains_nan(X)
    assert n_nan_cols == 1

    with pytest.raises(TypeError):
        pm.contains_nan('invalid type')

    # TEST REMOVE CORRELATED
    d       = {'col1': [1.1, 3.4, -2.3], 'col2': [3.1, 4.2, 2.9], 'col3': [3.1, 4.2, 3.0]}
    d_final = {'col1': [1.1, 3.4, -2.3], 'col2': [3.1, 4.2, 2.9]}
    X = pd.DataFrame(d)
    target_X = pd.DataFrame(d_final)

    d_y        = {'col1': [1.1, -2.3], 'col2': [4.2, 2.9], 'col3': [3.1, -3.0]}
    d_y_target = {'col1': [1.1, -2.3], 'col2': [4.2, 2.9]}
    Y = pd.DataFrame(d_y)
    target_Y = pd.DataFrame(d_y_target)

    with pytest.raises(IOError):
        pm.rm_correlated(X, True)

    X = pm.rm_correlated(X, False)
    assert_almost_equal(X.as_matrix(), target_X.as_matrix())
    assert os.path.exists(pm.get_output_path() + 'rm_correlated_dropped_cols.npy')

    Y = pm.rm_correlated(Y, True)
    assert_almost_equal(Y.as_matrix(), target_Y.as_matrix())

    with pytest.raises(TypeError):
        pm.rm_correlated('invalid type')

    # REMOVE LOW VARIANCE
    d_X = {'col1': [1.1, 1.14, 1.1], 'col2': [3.1, 4.2, 2.9], 'col3': [3.1, 4.2, 3.0]}
    d_X_target = {'col2': [3.1, 4.2, 2.9], 'col3': [3.1, 4.2, 3.0]}
    X = pd.DataFrame(d_X)
    target_X = pd.DataFrame(d_X_target)

    d_y        = {'col1': [1.1, -2.3], 'col2': [4.2, 2.9], 'col3': [3.1, -3.0]}
    d_y_target = {'col2': [4.2, 2.9], 'col3': [3.1, -3.0]}
    Y = pd.DataFrame(d_y)
    target_Y = pd.DataFrame(d_y_target)

    with pytest.raises(IOError):
        pm.rm_low_variance(X, True)

    X = pm.rm_low_variance(X, False)
    assert_almost_equal(X.as_matrix(), target_X.as_matrix())
    assert os.path.exists(pm.get_output_path() + 'rm_low_variance_drop_cols.npy')

    Y = pm.rm_low_variance(Y, True)
    assert_almost_equal(Y.as_matrix(), target_Y.as_matrix())

    with pytest.raises(TypeError):
        pm.rm_low_variance('invalid type')


def test_PreprocessingManager_fails_due_to_invalid_path_type():
    with pytest.raises(TypeError):
        pm = PreprocessingManager(False)
