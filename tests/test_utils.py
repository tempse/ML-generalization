import pytest
import warnings
import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from skopt import BayesSearchCV


from generalization.utils import ignore_warnings, get_optimal_CV_n_folds, \
    print_class_counts, get_search_results


@pytest.mark.todo
def test_ignore_warnings():
    #def func_raise_warning():
    #    warnings.warn('forced warning')
    #    pass
    #
    #with pytest.warns(UserWarning, match='forced warning'):
    #    func_raise_warning()
    #
    #@ignore_warnings
    #def func_suppressed_warning():
    #    warnings.warn('suppressed warning')
    #    pass
    #
    #with pytest.warns(UserWarning):
    #    func_suppressed_warning()
    pass
        

@pytest.mark.slowtest
def test_get_optimal_CV_n_folds(classification_data):
    X, y = classification_data
    nfolds = get_optimal_CV_n_folds(X, y)
    assert nfolds == 5

def test_get_optimal_CV_n_folds_fail_due_to_invalid_data(classification_data):
    X, y = classification_data
    with pytest.raises(ValueError):
        nfolds = get_optimal_CV_n_folds('not a numpy array', y)

    with pytest.raises(ValueError):
        nfolds = get_optimal_CV_n_folds(X, 'not a numpy array')


def test_print_class_counts(capsys):
    a = np.array([1,1,0])
    print_class_counts(a, 'training', background=0, signal=1)
    out, err = capsys.readouterr()
    assert 'Number of training samples: 3' in out
    assert 'Number of signal: 2 (66.67 percent)' in out
    assert 'Number of background: 1 (33.33 percent)' in out
    assert err == ''


def test_get_search_results_from_sklearn_model(classification_data):
    X, y = classification_data
    model = GaussianNB().fit(X,y)
    model_params = get_search_results(model)
    assert len(model_params) == 1
    assert model_params['priors'] == None


@pytest.mark.todo
def test_get_search_results_from_skopt_model(classification_data):
    pass

def test_get_search_results_fail_due_to_invalid_model():
    with pytest.raises(TypeError):
        model_params = get_search_results('not a valid model type')
