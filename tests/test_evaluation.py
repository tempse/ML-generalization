import pytest
from pytest import approx
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd

from generalization.evaluation import evaluate_nfold, \
    performance_difference


def test_evaluate_nfold_with_numpy_arrays(classification_data):
    X, y = classification_data
    model = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.5,
                                                        random_state=1)
    model.fit(X_train, y_train)
    num_folds = 1

    scores = evaluate_nfold(X_test, y_test, model,
                            num_folds)
    assert len(scores) == num_folds
    assert all((scores[i]>=0) & (scores[i]<=1) \
               for i in range(len(scores))) == True

    num_folds = 3
    scores = evaluate_nfold(X_test, y_test, model,
                            num_folds)
    assert len(scores) == num_folds
    assert all((scores[i]>=0) & (scores[i]<=1) \
               for i in range(len(scores))) == True


def test_evaluate_nfold_with_pandas_dataframes(classification_data):
    X, y = classification_data
    model = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.5,
                                                        random_state=1)
    model.fit(X_train, y_train)
    num_folds = 3

    scores = evaluate_nfold(pd.DataFrame(X_test), pd.DataFrame(y_test),
                            model, num_folds)
    assert len(scores) == num_folds
    assert all((scores[i]>=0) & (scores[i]<=1) \
               for i in range(len(scores))) == True


def test_evaluate_nfold_randomize(classification_data):
    X, y = classification_data
    model = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.5,
                                                        random_state=1)
    model.fit(X_train, y_train)
    num_folds = 2

    scores = evaluate_nfold(X_test, y_test, model,
                            num_folds, randomize=True)
    assert len(scores) == num_folds
    assert all((scores[i]>=0) & (scores[i]<=1) \
               for i in range(len(scores))) == True
    

def test_evaluate_nfold_bootstrapping(classification_data):
    X, y = classification_data
    model = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.5,
                                                        random_state=1)
    model.fit(X_train, y_train)
    num_folds = 3

    scores = evaluate_nfold(pd.DataFrame(X_test), pd.DataFrame(y_test),
                            model, num_folds, bootstrapping=True)
    assert len(scores) == num_folds
    assert all((scores[i]>=0) & (scores[i]<=1) \
               for i in range(len(scores))) == True


def test_evaluate_nfold_fail_due_to_zero_folds(classification_data):
    X, y = classification_data
    model = GaussianNB()
    num_folds = 0
    with pytest.raises(ValueError):
        evaluate_nfold(X, y, model, num_folds)


def test_evaluate_nfold_fail_due_to_invalid_data():
    X = 'not a numpy array...'
    y = '...nor a pandas data structure'
    model = GaussianNB()
    num_folds = 3
    with pytest.raises(ValueError):
        evaluate_nfold(X, y, model, num_folds)


def test_evaluate_nfold_fail_due_to_invalid_num_folds(classification_data):
    X, y = classification_data
    num_folds = 'not an integer'
    with pytest.raises(ValueError):
        evaluate_nfold(X, y, GaussianNB(), num_folds)

def test_evaluate_nfold_fail_due_to_invalid_scoring_func(classification_data):
    X, y = classification_data
    num_folds = 3
    scoring_func = 'invalid function'
    with pytest.raises(ValueError):
        evaluate_nfold(X, y, GaussianNB(), num_folds, scoring=scoring_func)

def test_evaluate_nfold_fail_due_to_invalid_bootstripping_param(classification_data):
    X, y = classification_data
    num_folds = 3
    bootstrapping = 'non-Boolean'
    with pytest.raises(ValueError):
        evaluate_nfold(X, y, GaussianNB(), num_folds,
                       bootstrapping=bootstrapping)


def test_performance_difference_of_three():
    a = [1,2,3]
    mean, std = performance_difference(a)
    assert mean == approx(4/3.0)
    assert std == approx(0.47140452)


def test_performance_difference_of_two():
    a = [1,3]
    mean, std = performance_difference(a)
    assert mean == 2.0
    assert std == 0.0


def test_performance_difference_of_one():
    a = [5]
    mean, std = performance_difference(a)
    assert mean == 0.0
    assert std == 0.0


def test_performance_difference_invalid_input():
    with pytest.raises(ValueError):
        performance_difference('non-sequence object')
