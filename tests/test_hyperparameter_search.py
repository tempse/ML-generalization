import pytest
import os

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from skopt import BayesSearchCV

from generalization.hyperparameter_search import hyperparameter_search


def test_hyperparameter_search():
    model = SVC()
    model = hyperparameter_search(model, 2, 3)
    assert isinstance(model, BayesSearchCV)

    model = GaussianNB()
    model = hyperparameter_search(model, 2, 3)
    assert isinstance(model, GaussianNB)

    model = MultinomialNB()
    with pytest.raises(TypeError):
        model = hyperparameter_search(model, 2, 3)
