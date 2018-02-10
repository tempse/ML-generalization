from itertools import combinations
import collections
import types
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, zero_one_loss
from sklearn.utils import shuffle, resample


def evaluate_nfold(X, y, model, num_test_samples, scoring='f1_score',
                   bootstrapping=False, randomize=False):
    """Evaluates a trained model on multiple test samples.

    Args:
        X: feature matrix of the entire test data
        y: target vector of the entire test data
        model: trained scikit-learn model
        num_test_samples: number of test samples
        scoring: (scikit-learn) scoring function (default: 'accuracy_score')
        bootstrapping: choose whether to create bootstrap samples from the test
            set (i.e., samples randomly drawn with replacement). If False,
            separate, distinct samples are randomly drawn (without replacement).
            (Default: False)
        randomize: whether to shuffle the data before splitting (default: False)

    Returns:
        List of evaluation scores for the test sample folds

    """

    if not isinstance(X, pd.DataFrame) and not isinstance(X, np.ndarray):
        raise ValueError('feature matrix X is neither a numpy array nor a pandas dataframe')

    if not isinstance(y, pd.DataFrame) and not isinstance(y, np.ndarray) and \
        not isinstance(y, pd.core.series.Series):
        raise ValueError('target vector y is neither a numpy array nor a pandas dataframe')

    if not isinstance(num_test_samples, int):
        raise ValueError('invalid, non-integer number of test samples: {}'.format(num_test_samples))

    supported_scoring_funcs = {
        'accuracy_score': accuracy_score,
        'precision_score': precision_score,
        'recall_score': recall_score,
        'f1_score': f1_score,
        'roc_auc_score': roc_auc_score,
        'zero_one_loss': zero_one_loss
    }

    if not isinstance(scoring, str) or \
       scoring not in supported_scoring_funcs.keys():
        raise ValueError('invalid scoring function: {} (should be one of {})'.format(
            scoring, [i for i in supported_scoring_funcs.keys()]))

    if not isinstance(bootstrapping, bool):
        raise ValueError('bootstrapping attribute is not boolean')

    if num_test_samples < 1:
        raise ValueError('num_test_sample must be larger than 1, ' \
                         'but is {}'.format(num_test_samples))
    
    if isinstance(X, pd.DataFrame):
        #warnings.warn('Feature matrix X converted from pandas dataframe to numpy array')
        X = X.as_matrix()

    if isinstance(y, pd.DataFrame) or isinstance(y, pd.core.series.Series):
        #warnings.warn('Target vector y converted from pandas dataframe to numpy array')
        y = y.as_matrix()
    
    scores = []

    if randomize:
        X, y = shuffle(X, y)

    if num_test_samples == 1:
        y_pred = model.predict(X)
        scores.append((supported_scoring_funcs[scoring])(y, y_pred))
        
    elif bootstrapping:
        for i in range(num_test_samples):
            X_sample, y_sample = resample(X, y, replace=True)
            y_pred = model.predict(X_sample)
            scores.append((supported_scoring_funcs[scoring])(y_sample, y_pred))
        
    else:
        k_fold = KFold(num_test_samples)
        for k, (fold_indices_train, fold_indices_test) in enumerate(k_fold.split(X, y)):
            y_pred = model.predict(X[fold_indices_test])
            scores.append((supported_scoring_funcs[scoring])(y[fold_indices_test], y_pred))

    return scores


def performance_difference(scores):
    """Considers pair-wise score differences and calculates mean performance.

    Args:
        scores: List of evaluated scores

    Returns:
        Mean and standard deviation of pair-wise performance differences.

    """

    if not isinstance(scores, collections.Sequence) \
        or isinstance(scores, str):
        raise ValueError('passed argument is not a list object')

    comb_indices = list(combinations([x for x in range(len(scores))], 2))

    if len(scores) <= 1:
        return 0.0, 0.0
    
    scores_diff = []

    for i,j in enumerate(comb_indices):
        scores_diff.append(abs(scores[comb_indices[i][0]] - 
                               scores[comb_indices[i][1]]))

    return np.mean(scores_diff), np.std(scores_diff)


