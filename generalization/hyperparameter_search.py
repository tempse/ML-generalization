import numpy as np

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


def hyperparameter_search(estimator, n_iter, cv_nfolds):
    """
    Args:
            estimator: a scikit learn model which has a "score" function

            n_iter:  int, default=128 Number of parameter settings that are sampled. 
                     n_iter trades off runtime vs quality of the solution.

            cv_nfolds:  number of CV folds

    ____________________________________________________________________

    Returns: 
            model which can be treated as a standard scikit model
            with these functions:
                - fit 
                - score
                - predict
                - predict_proba
                - decision_function
                - transform 
                - inverse_transform
    ____________________________________________________________________

    Operation breakdown: 
            Wrapper function which also creates the search space defined by:
            search_spaces:  dict, list of dict or list of tuple containing (dict, int). 
                One of these cases: 
                    1. dictionary, where keys are parameter names (strings) and values are 
                       skopt.space.Dimension instances (Real, Integer or Categorical) or any other 
                       valid value that defines skopt dimension (see skopt.Optimizer docs). 
                       Represents search space over parameters of the provided estimator. 

                    2. list of dictionaries: a list of dictionaries, where every dictionary fits 
                       the description given in case 1 above. If a list of dictionary objects 
                       is given, then the search is performed sequentially for every parameter 
                       space with maximum number of evaluations set to self.n_iter.

                    3. list of (dict, int > 0): an extension 
                       of case 2 above, where first element of every tuple is a dictionary 
                       representing some search subspace, similarly as in case 2, and second 
                       element is a number of iterations that will be spent optimizing over 
                       this subspace.
    """
    # Initially we hard code this bit as we want as much of the 
    # hyper-parameter space to be searched
    # if no prior is given the variables are picked from a uniform  distribution
    class_name = estimator.__class__.__name__
    if 'SVC' in class_name:
        search_spaces = { 'C'               : Real(1e-6, 1e+6, prior='log-uniform') 
                        , 'gamma'           : Real(1e-6, 1e+1, prior='log-uniform')
                        , 'degree'          : Integer(1,8) 
                        , 'kernel'          : Categorical(['linear', 'poly', 'rbf'])
                        }
    
    elif 'RandomForestClassifier' in class_name:
        search_spaces = { 'n_estimators'    : Integer(10,800)
                        , 'criterion'       : Categorical(['gini', 'entropy'])
                        , 'max_features'    : Categorical(['auto'])
                        , 'max_depth'       : Integer(2,50)
                        , 'min_samples_leaf': Integer(1,200) 
                        }

    elif 'GradientBoostingClassifier' in class_name:
        search_spaces = { 'loss'            : Categorical(['deviance', 'exponential'])
                        , 'learning_rate'   : Real(0.01, 0.3)
                        , 'n_estimators'    : Integer(10,800)
                        , 'max_depth'       : Integer(2,50)
                        , 'criterion'       : Categorical(['friedman_mse', 'mse', 'mae'])
                        , 'max_features'    : Categorical(['auto'])
                        , 'min_samples_leaf': Integer(1,200) 
                        }

    elif 'DecisionTreeClassifier' in class_name:
        search_spaces = { 'criterion'       : Categorical(['gini', 'entropy'])
                        , 'splitter'        : Categorical(['best', 'random'])
                        , 'max_depth'       : Integer(2,50)
                        , 'min_samples_leaf': Integer(1,200) 
                        , 'max_features'    : Categorical(['auto'])
                        }

    elif 'KNeighborsClassifier' in class_name:
        search_spaces = { 'n_neighbors'     : Integer(2,30)
                        , 'weights'         : Categorical(['uniform', 'distance'])
                        , 'p'               : Integer(1,15)
                        , 'metric'          : Categorical(['minkowski', 
                                                           'chebyshev', 
                                                           'hamming',
                                                           'canberra',
                                                           'braycurtis'])
                        }

    elif 'MLPClassifier' in class_name:
        search_spaces = { 'activation'          : Categorical(['relu', 
                                                               'tanh', 
                                                               'logistic'])
                        , 'solver'              : Categorical(['lbfgs',
                                                               'sgd',
                                                               'adam'])
                        , 'batch_size'          : Integer(1,1000)
                        , 'power_t'             : Real(0.01, 2)
                        , 'max_iter'            : Integer(30, 300)
                        , 'alpha'               : Real(1e-5, 1e-1, prior='log-uniform')
                        , 'tol'                 : Real(1e-5, 1e-3, prior='log-uniform')
                        , 'learning_rate'       : Categorical(['constant',
                                                               'invscaling',
                                                               'adaptive'])
                        , 'learning_rate_init'  : Real(1e-4, 1e0 , prior='log-uniform')
                        , 'momentum'            : Real(0.5, 0.99)
                        }


    elif 'GaussianNB' in class_name:
        return estimator 

    else:
        raise TypeError('We do not support {} as a valuable estimator input. \
                         Please use one of the following scikit learn classifiers:\n \
                         SVC\nRandomForestClassifier\nGradientBoostingClassifier\n \
                         DecisionTreeClassifier\nGaussianNB\n \
                         KNeighborsClassifier'.format(class_name))
    
    model = BayesSearchCV(estimator          = estimator,
                          search_spaces      = search_spaces,
                          n_iter             = n_iter,
                          # kwargs passed on to the base class Opimizer
                          # here we just want a Gaussian Process as base_estimator
                          optimizer_kwargs   = {'base_estimator': 'GP'},
                          scoring            = 'f1',
                          n_jobs             = -1,
                          pre_dispatch       = '2*n_jobs',
                          # If iid is True than the data is assumed to be identically distributed 
                          # across the folds, and the loss minimized is the total loss per sample, 
                          # and not the mean loss across the folds.
                          iid                = True,
                          refit              = True,
                          cv                 = cv_nfolds,
                          verbose            = 1,
                          error_score        = 'raise',
                          return_train_score = True)

    return model

