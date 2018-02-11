import sys
import os
import numpy as np
import errno

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate

import skopt

def get_optimal_CV_n_folds(X_train, y_train, average_over_n_fits=1):
    """
    Input:
            X_train: a numpy array containing the training data
            y_train: a numpy array containing the traget values
    ____________________________________________________________________

    Returns:
            the optimal number of CV folds.
    ____________________________________________________________________

    Operation breakdown:
            Fits the data multiple times with a Naive-Bayes model and
            returns the number of folds where the standard deviation
            is minimal regarding a certain metric - here: accuracy
    """

    if not isinstance(X_train, np.ndarray):
        raise ValueError('Inappropriate type: {} for the feature matrix X \
                          whereas a numpy array is expected'.format(type(X_train)))
    if not isinstance(y_train, np.ndarray):
        raise ValueError('Inappropriate type: {} for the target vector y \
                          whereas a numpy array is expected'.format(type(y_train)))
    metric = 'accuracy'
    model = GaussianNB()

    avg_min_var_nfolds = []
    for n_fits in range(average_over_n_fits):
        # calculate the best # of folds as an average
        for nfolds in range(3,10):
            # fit here the data with model
            scores = cross_validate(model, X_train, y_train,
                                    cv                 = nfolds,
                                    scoring            = metric,
                                    n_jobs             = -1,
                                    return_train_score = True)
            # check if variable exists, if not, create it
            if nfolds is 3:
                min_var = np.var(scores['test_score'])
                min_var_nfolds = nfolds
            elif min_var > np.var(scores['test_score']):
                min_var = np.var(scores['test_score'])
                min_var_nfolds = nfolds
            else:
                pass
        avg_min_var_nfolds.append(min_var_nfolds)

    return int(round(sum(avg_min_var_nfolds) / float(len(avg_min_var_nfolds))))


class logger(object):
    """
    Writes output both to terminal and to file.
    """

    def __init__(self, output_path):
        self.terminal = sys.stdout

        if not os.path.exists(output_path):
            print('Info: path {} does not exist, thus will be created.'.format(
                output_path))
            try:
                os.makedirs(output_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        self.log = open(output_path + 'stdout.log', 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def print_class_counts(y, name, **kwargs):
    """
    Prints the number of entries of a passed vector and the size of its contained classes.
    Usage example: "print_class_counts(y_train, 'training', background=0, signal=1)"
    """

    print('\nNumber of {} samples: {}'.format(name, y.shape[0]))

    class_list = []

    try:
        class_list = kwargs.iteritems()
    except AttributeError:
        # iteritems() does not work in Python 3
        pass

    try:
        class_list = list(kwargs.items())
    except Exception:
        print('Error in function print_class_counts: ' \
              'Could not iterate over the passed items. ' \
              '(Maybe a Python version conflict?)')
        print('\n Skip printing class counts.')

        return


    for key, item in class_list:
        print('  Number of {}: {} ({:.2f} percent)'.format(
            key,
            y[y==item].shape[0],
            y[y==item].shape[0]*100/y.shape[0]))


def get_search_results(model):
    """
    Args:
        model: the BayesSearchCV fitted model
   ___________________________________________________________________

    Returns:
	    a dictionary containing the model parameters
    ___________________________________________________________________

    Operation breakdown:
        Prints the BayesSearch results:
            - the search summary
            - the best estimator
            - the score of the best estimator
    """

    if isinstance(model, skopt.searchcv.BayesSearchCV):
        print('\n')
        print('{}\nBayes search summary (used folds: {})\n{}\n'.format(
            '-'*35, model.n_splits_, '-'*35))
        print('  Best estimator: {} '.format(model.best_estimator_))
        print('\n  Best score: {} (optimized metric: f1-score)'.format(
            model.best_score_))
        print('\n')

        for item in sorted(model.cv_results_.keys()):
            keywords = ['mean_', 'std_']

            if not any(filter(lambda x: item.startswith(x), keywords)):
                continue
            print('  {:35s}: {:8.4}'.format(item, model.cv_results_[item][model.best_index_]))
        model_pars =  model.best_estimator_.get_params()
    else:
        try:
            model_pars = model.get_params()
            print('\n  Model parameters: {}'.format(model_pars))
        except AttributeError:
            raise TypeError('Input variable model is not of type \
                    skopt.searchcv.BayesSearchCV but instead is {}'.format(type(model)))

    return model_pars


def pandas2arff(df,filename,wekaname = "pandasdata",cleanstringdata=False,cleannan=True):
    """
    converts the pandas dataframe to a weka compatible file
    df: dataframe in pandas format
    filename: the filename you want the weka compatible file to be in
    wekaname: the name you want to give to the weka dataset (this will be visible to you when you open it in Weka)
    cleanstringdata: clean up data which may have spaces and replace with "_", special characters etc which seem to annoy Weka.
                     To suppress this, set this to False
    cleannan: replaces all nan values with "?" which is Weka's standard for missing values.
              To suppress this, set this to False
    """

    import re

    def cleanstring(s):
        if s!="?":
            return re.sub('[^A-Za-z0-9]+', "_", str(s))
        else:
            return "?"

    dfcopy = df #all cleaning operations get done on this copy


    if cleannan:
        dfcopy = dfcopy.fillna(-999999999) #this is so that we can swap this out for "?"
        #this makes sure that certain numerical columns with missing values don't get stuck with "object" type

    f = open(filename,"w")
    arffList = []
    arffList.append("@RELATION " + wekaname + "\n")
    #look at each column's dtype. If it's an "object", make it "nominal" under Weka for now (can be changed in source for dates.. etc)
    for i in range(df.shape[1]):
        if dfcopy.dtypes[i]=='O' or (df.columns[i] in ["Class","CLASS","class"]):
            if cleannan:
                dfcopy.iloc[:,i] = dfcopy.iloc[:,i].replace(to_replace=-999999999, value="?")
            if cleanstringdata:
                dfcopy.iloc[:,i] = dfcopy.iloc[:,i].apply(cleanstring)
            _uniqueNominalVals = [str(_i) for _i in np.unique(dfcopy.iloc[:,i])]
            _uniqueNominalVals = ",".join(_uniqueNominalVals)
            _uniqueNominalVals = _uniqueNominalVals.replace("[","")
            _uniqueNominalVals = _uniqueNominalVals.replace("]","")
            _uniqueValuesString = " {" + _uniqueNominalVals +"}"
            arffList.append("@ATTRIBUTE " + df.columns[i] + _uniqueValuesString + "\n")
        else:
            arffList.append("@ATTRIBUTE " + df.columns[i] + " real\n")
            #even if it is an integer, let's just deal with it as a real number for now
    arffList.append("@DATA\n")
    for i in range(dfcopy.shape[0]):#instances
        _instanceString = ""
        for j in range(df.shape[1]):#features
                if dfcopy.dtypes[j]=='O':
                    _instanceString+="\"" + str(dfcopy.iloc[i,j]) + "\""
                else:
                    _instanceString+=str(dfcopy.iloc[i,j])
                if j!=dfcopy.shape[1]-1:#if it's not the last feature, add a comma
                    _instanceString+=","
        _instanceString+="\n"
        if cleannan:
            _instanceString = _instanceString.replace("-999999999.0","?") #for numeric missing values
            _instanceString = _instanceString.replace("\"?\"","?") #for categorical missing values
        arffList.append(_instanceString)
    f.writelines(arffList)
    f.close()
    del dfcopy
    return True


def print_info(msg, marker='', cnt=4):
    """Function for simple terminal output formatting"""

    if not isinstance(msg, str) or not isinstance(marker, str):
        raise ValueError
    if not isinstance(cnt, int):
        raise ValueError

    if marker == '=':
        print('\n{}\n{}{}\n{}'.format('='*cnt, ' '*4, msg, '='*cnt))
    elif marker == '-':
        print('\n{}\n{}{}\n{}'.format('-'*cnt, ' '*4, msg, '-'*cnt))
    elif marker == '*':
        print('\n{}\n{}{}\n{}'.format('*'*cnt, ' '*4, msg, '*'*cnt))
    elif marker == ':':
        print('\n{} {}'.format(':'*cnt, msg))
    else:
        print('{} {}'.format(' '*cnt, msg))
