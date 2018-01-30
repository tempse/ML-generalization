import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier


def parse_target_labels(y, pos_label, neg_label):
    """Parses labels in the target vector to 1 (positive label) and 0 (negative label)

    Args:
        y: the target vector with string labels in form of a Pandas dataframe

        pos_label: string with the name of the positive label

        neg_label: string with the name of the negative label

    Returns:
        The target vector with pos_label and neg_label encoded as ones and zeros

    """

    assert isinstance(y, pd.DataFrame) or isinstance(y, pd.core.series.Series)
    #assert isinstance(pos_label, str)
    #assert isinstance(neg_label, str)
    assert pos_label != neg_label

    if not isinstance(y, pd.core.series.Series) and y.shape[1]!=1:
        raise ValueError('target vector has to be one-dimensional, but has ' \
                         'shape {}'.format(y.shape))

    if not np.any((y==pos_label).astype(int)):
        raise ValueError('class {} not contained in target vector'.format(
            pos_label))

    if not np.any(y==neg_label):
        raise ValueError('class {} not contained in target vector'.format(
            neg_label))

    if not np.all((y==pos_label) | (y==neg_label)):
        raise ValueError('target vector contains more than two classes, ' \
                         'which is not supported yet')

    y = y.replace(pos_label, 1)
    y = y.replace(neg_label, 0)
    
    return y
    

class PreprocessingManager():

    def __init__(self, outpath):
        self.path       = outpath
        self.param_path = {}

        if not isinstance(outpath, str):
            raise TypeError('The variable "outpath" should be of type string \
                             however you provided a {}'.format(type(outpath)))

        if not self.path.endswith('/'):
            self.path += '/'

    def get_output_path(self):
        return self.path

    def standard_scale(self, X, load_fitted_attributes=False):
        """
        Args:
            X: the feature matrix (type: pandas dataframe or numpy array)

            load_fitted_attributes: bool that determines if we are dealing with
                                    train or test data
        _________________________________________________________________________

        Returns:
            the standard scaled feature matrix X
        _________________________________________________________________________
        
        Operations breakdown:
            standard scaling to 0-mean and std-derivation of 1

        """
        if not isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray):
            raise TypeError('Feature matrix should be \
                    a dataframe or a numpy array but instead is {}'.format(type(X)))

        if not load_fitted_attributes:
            print('\nStandard scaling training data...')
            if isinstance(X, pd.DataFrame):
                X = X.as_matrix()
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            Xfeats_mean = scaler.mean_
            Xfeats_scale = scaler.scale_
            Xfeats_var = scaler.var_

            self.param_path['standard_scale'] = self.path + 'standard_scale_attributes.npy'
            np.save(self.param_path['standard_scale'], 
                    np.array([Xfeats_mean, Xfeats_scale, Xfeats_var]))

        else: 
            try:
                filename = self.param_path['standard_scale']
                if not os.path.isfile(filename):
                    raise IOError('File {} does not exist'.format(filename))
            except KeyError:
                raise IOError('Parameters have not been fitted \
                        so no attributes cannot be loaded!')

            print('Loading previously determined scaler attributes...')
            print('Standard scaling test data...\n')
            scaler_attributes = np.load(filename)
            scaler_mean = scaler_attributes[0,:]
            scaler_scale = scaler_attributes[1,:]
            if isinstance(X, pd.DataFrame):
                X = X.as_matrix()
            X = X.astype(float)
            X -= scaler_mean
            X /= scaler_scale

        return pd.DataFrame(X)



    def parse_object_columns(self, X, load_fitted_attributes=False):
        """
        Args:
            X: the feature matrix (type: pandas dataframe)

            load_fitted_attributes: bool that determines if we are dealing with
                                    train or test data
        _________________________________________________________________________

        Returns:
            the parsed feature matrix X
        _________________________________________________________________________
        
        Operations breakdown:
            Parse all object-columns in X to integer 
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Feature matrix should be ' \
                            'a pandas dataframe but instead is {}'.format(type(X)))

        if not load_fitted_attributes:
            print('\nLooking for object columns to parse...')
            fix_list = [col for col in X.columns if X[col].dtype == np.dtype('O')]

            if not fix_list:
                print('No object columns to parse.\n')
                return X

            print('Parsing the following features: {}'.format(fix_list))
            dic_list = []
            
            for col in fix_list:
                X[col] = X[col].astype('category')
                # we have to save the mapping to the new values
                # and redo it the same way with the unlabled data
                # dictionary is as such: {0:'a', 1:'c',...}
                # therefore we have to reverse key and item
                dic = dict(enumerate(X[col].cat.categories))
                rev_dic = { v:k for k,v in dic.items() }
                dic_list.append(rev_dic)
                X[col] = X[col].cat.codes
                
            # we save the column and the corresponding mapping
            zipList = zip(fix_list, dic_list)
            self.param_path['parse_object_columns'] = self.path + 'parse_object_columns_mapping.npy'
            np.save(self.param_path['parse_object_columns'], zipList)

            # column is transformed in a floating point column
            # if nans are in it. This is corrected by __get_true_integer_columns
            X[fix_list] = X[fix_list].replace(-1, np.nan)

        else:
            try:
                filename = self.param_path['parse_object_columns']
                if not os.path.isfile(filename):
                    raise IOError('File {} does not exist'.format(scaler_attributes_filename))
            except KeyError:
                raise IOError('Parameters have not yet been fitted, ' \
                              'so attributes cannot be loaded!')
                    
            print('Picking up the parser map from a file to parse the test data...\n')
            # [()] returns a zip-list instead of a it being wrapped in numpy array
            parse_map = np.load(filename)[()]
            for col, dic in parse_map:
                if col in X.columns:
                    X[col] = X[col].astype('category')
                    X[col] = list(map(dic.get, X[col].tolist(), X[col].tolist()))
                    if X[col].dtype == np.dtype('O'):
                        warnings.warn('Training data has unknown object values \
                                in feature {}. Parsing them to minus signes!'.format(col))
                        unkn_list = X[col][X[col].apply(lambda x: type(x)!=int and \
                                                        (not pd.isnull(x)))].tolist()
                        unkn_dic = {unkn_list[idx]:-idx-2 for idx in range(len(unkn_list))}
                        X[col] = list(map(unkn_dic.get, X[col].tolist(), X[col].tolist()))
                        if X[col].dtype == np.dtype('O'): 
                            raise TypeError('Training data contains ' \
                                            'unknown class in column: {}'.format(col))             
        return X


    def fill_numerical(self, X, *args):
        """
        Args:
            X: the feature matrix (type: pandas dataframe)

            load_fitted_attributes: bool that determines if we are dealing with
                                    train or test data
        _________________________________________________________________________

        Returns:
            the feature matrix X with all nan values of 
            numerical columns filled up
        _________________________________________________________________________
        
        Operations breakdown:
            Fill missing numerical (nan values) values using average
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Feature matrix should be \
                    a pandas dataframe but instead is {}'.format(type(X)))

        #Fill average data in numerical column
        print('\nFilling up nan values of the numerical columns using averages...\n')
        NumericalData = [col for col in X.columns if X[col].dtype == np.dtype('float') and \
                                                  sum(X[col].isnull())>0]
        for col in NumericalData:
            X[col] = X[col].astype(float)
            Mean = np.mean(X[col])
            X[col] = X[col].fillna(Mean)

        return X
     


    def fill_categorical(self, X, *args):
        """
        Args:
            X: the feature matrix (type: pandas dataframe)

            load_fitted_attributes: bool that determines if we are dealing with
                                    train or test data
        _________________________________________________________________________

        Returns:
            the feature matrix X with all nan values of 
            categorical columns filled up
        _________________________________________________________________________
        
        Operations breakdown:
            Machine learning to fill categorical values in
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Feature matrix should be ' \
                            'a pandas dataframe but instead is {}'.format(type(X)))

        print('\nFilling up nan values of the categorical columns \
                using an ExtraTreesClassifier...\n')
        # labels list contains all columns that we have to fill
        labels_list = []
        # first of all we get all integer columns
        nom_list = self._get_true_integer_columns(X)

        for item in nom_list:
            if item in X.columns:
                if sum(X[item].isnull()) > 0:
                    labels_list.append(item)
      
        for item in labels_list:
            feature_list = [col for col in X.columns if col not in labels_list]
            X = self.__learn_vals(X,item,feature_list)

        return X

        
    def contains_nan(self, X, *args):
        """
        Args:
            X: the feature matrix (type: pandas dataframe)
        _________________________________________________________________________

        Returns:
            the number of colums in the feature matrix that contain
            nan values
        _________________________________________________________________________
        
        Operations breakdown:
            checks if column contrains nan and prints it out
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Feature matrix should be ' \
                            'a pandas dataframe but instead is {}'.format(type(X)))

        count = 0
        for col in X.columns:
            if sum(X[col].isnull()) > 0:
                count += 1

        return count


    def rm_correlated(self, X, load_fitted_attributes=False):
        """
        Args:
            X: the feature matrix (type: pandas dataframe)

            load_fitted_attributes: bool that determines if we are dealing with
                                    train or test data
        _________________________________________________________________________

        Returns:
            the feature matrix with columns removed from it
            that have similar entries than another columns
        _________________________________________________________________________
        
        Operations breakdown:
            Check for correlation between columns, and remove duplicate information

            Prior to rm_correlated, rm_low_variance should be called as this would
            get rid of vanishing stdevs in the correlation formula:
                cor(i,j) = cov(i,j)/[stdev(i)*stdev(j)]
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Feature matrix should be ' \
                            'a pandas dataframe but instead is {}'.format(type(X)))

        correlation = 0.99
        if not load_fitted_attributes:
            print('\nDropping highly correlated columns (w/ correlation > {})'.format(correlation))
            print('Training data...')
            # first we drop columns with only one value to avoid dividing by 0
            drop_list_zero_var = [col for col in X.columns if X[col].var()==0.]
            X.drop(drop_list_zero_var,axis = 1,inplace = True)

            drop_list = []
            corr_mat = X.corr()
            cols = list(corr_mat.columns)
            for n in range(corr_mat.shape[0]):
                for m in range(n+1,corr_mat.shape[0]):
                    if abs(corr_mat.loc[cols[n],cols[m]]) > correlation:
                        drop_list.append(cols[m])

            print('number of high correlated columns: {}'.format(len(list(set(drop_list)))))
            X.drop(drop_list,axis = 1,inplace = True)
            print('saving dropped columns...')
            self.param_path['rm_correlated'] = self.path + 'rm_correlated_dropped_cols.npy'
            drop_list.extend(drop_list_zero_var) 
            drop_list = list(set(drop_list))
            np.save(self.param_path['rm_correlated'], drop_list)
            
        else:
            try:
                filename = self.param_path['rm_correlated']
                if not os.path.isfile(filename):
                    raise IOError('File {} does not exist'.format(filename))
            except KeyError:
                raise IOError('Parameters have not been fitted ' \
                              'so attributes cannot be loaded!')
            print('Test data:')
            print('Loading list of columns to drop from file...\n')
            drop_list = np.load(filename)
            if(len(drop_list)>0 ):
                for element in drop_list:
                    if(element in X):
                        X.drop(drop_list,axis = 1,inplace = True)
     
        return X


    def rm_low_variance(self, X, load_fitted_attributes=False):
        """
        Args:
            X: the feature matrix (type: pandas dataframe)

            load_fitted_attributes: bool that determines if we are dealing with
                                    train or test data
        _________________________________________________________________________

        Returns:
            the feature matrix with columns removed from it
            that have a small variance in its values
        _________________________________________________________________________
        
        Operations breakdown:
            dropping colmns that have a too small variance
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Feature matrix should be ' \
                            'a pandas dataframe but instead is {}'.format(type(X)))

        variance = 0.001
        # when predicting we drop columns from a file that contains previously dropped cols
        if not load_fitted_attributes:
            print('\nDropping columns with low variance (w/ variance < {})'.format(variance))
            print('Training data...')
            drop_list = [col for col in X.columns if X[col].var() < variance ]

            print('number of low variance columns: {}'.format(len(list(set(drop_list)))))
            X.drop(drop_list,axis = 1,inplace = True)

            print('saving dropped columns...')
            self.param_path['rm_low_variance'] = self.path + 'rm_low_variance_drop_cols.npy'
            np.save(self.param_path['rm_low_variance'], drop_list)

        else:
            try:
                filename = self.param_path['rm_low_variance']
                if not os.path.isfile(filename):
                    raise IOError('File {} does not exist'.format(filename))
            except KeyError:
                raise IOError('Parameters have not been fitted ' \
                              'so attributes cannot be loaded!')

            print('Test data:')
            print('Loading columns to drop from file...\n')
            drop_list = np.load(filename)
            if(len(drop_list)>0 ):
                for element in drop_list:
                    if(element in X):
                        X.drop(drop_list,axis = 1,inplace = True)
     
        return X
       


    def _get_true_integer_columns(self, X):
        """
        Args:
            X: the feature matrix (type: pandas dataframe)
        _________________________________________________________________________

        Returns:
            A list of (true) integer-column names
        _________________________________________________________________________
        
        Operations breakdown:
            If a integer column has at least one nan value it is
            automatically percieved as a floating dtype. 
            We iterate trough the column and check each individual element.
            If a column then contrains only ints and nans we append it to the
            output list.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Feature matrix should be ' \
                            'a pandas dataframe but instead is {}'.format(type(X)))

        int_col_list   = [col for col in X.columns if X[col].dtype == np.dtype('int')]
        float_col_list = [col for col in X.columns if X[col].dtype == np.dtype('float')] 
        for col in float_col_list:
            if col in int_col_list:
                continue
            count = 0
            for val in X[col]:
                # we increment the count if we have a nan or an integer
                if pd.isnull(val):
                    count += 1
                if val.is_integer():
                    count += 1
            # after looping through column we check if the lenght 
            # and number of ints+nans are the same
            if count == len(X[col]):
                int_col_list.append(col)

        return int_col_list


    def __learn_vals(self, X, label, features):
        """
        Args:
            X: the feature matrix (type: pandas dataframe)

            label: column to learn

            features: columns to learn from
        _________________________________________________________________________

        Returns:
            the feature matrix with its categorical columns filled up
        _________________________________________________________________________
        
        Operations breakdown:
            The nan values in categorical columns are filled up by 
            using an ExtraTreesClassifier that is trained on columns
            that do not contain nan values
        """
        # Check that all columns are in data frame
        for item in features:
            if item not in list(X.columns):
                features.remove(item)

        if label not in list(X.columns) or sum(X[label].isnull())==0:
            return X
      
        print('Filling in {} values...'.format(label))
        et = ExtraTreesClassifier(n_estimators=100, 
                                  max_depth=None, 
                                  min_samples_split=2, 
                                  random_state=0,
                                  verbose=True,
                                  n_jobs=-1)
     
        # the labels ought not contain NaNs!
        labels_train = X[label][X[label].isnull() == 0].values

        features_train = X[features][X[label].isnull() == 0].values
        features_test  = X[features][X[label].isnull()].values

        et.fit(features_train,labels_train)
        labels_test = pd.Series(et.predict(features_test),index = X.index[X[label].isnull()])
        new_col = pd.concat([ X[label][X[label].isnull()==0] , labels_test] )
      
        X = X.combine_first( pd.DataFrame({label:new_col}) )

        return X
