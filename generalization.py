import sys
import warnings
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
#from sklearn.svm import SVC

from skopt.callbacks import DeltaXStopper

from generalization.file_management import OutputManager
from generalization.control import config_file_to_dict, check_data_config_requirements
from generalization.load_data import load_data
from generalization.preprocessing import PreprocessingManager, parse_target_labels
from generalization.utils import logger, print_class_counts, get_optimal_CV_n_folds, \
                                 get_search_results, print_info
from generalization.hyperparameter_search import hyperparameter_search
from generalization.evaluation import evaluate_nfold, performance_difference
from generalization.plotting import plot_performance_diff


supported_models = {'KNeighborsClassifier': KNeighborsClassifier,
                    'RandomForestClassifier': RandomForestClassifier,
                    'GaussianNB': GaussianNB,
                    'GradientBoostingClassifier': GradientBoostingClassifier,
                    'MLPClassifier': MLPClassifier}

def main():

    start_time_main = time.time()

    print_info('Reading config files...', ':')
    run_config = config_file_to_dict(config_path + 'run_params.conf')
    data_config = config_file_to_dict(config_path + 'data_params.conf')
    model_config = config_file_to_dict(config_path + 'model_params.conf')

    if run_mode_user in run_config:
        frac_train_sample = run_config[run_mode_user]['frac_train_sample']
        num_test_samples = run_config[run_mode_user]['num_test_samples']
        num_CV_folds = run_config[run_mode_user]['num_CV_folds']
        do_optimize_params = run_config[run_mode_user]['do_optimize_params']
        n_iter = run_config[run_mode_user]['n_iter']
        print_info('Chosen run mode is {}: {}'.format(run_mode_user,
                                                      run_config[run_mode_user]))
    else:
        raise KeyError('{} is not a valid run mode setting ' \
                       '(use, e.g., "run_params")'.format(
            run_mode_user))

    # collection of performance measures to be applied to the test set(s)
    scoring_funcs = ['accuracy_score', 'precision_score', 'recall_score', \
                     'f1_score']

    final_results_labels = [
        'dataset',
        'model',
        'model_params',
        'num_test_sets',
        'num_CV_folds',
        'elapsed_time_train',
        'elapsed_time_test'
    ]
    final_results_labels += ['test_{}_1fold'.format(i) for i in scoring_funcs]
    final_results_labels += ['train_{}'.format(i) for i in scoring_funcs]
    final_results_labels += ['test_{}'.format(i) for i in scoring_funcs]
    final_results_labels += ['test_{}_bootstrap'.format(i) for i in scoring_funcs]
    final_results_labels += ['test_{}_diff_max'.format(i) for i in scoring_funcs]
    final_results_labels += ['test_{}_diff_max_bootstrap'.format(i) for i in scoring_funcs]
    final_results_labels += ['test_{}_diff_mean'.format(i) for i in scoring_funcs]
    final_results_labels += ['test_{}_diff_std'.format(i) for i in scoring_funcs]
    final_results_labels += ['test_{}_diff_mean_bootstrap'.format(i) for i in scoring_funcs]
    final_results_labels += ['test_{}_diff_std_bootstrap'.format(i) for i in scoring_funcs]

    final_results = pd.DataFrame(columns=final_results_labels)


    # loop over all sections of the data params config file
    for d_cnt,d in enumerate(data_config):

        print_info('Processing dataset: {} ({} of {})'.format(
            d,d_cnt+1,len(data_config)), '=', 50)

        current_data_results = {}

        current_data_params = data_config[d]
        check_data_config_requirements(current_data_params)

        print_info('Loading data...', ':')
        data = load_data(current_data_params)

        print_info('Preparing target vector...', ':')
        X = data.drop(current_data_params['data_target_col'], axis=1)
        y = data[current_data_params['data_target_col']]

        y = parse_target_labels(y,
                                current_data_params['data_target_positive_label'],
                                current_data_params['data_target_negative_label'])

        del data

        print_info('Dimensions of feature matrix X: {}'.format(X.shape))
        print_info('Dimensions of target vector y:  {}'.format(y.shape))

        print_info('Splitting the data: splitting off the training sample...', ':')
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=1-frac_train_sample)

        del X, y

        print_info('Preprocessing the data...', ':')
        pm = PreprocessingManager(om.get_session_folder())

        for func in current_data_params['data_preprocessing']:
            X_train = getattr(pm, func)(X_train, False)
            X_test = getattr(pm, func)(X_test, True)

        print_class_counts(y_train, 'training', background=0, signal=1)
        print_class_counts(y_test, 'test', background=0, signal=1)

        # hyperparameter optimization, if required
        if num_CV_folds is None:
            print_info('Optimizing the number of cross-validation folds...', ':')
            num_CV_folds = get_optimal_CV_n_folds(X_train.as_matrix(), y_train.as_matrix())


        for mod in model_config:

            print_info('Training model: {}'.format(mod), '-', 50)

            try:
                model_params = model_config[mod]
                model = supported_models[mod](**model_params)
            except KeyError:
                raise KeyError('Model {} not supported. Choose a valid input ' \
                               'from this list: {}'.format(mod, supported_models))

            fitkwargs = {'X': X_train, 'y': y_train}
            if do_optimize_params:
                print_info('Optimizing hyperparameters...', ':')
                model = hyperparameter_search(model, n_iter, num_CV_folds)
                if mod != 'GaussianNB':
                    fitkwargs['callback'] = DeltaXStopper(1e-2)

            start_time_train = time.time()
            print_info('Fitting the model...', ':')
            model.fit(**fitkwargs)
            elapsed_time_train = time.time() - start_time_train

            model_parameters = get_search_results(model)

            # evaluate model on the training sample
            print_info('Evaluating the model on the training sample...', ':')
            for scoring_func in scoring_funcs:
                try:
                    model_scores_train = evaluate_nfold(X_train, y_train, model, 1,
                                                        scoring=scoring_func)
                    current_data_results['train_{}'.format(
                        scoring_func)] = model_scores_train[0]
                except ValueError:
                    warnings.warn('ValueError when evaluating with {}. ' \
                                  'Ignoring and continuing...'.format(
                                      scoring_func))


            # evaluate model on the test sample(s)
            print_info('Evaluating the model on the test sample(s)...', ':')

            test_performance_1fold = -1 # must be initialized with a negative number

            for t in range(1,num_test_samples+1):

                start_time_test = time.time()

                for scoring_func in scoring_funcs:
                    try:
                        model_scores_test = evaluate_nfold(X_test, y_test, model, t,
                                                           scoring=scoring_func,
                                                           bootstrapping=False)
                        model_scores_test_bootstrap = evaluate_nfold(X_test, y_test, model, t,
                                                                     scoring=scoring_func,
                                                                     bootstrapping=True)

                        if test_performance_1fold < 0:
                            test_performance_1fold = model_scores_test[0]
                        else:
                            pass

                        current_data_results['test_{}_1fold'.format(
                            scoring_func)] = test_performance_1fold

                        current_data_results['test_{}'.format(
                            scoring_func)] = str(model_scores_test)
                        current_data_results['test_{}_bootstrap'.format(
                            scoring_func)] = str(model_scores_test_bootstrap)

                        current_data_results['test_{}_diff_max'.format(
                            scoring_func)] = max(model_scores_test) - min(model_scores_test)

                        current_data_results['test_{}_diff_max_bootstrap'.format(
                            scoring_func)] = max(model_scores_test_bootstrap) - min(model_scores_test_bootstrap)

                        scores_mean, scores_std = performance_difference(model_scores_test)
                        current_data_results['test_{}_diff_mean'.format(
                            scoring_func)] = scores_mean
                        current_data_results['test_{}_diff_std'.format(
                            scoring_func)] = scores_std

                        scores_mean_bootstrap, scores_std_bootstrap = performance_difference(
                            model_scores_test_bootstrap)
                        current_data_results['test_{}_diff_mean_bootstrap'.format(
                            scoring_func)] = scores_mean_bootstrap
                        current_data_results['test_{}_diff_std_bootstrap'.format(
                            scoring_func)] = scores_std_bootstrap

                    except ValueError:
                        warnings.warn('ValueError when evaluating with {}. ' \
                                      'Ignoring and continuing...'.format(
                                          scoring_func))
                        current_data_results['test_{}_1fold'.format(
                            scoring_func)] = -1
                        #current_data_results['test_{}'.format(
                        #    scoring_func)] = "-1"
                        #current_data_results['test_{}_bootstrap'.format(
                        #    scoring_func)] = "-1"
                        current_data_results['test_{}_diff_mean'.format(
                            scoring_func)] = -1
                        current_data_results['test_{}_diff_std'.format(
                            scoring_func)] = -1
                        current_data_results['test_{}_diff_mean_bootstrap'.format(
                            scoring_func)] = -1
                        current_data_results['test_{}_diff_std_bootstrap'.format(
                            scoring_func)] = -1

                print_info('Model score differences (mean, std) for {} ' \
                           'test sample folds: {:.5f}, {:.5f}'.format(
                               t, scores_mean, scores_std))

                model_params_string = ','.join('{}:{}'.format(key, val) \
                                              for key, val in \
                                               sorted(model_parameters.items()))

                current_data_results['dataset'] = str(d)
                current_data_results['model'] = str(mod)
                current_data_results['model_params'] = model_params_string
                current_data_results['num_test_sets'] = t
                current_data_results['num_CV_folds'] = num_CV_folds
                current_data_results['elapsed_time_train'] = elapsed_time_train
                current_data_results['elapsed_time_test'] = time.time()-start_time_test

                final_results = final_results.append(current_data_results,
                                                     ignore_index=True)



        print_info('Creating results plots...', ':')
        scoring_func_plot = 'f1_score'

        train_differences = []

        current_data_plot_nsplits = final_results.query('(dataset=="{}") & (model=="{}")'.format(
            d,mod))['num_test_sets']

        # explicit conversion to floats is necessary for the np.isfinite method,
        # which is implicitely called during plotting
        current_data_plot_xyvals = [current_data_plot_nsplits.values.astype(np.float32)]
        current_data_plot_xyvals_bootstrap = [current_data_plot_nsplits.values.astype(np.float32)]

        current_data_plot_xyvals_max = [current_data_plot_nsplits.values.astype(np.float32)]
        current_data_plot_xyvals_max_bootstrap = [current_data_plot_nsplits.values.astype(np.float32)]


        for mod in model_config:
            current_data_plot_xyvals.append(
                final_results.query('(dataset=="{}") & (model=="{}")'.format(
                    d,mod))['test_{}_diff_mean'.format(scoring_func_plot)].values.astype(np.float32))
            current_data_plot_xyvals.append(
                final_results.query('(dataset=="{}") & (model=="{}")'.format(
                    d,mod))['test_{}_diff_std'.format(scoring_func_plot)].values.astype(np.float32))

            current_data_plot_xyvals_max.append(
                final_results.query('(dataset=="{}") & (model=="{}")'.format(
                    d,mod))['test_{}_diff_max'.format(scoring_func_plot)].values.astype(np.float32))
            current_data_plot_xyvals_max.append(
                    np.zeros(current_data_plot_xyvals_max[-1].shape))

            current_data_plot_xyvals_max_bootstrap.append(
                final_results.query('(dataset=="{}") & (model=="{}")'.format(
                    d,mod))['test_{}_diff_max_bootstrap'.format(scoring_func_plot)].values.astype(np.float32))
            current_data_plot_xyvals_max_bootstrap.append(
                    np.zeros(current_data_plot_xyvals_max_bootstrap[-1].shape))

            train_differences.append(
                abs(final_results.query('(dataset=="{}") & '\
                                        '(model=="{}")'.format(
                                            d,mod))['train_{}'.format(
                                                scoring_func_plot)].iloc[0] -
                    final_results.query('(dataset=="{}") & (model=="{}")'.format(
                        d,mod))['test_{}_1fold'.format(scoring_func_plot)].iloc[0])
            )

            current_data_plot_xyvals_bootstrap.append(
                final_results.query('(dataset=="{}") & (model=="{}")'.format(
                    d,mod))['test_{}_diff_mean_bootstrap'.format(
                        scoring_func_plot)].values.astype(np.float32))
            current_data_plot_xyvals_bootstrap.append(
                final_results.query('(dataset=="{}") & (model=="{}")'.format(
                    d,mod))['test_{}_diff_std_bootstrap'.format(
                        scoring_func_plot)].values.astype(np.float32))

        xmax_list = [None]
        for i in range(10,100,10):
            if num_test_samples > i:
                xmax_list.append(i)

        for lim in xmax_list:
            current_data_plot = plot_performance_diff(
                *current_data_plot_xyvals,
                labels = [m for m in model_config],
                train_difference=train_differences,
                xmax=lim,
                xlabel='number of samples',
                ylabel='mean performance difference'
            )
            plot_filename = '{}_performance-diff_num-splits_full'.format(d)
            if lim is not None:
                plot_filename += '_zoomed-{}'.format(lim)

            om.save(current_data_plot, plot_filename)

            current_data_plot = plot_performance_diff(
                *current_data_plot_xyvals,
                labels = [m for m in model_config],
                xmax=lim,
                xlabel='number of samples',
                ylabel='mean performance difference'
            )
            plot_filename = '{}_performance-diff_num-splits'.format(d)
            if lim is not None:
                plot_filename += '_zoomed-{}'.format(lim)

            om.save(current_data_plot, plot_filename)

            current_data_plot = plot_performance_diff(
                *current_data_plot_xyvals_max,
                labels = [m for m in model_config],
                xmax=lim,
                xlabel='number of samples',
                ylabel='maximum performance difference'
            )
            plot_filename = '{}_performance-diff_max_num-splits'.format(d)
            if lim is not None:
                plot_filename += '_zoomed-{}'.format(lim)

            om.save(current_data_plot, plot_filename)

            current_data_plot = plot_performance_diff(
                *current_data_plot_xyvals_max_bootstrap,
                labels = [m for m in model_config],
                xmax=lim,
                xlabel='number of samples',
                ylabel='maximum performance difference'
            )
            plot_filename = '{}_performance-diff_max_num-splits_bootstrap'.format(d)
            if lim is not None:
                plot_filename += '_zoomed-{}'.format(lim)

            om.save(current_data_plot, plot_filename)

            current_data_plot = plot_performance_diff(
                *current_data_plot_xyvals_bootstrap,
                labels = [m for m in model_config],
                train_difference=train_differences,
                xmax=lim,
                xlabel='number of samples',
                ylabel='mean performance difference'
            )
            plot_filename = '{}_performance-diff_num-splits_full_bootstrap'.format(d)
            if lim is not None:
                plot_filename += '_zoomed-{}'.format(lim)

            om.save(current_data_plot, plot_filename)

            current_data_plot = plot_performance_diff(
                *current_data_plot_xyvals_bootstrap,
                labels = [m for m in model_config],
                xmax=lim,
                xlabel='number of samples',
                ylabel='mean performance difference'
            )
            plot_filename = '{}_performance-diff_num-splits_bootstrap'.format(d)
            if lim is not None:
                plot_filename += '_zoomed-{}'.format(lim)

            om.save(current_data_plot, plot_filename)


        print_info('Saving the final results...', ':')
        om.save(final_results, '{}_final-results'.format(d))

        final_results_dict = final_results.to_dict('dict')
        final_results_dict['relation'] = str(d) # needed for ARFF
        final_results_dict['description'] = u'' # needed for ARFF
        om.save(final_results, '{}_final-results'.format(d), to_arff=True)

        print_info('\n')
        print_info('Everything done. (Elapsed overall time: {} seconds)\n'.format(
            time.time() - start_time_main))


if __name__ == '__main__':

    if sys.version_info[0] < 3:
        warnings.warn('This software was developed using Python 3.5.2, ' \
                      'you are using Python {}.{}.{}. ' \
                      'Proceed at your own risk.'.format(
                          sys.version_info[0],
                          sys.version_info[1],
                          sys.version_info[2]))
        time.sleep(10)

    output_path = 'output/'
    config_path = 'config/'

    om = OutputManager(output_path, keep_sessions=20)
    sys.stdout = logger(om.get_session_folder())

    # fix random seed for reproducibility
    np.random.seed(7)

    user_argv = None

    if(len(sys.argv) > 1):
        user_argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-run_mode', '-run_setting',
                        help='keyword to identify which run settings to choose from the config file (default: "run_params")',
                        action='store',
                        dest='run_mode',
                        default='run_params',
                        type=str)
    command_line_args = parser.parse_args(user_argv)

    run_mode_user = command_line_args.run_mode

    main()
