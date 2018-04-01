# ML-generalization, A Python Framework for Studies of Mac# Usage

In order to execute the program, run `python generalization.py`.

For the most part, the program is controlled by the three configuration files located in `config/`.

## Configure run parameters in `run_params.conf`
The default section is called `[run_params]`. If not otherwise specified, the parameters contained in it are loaded upon program execution.
> If a different set of run parameters should be used, just define a custom section and pass its name to the program as the terminal argument `-run_mode <my_parameter_set>` to invoke loading of these parameters instead of the default ones.

Required arguments for `run_params.conf`:
- `frac_train_sample`: fractional size of training sample (float)
- `num_test_samples`: number of test samples (integer)
- `num_CV_folds`: number of cross-validation folds (integer)
- `do_optimize_params`: specify whether automatic parameter optimization should be performed instead of using the model parameters defined in `model_params.conf`
- `n_iter`: number of parameter settings that are samples during hyperparameter optimization via Bayesian Optimization

## Configure data parameters in `data_params.conf`
In this configuration file, a virtually arbitrary number of parameter sections can be defined. All datasets that are defined here are processed by the program.

Required arguments for `data_params.conf`:
- `data_identifier`: a string that unambiguously identifies the dataset (string)
- `data_path`: relative or absolute path to the data file (string)
- `data_read_func`: name of a Pandas method to read the file (string)
- `data_target_col`: name of the column holding the true class label (string)
- `data_target_positive_label`: value of the positive class label
- `data_target_negative_label`: value of the negative class label
- `data_preprocessing`: list of preprocessing methods that are applied to the data in the given order
    > currently implemented functions: `standard_scale`, `parse_object_columns`, `fill_numerical`, `fill_categorical`, `rm_correlated`, `rm_low_variance`

Optional arguments for `data_params.conf`:
All arguments that are known ones for the function given in `data_read_func` can be stated here. *(Example: For the Pandas method `read_csv', valid options are "sep", "header", etc...)*

## Configure model parameters in `model_params.conf`
In this configuration file, a virtually arbitrary number of model parameter sections can be defined. All models that are defined here are processed by the program. Parameters that are not explicitly specified here take on their scikit-learn default values
> The section header (the string between the square brackets) must be identical to the scikit-learn model name of the algorithm (e.g., "KNeighborsClassifier", "RandomForestClassifier",...)!

## Results output structure

Each program run, a unique *session folder* is created in the subfolder `output/` that contains date and time of when the script was started. All output files of the particular run are stored in this folder. Per default, only the 20 latest sessions folders are kept and older ones are removed when the program is started again. In this way, the results are not only clearly separated for each program run, but accidental overwriting of old results can be largely avoided.
> The automatic cleanup of old session folders can be deactivated altogether by setting `keep_sessions=None` at initialization of the `OutputManager` object in the source code.