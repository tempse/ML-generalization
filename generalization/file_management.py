import os
import errno
import shutil
import datetime
import warnings
import pickle
import arff

import matplotlib
import numpy as np
import pandas as pd

from generalization.utils import pandas2arff


session_prefix = 'session_'


def delete_old_sessions(folder, keep_sessions=20):
    """Keeps a given number of newest session folders and deletes all others.

    Objects of the class OutputManager create new session folders at each run.
    This function searches the relevant directory, keeps a user-specified number
    of latest session outputs and deletes all older ones.

    Args:
        folder: path to the directory with all the session output folders

        keep_sessions: number of session folders to keep (default: 20)

    """

    if not isinstance(folder, str):
        raise ValueError('folder attribute is not a string')
    
    if not isinstance(keep_sessions, int) and keep_sessions is not None:
        raise ValueError('keep_sessions attribute is not int or None')

    if keep_sessions is not None and keep_sessions <= 0:
        raise ValueError('variable "keep_sessions" must be > 0')
    
    if keep_sessions is None:
        # skip any cleaning operations
        return
    
    if not os.path.exists(folder):
        raise OSError('path {} does not exist, unable to clean ' \
                      'old session files'.format(folder))

    if os.path.isabs(folder):
        output_basepath = folder
    else:
        output_basepath = os.path.abspath(folder)
    
    if not output_basepath.endswith('/'):
        output_basepath += '/'

    initial_path = os.getcwd()+'/'
    
    # create list of session names, sorted from newest to oldest
    os.chdir(output_basepath)
    files = sorted([f for f in os.listdir(output_basepath) \
                    if f.startswith(session_prefix)],
                   key=os.path.getctime,
                   reverse=True)
    
    if len(files) < keep_sessions:
        os.chdir(initial_path)
        # session number below the limit, nothing else to be done
        return

    for i in range(keep_sessions, len(files)):
        try:
            shutil.rmtree(files[i])
            print('Cleanup: removed folder {}'.format(files[i]))
        except Exception as e:
            if e.errno == e.ENOENT:
                warnings.warn('folder {} not found, thus not deleted'.format(
                    files[i]))
            if e.errno == e.ENOTDIR:
                os.remove(files[i])

    os.chdir(initial_path)

class OutputManager:
    """Creates run-specific session folders and directs the output to them.

    Upon initialization, this class generates a session folder that is specific
    to each run (named by date and time). Any object that needs saving can be
    directed to the current session folder by invoking the implemented 'save'
    method. Per default, this class triggers the cleanup of old session folders.

    Args:
        output_basepath: path at which the session output folder is generated

        keep_sessions: integer specifying the number of sessions to keep;
            if None, cleanup of old session folders will be deactivated
            (default: 20)

    Methods:
        save(output_object, output_name, folder=None): stores the passed object
            in the current session's output folder. The 'folder' argument takes
            an optional string and creates a corresponding subfolder. Objects of
            certain supported types are stored via their own methods. All other
            objects are exported using pickle.

    """

    def __init__(self, output_basepath='output/', keep_sessions=20):

        if not isinstance(output_basepath, str):
            raise ValueError('output_basepath attribute is not a string')
        
        if not isinstance(keep_sessions, int) and keep_sessions is not None:
            raise ValueError('keep_sessions is not int or None')

        if not os.path.isabs(output_basepath):
            output_basepath = os.path.abspath(output_basepath)
        
        if not output_basepath.endswith('/'):
            output_basepath += '/'

        self.keep_sessions = keep_sessions

        now = datetime.datetime.now()
            
        self.session_dir = output_basepath + session_prefix + \
                           '{}-{}-{}_{}-{}-{}/'.format(
                               now.year,
                               now.month,
                               now.day,
                               now.hour,
                               now.minute,
                               now.second)
        
        try:
            os.makedirs(self.session_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                raise OSError('session output folder already exists ' \
                              '(just re-run the program to fix this)')
            else:
                raise OSError('cannot create session output folder: {}'.format(
                    output_basepath+self.session_dir))

        if keep_sessions is not None:
            delete_old_sessions(output_basepath, keep_sessions)


    def get_session_folder(self, subfolder=None):
        """Returns the path of the current session folder

        Args:
            subfolder: optional subfolder string (default: None)

        Returns:
            The path of the current session folder (incl. a subfolder, if specified).
        """
        
        if not isinstance(subfolder, str) and subfolder is not None:
            raise ValueError('subfolder attribute is not string or None')
        
        if not os.path.exists(self.session_dir):
            raise IOError('path {} does not exist'.format(self.session_dir))
        
        if subfolder is not None and \
           not os.path.exists(self.session_dir+'/'+subfolder):
            raise OSError('the folder {} does not exist in {}'.format(
                subfolder, self.session_dir))
        
        return self.session_dir

            
    def save(self, output_object, output_name, folder=None, to_arff=False):
        """Stores the passed object in the current session's output folder.

        Args:
            output_object: the object to be stored

            output_name: the output file name

            folder: optional subfolder name (if None, no subfolder is created)

        """

        if not os.path.exists(self.session_dir):
            raise OSError('output path {} does not exist ' \
                          '(just re-run the program to fix this)'.format(
                              self.session_dir))

        if not isinstance(output_name, str):
            raise ValueError('output_name attribute is not a string')
        
        if not isinstance(folder, str) and folder is not None:
            raise ValueError('folder attribute is not string or None')
        
        if not isinstance(to_arff, bool) and to_arff is not None:
            raise ValueError('to_arff attribute is not boolean or None')

        output_basename = self.session_dir

        if folder is not None:
            output_basename += folder
            try:
                os.makedirs(output_basename)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        output_basename += output_name
                
        if isinstance(output_object, matplotlib.figure.Figure):
            output_object.savefig(output_basename + '.pdf')
            output_object.savefig(output_basename + '.png')

        elif isinstance(output_object, np.ndarray):
            np.save(output_basename + '.npy', output_object,
                    allow_pickle=False)

        elif isinstance(output_object, dict) and to_arff:
            arff.dump(output_object, open(output_basename+'.arff','w'))

        elif isinstance(output_object, pd.core.frame.DataFrame):
            if to_arff:
                pandas2arff(output_object, output_basename + '.arff',
                            wekaname=output_name)
            else:
                np.save(output_basename + '.npy', output_object.as_matrix(),
                        allow_pickle=True)
            
        else:
            warnings.warn('no methods to save objects of type {} implemented, ' \
                          'using pickle to store the object'.format(
                              type(output_object)))
            with open(output_basename + '.pkl', 'wb') as picklefile:
                pickle.dump(output_object, picklefile)


