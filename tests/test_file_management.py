import os
import unittest
import tempfile

from generalization.file_management import OutputManager, delete_old_sessions
from generalization.utils import ignore_warnings


class TestFileManagement(unittest.TestCase):

    def setUp(self):
        if os.environ.get('DISPLAY') == '':
            print('No display name found. Using matplotlib Agg backend. ' \
                  '(Current class: {})'.format(self.__class__.__name__))
            import matplotlib
            matplotlib.use('Agg')
            
    def test_delete_old_sessions(self):
        import time, os

        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            num_testfolders = 15
            num_keep_folders = 3
            for i in range(num_testfolders):
                os.makedirs(tmpdir + '/session_{}'.format(i))
                time.sleep(.1) # provide a coarse enough time resolution

            assert len(os.listdir(tmpdir)) == num_testfolders

            # test case: set higher limit to skip cleanup
            num_folders_before = len(os.listdir(tmpdir))
            delete_old_sessions(tmpdir, keep_sessions=num_testfolders+1)
            assert len(os.listdir(tmpdir)) == num_folders_before

            # test case: remove all but the latest 'num_keep_folders' folders
            delete_old_sessions(tmpdir, keep_sessions=num_keep_folders)

            assert len(os.listdir(tmpdir)) == num_keep_folders

            for i in range(num_testfolders-num_keep_folders):
                assert not os.path.exists(tmpdir + '/session_{}'.format(i))

            for i in range(num_testfolders-num_keep_folders, num_testfolders):
                assert os.path.exists(tmpdir + '/session_{}'.format(i))

            # test case: call cleanup, but actually skip it
            num_folders_before = len(os.listdir(tmpdir))
            delete_old_sessions(tmpdir, keep_sessions=None)
            assert len(os.listdir(tmpdir)) == num_folders_before

            # test case: check handling of invalid paths
            with self.assertRaises(OSError):
                delete_old_sessions('no_chance_this_path_exists')


    @ignore_warnings
    def test_OutputManager(self):
        import os
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            om = OutputManager('output/')

            assert os.path.exists(tmpdir+'/output/')

            os.chdir(tmpdir+'/output/')

            try:
                session_name = sorted([f for f in os.listdir(tmpdir+'/output/')],
                                      key=os.path.getctime)[-1]
            except IndexError:
                raise IndexError('no session output folder created in {}'.format(
                    tmpdir))

            # test numpy object saving
            om.save(np.ones((3,3)), 'numpytest')
            assert os.path.exists(tmpdir+'/output/'+session_name+'/numpytest.npy')

            # test matplotlib object saving
            om.save(plt.figure(), 'matplotlibtest')
            assert os.path.exists(tmpdir+'/output/'+session_name+'/matplotlibtest.png')

            # test pandas dataframe saving
            om.save(pd.DataFrame([1,2,3]), 'pandastest')
            assert os.path.exists(tmpdir+'/output/'+session_name+'/pandastest.npy')

            # test pandas dataframe saving (to ARFF)
            om.save(pd.DataFrame([1,2,3], columns=['duh']),
                    'pandasarfftest', to_arff=True)
            assert os.path.exists(tmpdir+'/output/'+session_name+'/pandasarfftest.arff')

            # test generic object saving
            om.save(['what','a','pointless','list'], 'pickletest')
            assert os.path.exists(tmpdir+'/output/'+session_name+'/pickletest.pkl')

            # test subfolder creation
            subfolder_name = 'subfolder'
            om.save(['make','me','a','subfolder'], 'subfolderlist', folder=subfolder_name)
            assert os.path.exists(tmpdir+'/output/'+session_name+'/'+subfolder_name+\
                                  'subfolderlist.pkl')

            # test get_session_folder
            assert isinstance(om.get_session_folder(), str)
            with self.assertRaises(OSError):
                om.get_session_folder('notAnActualSubfolder')



if __name__ == '__main__':
    unittest.main()
