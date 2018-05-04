import pytest
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generalization.file_management import OutputManager, delete_old_sessions
from generalization.utils import ignore_warnings


@pytest.mark.slowtest
def test_delete_old_sessions(tmpdir):
    num_testfolders = 15
    num_keep_folders = 3

    for i in range(num_testfolders):
        tmpdir.mkdir('session_{}'.format(i))
        time.sleep(.1) # provide a coarse enough time resolution

    assert len(os.listdir(str(tmpdir))) == num_testfolders

    # test case: set higher limit to skip cleanup
    num_folders_before = len(os.listdir(str(tmpdir)))
    delete_old_sessions(str(tmpdir), keep_sessions=num_testfolders+1)
    assert len(os.listdir(str(tmpdir))) == num_folders_before

    # test case: remove all but the latest 'num_keep_folders' folders
    delete_old_sessions(str(tmpdir), keep_sessions=num_keep_folders)
    assert len(os.listdir(str(tmpdir))) == num_keep_folders

    for i in range(num_testfolders-num_keep_folders):
        assert not os.path.exists(str(tmpdir) + '/session_{}'.format(i))

    for i in range(num_testfolders-num_keep_folders, num_testfolders):
        assert os.path.exists(str(tmpdir) + '/session_{}'.format(i))

    # test case: call cleanup, but actually skip it
    num_folders_before = len(os.listdir(str(tmpdir)))
    delete_old_sessions(str(tmpdir), keep_sessions=None)
    assert len(os.listdir(str(tmpdir))) == num_folders_before

    # test case: check handling of invalid paths
    with pytest.raises(OSError):
        delete_old_sessions('no_chance_this_path_exists')

    # test misc. error handling
    with pytest.raises(ValueError):
        delete_old_sessions(False)

    with pytest.raises(ValueError):
        delete_old_sessions(str(tmpdir), keep_sessions='invalid type')

    with pytest.raises(ValueError):
        delete_old_sessions(str(tmpdir), keep_sessions=-1)


def test_OutputManager(tmpdir, monkeypatch):
    saved_path = os.getcwd()
    
    om = OutputManager(str(tmpdir) + '/output/')
    assert os.path.exists(str(tmpdir) + '/output/')

    os.chdir(str(tmpdir)+'/output/')
    
    try:
        session_name = sorted([f for f in os.listdir(str(tmpdir)+'/output/')],
                              key=os.path.getctime)[-1]
    except IndexError:
        raise IndexError('no session output folder created in {}'.format(
            str(tmpdir)))

    # test numpy object saving
    om.save(np.ones((3,3)), 'numpytest')
    assert os.path.exists(str(tmpdir)+'/output/'+session_name+'/numpytest.npy')

    # test matplotlib object saving
    om.save(plt.figure(), 'matplotlibtest')
    assert os.path.exists(str(tmpdir)+'/output/'+session_name+'/matplotlibtest.png')
    
    # test pandas dataframe saving
    om.save(pd.DataFrame([1,2,3]), 'pandastest')
    assert os.path.exists(str(tmpdir)+'/output/'+session_name+'/pandastest.npy')
    
    # test pandas dataframe saving (to ARFF)
    om.save(pd.DataFrame([1,2,3], columns=['duh']),
            'pandasarfftest', to_arff=True)
    assert os.path.exists(str(tmpdir)+'/output/'+session_name+'/pandasarfftest.arff')
    
    # test generic object saving
    om.save(['what','a','pointless','list'], 'pickletest')
    assert os.path.exists(str(tmpdir)+'/output/'+session_name+'/pickletest.pkl')

    # test error handling
    with pytest.raises(ValueError):
        om.save(np.ones((2,2)), 999)

    with pytest.raises(ValueError):
        om.save(np.ones((1,1)), 'numpyfail', folder=False)

    with pytest.raises(ValueError):
        om.save(np.ones((1,1)), 'arfffail', to_arff=999)
    
    # test subfolder creation
    subfolder_name = 'subfolder'
    om.save(['make','me','a','subfolder'], 'subfolderlist', folder=subfolder_name)
    assert os.path.exists(str(tmpdir)+'/output/'+session_name+'/'+subfolder_name+\
                          'subfolderlist.pkl')

    # test misc. error handling    
    with pytest.raises(ValueError):
        om_2 = OutputManager(False)

    with pytest.raises(ValueError):
        om_2 = OutputManager(str(tmpdir)+'/output_2', keep_sessions='invalid type')

    # test get_session_folder
    assert isinstance(om.get_session_folder(), str)
    with pytest.raises(OSError):
        om.get_session_folder('not an actual subfolder')

    with pytest.raises(ValueError):
        om.get_session_folder(False)

    # monkeypatching (changing certain attributes such that some test should fail)
    monkeypatch.setattr(om, 'session_dir', 'non-existing folder')
    
    with pytest.raises(IOError):
        om.get_session_folder()

    with pytest.raises(OSError):
        om.save(np.ones((3,3)), 'numpyfail')

    os.chdir(saved_path)
