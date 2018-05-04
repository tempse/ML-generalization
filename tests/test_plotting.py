import pytest
import os
import numpy as np
import matplotlib

from generalization.plotting import plot_performance_diff


def test_plot_performance_diff():
    split = np.arange(1,10)
    means_1 = np.array([i*np.random.normal(0.2,0.01) for i in range(1,10)])
    stds_1 = np.array([abs(i*np.random.normal(0.0,0.01)) for i in range(1,10)])
    means_2 = np.array([i*np.random.normal(0.1,0.05)+0.2 for i in range(1,10)])
    stds_2 = np.array([abs(i*np.random.normal(0.0,0.05)) for i in range(1,10)])
    labels = ['classifier 1', 'classifier 2']

    with pytest.raises(ValueError):
        plot_performance_diff()

    with pytest.raises(ValueError):
        plot_performance_diff(means_1)

    with pytest.raises(ValueError):
        plot_performance_diff(means_1, stds_1)

    with pytest.raises(TypeError):
        plot_performance_diff(split, means_1, stds_1, invalid_option='')

    with pytest.raises(ValueError):
        plot_performance_diff(split, means_1, stds_1, means_2, stds_2,
                              labels='not enough labels')

    with pytest.raises(ValueError):
        plot_performance_diff(split, means_1, stds_1, means_2)

    with pytest.raises(ValueError):
        plot_performance_diff(split, means_1, stds_1, means_2,
                              'not a numpy array')

    with pytest.raises(ValueError):
        plot_performance_diff(split, means_1, stds_1, xlabel=False)

    with pytest.raises(ValueError):
        plot_performance_diff(split, means_1, stds_1, ylabel=False)

    with pytest.raises(ValueError):
        plot_performance_diff(split, means_1, stds_1, train_difference=42)

    with pytest.raises(ValueError):
        plot_performance_diff(split, means_1, stds_1, xmax=-1.0,
                              labels=['classifier 1'])

    fig = plot_performance_diff(split, means_1, stds_1, means_2, stds_2,
                                labels=labels)

    fig = plot_performance_diff(split, means_1, stds_1, means_2, stds_2,
                                labels=labels, train_difference=[1,2])
    
    fig = plot_performance_diff(split, means_1, stds_1, means_2, stds_2,
                                train_difference=[1,2], xmax=2)
    
    assert isinstance(fig, matplotlib.figure.Figure)
