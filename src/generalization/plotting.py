import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)},
        font_scale=1.5)


def plot_performance_diff(*arrays, **options):
    """Plots performance difference vs. number of test samples for an arbitrary
    number of (mean,std) pairs.

    Args:
        arrays: data arrays, the first array has to be the array with the number
            of test splits, followed by pairs of (mean, standard deviation)
            arrays
        labels: list of string labels for the classifiers
        xmax: maximum value for x axis. If None, no limits are enforced.

    Returns:
        The created plot as a matplotlib figure

    """

    if len(arrays) < 3:
        raise ValueError('at least three arrays (splits, mean, std) ' \
                         'required as input')
    elif (len(arrays)%2) == 0:
        raise ValueError('even number of arrays, they have to come in ' \
                         'splits + n*(mean-std pairs)')

    if not all([isinstance(arrays[i], np.ndarray) for i,j in enumerate(arrays)]):
        raise ValueError('not all passed arrays are numpy arrays')

    labels = options.pop('labels', None)
    xlabel = options.pop('xlabel', 'x')
    ylabel = options.pop('ylabel', 'y')
    train_difference = options.pop('train_difference', None)
    xmax = options.pop('xmax', None)

    if options:
        raise TypeError('invalid options passed: {}'.format(options))

    if not isinstance(xlabel, str) and xlabel is not None:
        raise ValueError('xlabel has to be a string, but is {}.'.format(type(xlabel)))

    if not isinstance(ylabel, str) and ylabel is not None:
        raise ValueError('ylabel has to be a string, but is {}.'.format(type(ylabel)))

    if labels is not None and len(labels) != (len(arrays)-1)*0.5:
        raise ValueError('the number of labels does not match the number of ' \
                         'passed arrays: {}'.format(len(labels)))

    if not isinstance(train_difference, list) and train_difference is not None:
        raise ValueError('train_difference is not of type list, ' \
                         'but of type {}'.format(type(train_difference)))

    if not isinstance(xmax, int) and xmax is not None:
        raise ValueError('xmax has to be a positive integer, but is {}.'.format(
            xmax))

    sns.set_palette(sns.color_palette("deep", n_colors=int((len(arrays)-1)*0.5)))

    fig = plt.figure()

    curves = []

    for arr in range(1,len(arrays),2):
        curve, = plt.plot(arrays[0], arrays[arr])
        plt.fill_between(arrays[0],
                         arrays[arr]-arrays[arr+1], arrays[arr]+arrays[arr+1],
                         alpha=0.3)
        curves.append(curve)

    if train_difference is not None:
        for i,j in enumerate(train_difference):
            plt.plot(1,
                     train_difference[i],
                     marker='*',
                     markersize=15,
                     linestyle=None)

    xint = range(np.min(arrays[0].astype(int)), np.max(arrays[0].astype(int))+1)
    plt.xticks(xint)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #plt.ylim((-0.05,1.05))
    if xmax is not None:
        plt.xlim(xmax=xmax)

    if labels is not None:
        plt.legend(curves, labels, loc='best')

    plt.tight_layout()

    return fig
