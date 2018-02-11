import unittest

from generalization.plotting import plot_performance_diff


class TestPlotting(unittest.TestCase):

    def test_plot_performance_diff(self):
        import numpy as np
        import matplotlib

        split = np.arange(1,10)
        means_1 = np.array([i*np.random.normal(0.2,0.01) for i in range(1,10)])
        stds_1 = np.array([abs(i*np.random.normal(0.0,0.01)) for i in range(1,10)])
        means_2 = np.array([i*np.random.normal(0.1,0.05)+0.2 for i in range(1,10)])
        stds_2 = np.array([abs(i*np.random.normal(0.0,0.05)) for i in range(1,10)])
        labels = ['classifier 1', 'classifier 2']

        with self.assertRaises(ValueError):
            plot_performance_diff()

        with self.assertRaises(ValueError):
            plot_performance_diff(means_1)

        with self.assertRaises(ValueError):
            plot_performance_diff(means_1, stds_1)

        with self.assertRaises(TypeError):
            plot_performance_diff(split, means_1, stds_1, not_an_option='')

        with self.assertRaises(ValueError):
            plot_performance_diff(split, means_1, stds_1, means_2, stds_2,
                                  labels='not enough labels')

        fig = plot_performance_diff(split, means_1, stds_1, means_2, stds_2,
                                    labels=labels)
        assert isinstance(fig, matplotlib.figure.Figure)



if __name__ == '__main__':
    unittest.main()
