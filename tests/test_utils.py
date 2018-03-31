import unittest

from sklearn.datasets import make_classification

from generalization.utils import get_optimal_CV_n_folds


class TestUtils(unittest.TestCase):

    def test_get_optimal_CV_nfolds(self):
        X, y = make_classification(random_state=3)
        nfolds = get_optimal_CV_n_folds(X, y)
        self.assertEqual(nfolds, 5)

        X, y = make_classification(random_state=1)
        nfolds = get_optimal_CV_n_folds(X, y)
        self.assertEqual(nfolds, 3)


if __name__ == '__main__':
    unittest.main()
