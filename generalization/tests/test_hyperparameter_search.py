import unittest

from generalization.hyperparameter_search import hyperparameter_search


class TestHyperparameterSearch(unittest.TestCase):
    
    def test_hyperparameter_search(self):
        from sklearn.naive_bayes import GaussianNB, MultinomialNB
        from sklearn.svm import SVC
        from skopt import BayesSearchCV
        
        model = SVC()
        model = hyperparameter_search(model, 2, 3)
        self.assertTrue(isinstance(model, BayesSearchCV))

        model = GaussianNB()
        model = hyperparameter_search(model, 2, 3)
        self.assertTrue(isinstance(model, GaussianNB))

        model = MultinomialNB()
        with self.assertRaises(TypeError):
            model = hyperparameter_search(model, 2, 3)


if __name__ == '__main__':
    unittest.main()
 
