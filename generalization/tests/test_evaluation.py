import unittest

from sklearn.datasets import make_classification

from utils import ignore_warnings
from generalization.evaluation import performance_difference, evaluate_nfold


class TestEvaluation(unittest.TestCase):
    
    @ignore_warnings
    def test_evaluate_nfold(self):
        from sklearn.naive_bayes import GaussianNB
        from sklearn.model_selection import train_test_split
        import pandas as pd
        
        X, y = make_classification(n_features=5, random_state=1)
        model = GaussianNB()

        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.5,
                                                            random_state=1)
        model.fit(X_train, y_train)
        num_folds = 3
        
        scores = evaluate_nfold(X_test, y_test, model, num_folds, randomize=False)
        self.assertEqual(len(scores), num_folds)
        self.assertTrue(all((scores[i]>=0) & (scores[i]<=1) \
                            for i in range(len(scores))))
        
        scores = evaluate_nfold(X_test, y_test, model, num_folds, randomize=True)
        self.assertEqual(len(scores), num_folds)
        self.assertTrue(all((scores[i]>=0) & (scores[i]<=1) \
                            for i in range(len(scores))))

        scores = evaluate_nfold(pd.DataFrame(X_test), pd.DataFrame(y_test),
                                model, num_folds, randomize=True)
        self.assertEqual(len(scores), num_folds)
        self.assertTrue(all((scores[i]>=0) & (scores[i]<=1) \
                            for i in range(len(scores))))
        
        num_folds = 0
        with self.assertRaises(ValueError):
            scores = evaluate_nfold(X_test, y_test, model, num_folds, randomize=False)
            
    
    def test_performance_difference(self):
        a = [1,2,3]
        mean, std = performance_difference(a)
        self.assertAlmostEqual(mean, 4/3.0)
        self.assertAlmostEqual(std, 0.47140452)

        b = [1,3]
        mean, std = performance_difference(b)
        self.assertEqual(mean, 2)
        self.assertEqual(std, 0)

        c = [5]
        mean, std = performance_difference(c)
        self.assertEqual(mean, 0.)
        self.assertEqual(std, 0.)

    
if __name__ == '__main__':
    unittest.main()
 
