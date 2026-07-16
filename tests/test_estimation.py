import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.estimation import ModelEstimator

class TestModelEstimator(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 3
        }
        
        self.estimator = ModelEstimator(self.config)
        
        np.random.seed(42)
        
        self.classification_df = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200) * 2 + 5,
            'feature3': np.random.randn(200) * 0.5,
            'target': np.random.randint(0, 3, 200)
        })
        
        self.regression_df = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200) * 2,
            'feature3': np.random.randn(200) * 0.5,
            'target': np.random.randn(200) * 100 + 500
        })
    
    def test_task_detection_classification(self):
        task = self.estimator._detect_task_type(self.classification_df['target'])
        self.assertEqual(task, 'classification')
    
    def test_task_detection_regression(self):
        task = self.estimator._detect_task_type(self.regression_df['target'])
        self.assertEqual(task, 'regression')
    
    def test_classification_pipeline(self):
        results = self.estimator.fit_and_evaluate(
            self.classification_df, 
            'target', 
            'classification'
        )
        
        self.assertIn('task_type', results)
        self.assertEqual(results['task_type'], 'classification')
        
        self.assertIn('best_model', results)
        self.assertIsNotNone(results['best_model'])
        
        self.assertIn('best_score', results)
        self.assertGreater(results['best_score'], 0)
        
        self.assertIn('models', results)
        self.assertGreater(len(results['models']), 0)
        
        for model_name, metrics in results['models'].items():
            self.assertIn('accuracy', metrics)
            self.assertIn('cv_mean', metrics)
    
    def test_regression_pipeline(self):
        results = self.estimator.fit_and_evaluate(
            self.regression_df, 
            'target', 
            'regression'
        )
        
        self.assertIn('task_type', results)
        self.assertEqual(results['task_type'], 'regression')
        
        self.assertIn('best_model', results)
        self.assertIsNotNone(results['best_model'])
        
        self.assertIn('models', results)
        
        for model_name, metrics in results['models'].items():
            self.assertIn('r2_score', metrics)
            self.assertIn('mse', metrics)
            self.assertIn('mae', metrics)
    
    def test_auto_task_detection(self):
        results = self.estimator.fit_and_evaluate(
            self.classification_df, 
            'target', 
            'auto'
        )
        
        self.assertEqual(results['task_type'], 'classification')
    
    def test_feature_importance_extraction(self):
        results = self.estimator.fit_and_evaluate(
            self.classification_df, 
            'target', 
            'classification'
        )
        
        self.assertIn('feature_importance', results)
        
        if results['feature_importance']:
            self.assertIsInstance(results['feature_importance'], dict)
            self.assertGreater(len(results['feature_importance']), 0)
    
    def test_prediction(self):
        self.estimator.fit_and_evaluate(
            self.classification_df, 
            'target', 
            'classification'
        )
        
        X_new = self.classification_df.drop(columns=['target']).iloc[:5]
        predictions = self.estimator.predict(X_new)
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(pred in [0, 1, 2] for pred in predictions))

class TestModelEstimatorEdgeCases(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            'test_size': 0.2,
            'random_state': 42,
            'cv_folds': 3
        }
        
        self.estimator = ModelEstimator(self.config)
    
    def test_binary_classification(self):
        df = pd.DataFrame({
            'f1': np.random.randn(100),
            'f2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        results = self.estimator.fit_and_evaluate(df, 'target', 'classification')
        
        self.assertEqual(results['task_type'], 'classification')
        self.assertIn('best_model', results)
    
    def test_multiclass_classification(self):
        df = pd.DataFrame({
            'f1': np.random.randn(150),
            'f2': np.random.randn(150),
            'target': np.random.randint(0, 5, 150)
        })
        
        results = self.estimator.fit_and_evaluate(df, 'target', 'classification')
        
        self.assertEqual(results['task_type'], 'classification')
    
    def test_small_dataset(self):
        df = pd.DataFrame({
            'f1': np.random.randn(30),
            'f2': np.random.randn(30),
            'target': np.random.randn(30)
        })
        
        results = self.estimator.fit_and_evaluate(df, 'target', 'regression')
        
        self.assertIn('best_model', results)
    
    def test_invalid_target_column(self):
        df = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50),
            'target': np.random.randn(50)
        })
        
        with self.assertRaises(ValueError):
            self.estimator.fit_and_evaluate(df, 'nonexistent_column', 'regression')
    
    def test_predict_without_training(self):
        df = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50)
        })
        
        with self.assertRaises(ValueError):
            self.estimator.predict(df)

class TestMetricsCalculation(unittest.TestCase):
    
    def setUp(self):
        self.config = {'test_size': 0.2, 'random_state': 42}
        self.estimator = ModelEstimator(self.config)
    
    def test_classification_metrics(self):
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 1])
        
        metrics = self.estimator._calculate_classification_metrics(y_true, y_pred)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertTrue(0 <= metrics['precision'] <= 1)
        self.assertTrue(0 <= metrics['recall'] <= 1)
        self.assertTrue(0 <= metrics['f1_score'] <= 1)
    
    def test_regression_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.2])
        
        metrics = self.estimator._calculate_regression_metrics(y_true, y_pred)
        
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('r2_score', metrics)
        
        self.assertGreater(metrics['mse'], 0)
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mae'], 0)

if __name__ == '__main__':
    unittest.main()