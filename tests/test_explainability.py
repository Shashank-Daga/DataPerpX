import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.explainability import ExplainabilityAnalyzer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class TestExplainabilityAnalyzer(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            'enable_shap': True,
            'enable_lime': True,
            'num_samples': 50,
            'num_lime_samples': 3
        }
        
        self.analyzer = ExplainabilityAnalyzer(self.config)
        
        np.random.seed(42)
        
        self.X_class = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100) * 2,
            'feature3': np.random.randn(100) * 0.5
        })
        self.y_class = pd.Series(np.random.randint(0, 2, 100))
        
        self.X_reg = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100) * 2,
            'feature3': np.random.randn(100) * 0.5
        })
        self.y_reg = pd.Series(np.random.randn(100) * 100 + 500)
        
        self.clf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.clf_model.fit(self.X_class, self.y_class)
        
        self.reg_model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.reg_model.fit(self.X_reg, self.y_reg)
    
    def test_shap_analysis_classification(self):
        results = self.analyzer.analyze(
            self.clf_model, 
            self.X_class, 
            self.y_class, 
            'classification'
        )
        
        self.assertIsNotNone(results)
        self.assertIn('shap_analysis', results)
        
        if results['shap_analysis']:
            self.assertIn('feature_importance', results['shap_analysis'])
            self.assertIn('explainer_type', results['shap_analysis'])
    
    def test_shap_analysis_regression(self):
        results = self.analyzer.analyze(
            self.reg_model, 
            self.X_reg, 
            self.y_reg, 
            'regression'
        )
        
        self.assertIsNotNone(results)
        self.assertIn('shap_analysis', results)
        
        if results['shap_analysis']:
            self.assertIn('feature_importance', results['shap_analysis'])
    
    def test_lime_analysis_classification(self):
        results = self.analyzer.analyze(
            self.clf_model, 
            self.X_class, 
            self.y_class, 
            'classification'
        )
        
        self.assertIn('lime_analysis', results)
        
        if results['lime_analysis']:
            self.assertIn('explanations', results['lime_analysis'])
            self.assertGreater(len(results['lime_analysis']['explanations']), 0)
    
    def test_lime_analysis_regression(self):
        results = self.analyzer.analyze(
            self.reg_model, 
            self.X_reg, 
            self.y_reg, 
            'regression'
        )
        
        self.assertIn('lime_analysis', results)
        
        if results['lime_analysis']:
            self.assertIn('explanations', results['lime_analysis'])
    
    def test_global_importance_extraction(self):
        results = self.analyzer.analyze(
            self.clf_model, 
            self.X_class, 
            self.y_class, 
            'classification'
        )
        
        if results['shap_analysis']:
            global_importance = results['global_importance']
            
            self.assertIsInstance(global_importance, dict)
            self.assertGreater(len(global_importance), 0)
            
            for feature, importance in global_importance.items():
                self.assertIsInstance(importance, (int, float))
                self.assertGreaterEqual(importance, 0)
    
    def test_local_explanations_structure(self):
        results = self.analyzer.analyze(
            self.clf_model, 
            self.X_class, 
            self.y_class, 
            'classification'
        )
        
        if results['lime_analysis']:
            explanations = results['lime_analysis']['explanations']
            
            for exp in explanations:
                self.assertIn('instance_index', exp)
                self.assertIn('prediction', exp)
                self.assertIn('actual', exp)
                self.assertIn('feature_contributions', exp)
                
                self.assertIsInstance(exp['feature_contributions'], dict)
    
    def test_plot_generation(self):
        self.analyzer.analyze(
            self.clf_model, 
            self.X_class, 
            self.y_class, 
            'classification'
        )
        
        output_dir = Path('test_explainability_output')
        output_dir.mkdir(exist_ok=True)
        
        try:
            plot_paths = self.analyzer.generate_plots(output_dir)
            
            self.assertIsInstance(plot_paths, dict)
            
        finally:
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir)
    
    def test_feature_explanations(self):
        self.analyzer.analyze(
            self.clf_model, 
            self.X_class, 
            self.y_class, 
            'classification'
        )
        
        explanations = self.analyzer.get_feature_explanations(top_n=3)
        
        if explanations:
            self.assertIsInstance(explanations, dict)
            self.assertLessEqual(len(explanations), 3)
            
            for feature, explanation in explanations.items():
                self.assertIsInstance(explanation, str)
                self.assertGreater(len(explanation), 0)

class TestExplainabilityEdgeCases(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            'enable_shap': True,
            'enable_lime': True
        }
        
        self.analyzer = ExplainabilityAnalyzer(self.config)
    
    def test_small_dataset(self):
        X_small = pd.DataFrame({
            'f1': np.random.randn(30),
            'f2': np.random.randn(30)
        })
        y_small = pd.Series(np.random.randint(0, 2, 30))
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_small, y_small)
        
        results = self.analyzer.analyze(model, X_small, y_small, 'classification')
        
        self.assertIsNotNone(results)
    
    def test_single_feature(self):
        X_single = pd.DataFrame({
            'feature1': np.random.randn(100)
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_single, y)
        
        results = self.analyzer.analyze(model, X_single, y, 'classification')
        
        self.assertIsNotNone(results)

if __name__ == '__main__':
    unittest.main()