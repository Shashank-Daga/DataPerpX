import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.preprocess import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            'missing_strategy': 'median',
            'missing_threshold': 0.5,
            'outlier_method': 'iqr',
            'cap_outliers': True,
            'onehot_threshold': 10,
            'scale_features': True,
            'scaler': 'standard',
            'feature_engineering': False
        }
        
        self.preprocessor = DataPreprocessor(self.config)
        
        np.random.seed(42)
        self.sample_df = pd.DataFrame({
            'numeric1': np.random.randn(100),
            'numeric2': np.random.randn(100) * 10 + 50,
            'category1': np.random.choice(['A', 'B', 'C'], 100),
            'category2': np.random.choice(['X', 'Y'], 100),
            'target': np.random.randint(0, 2, 100)
        })
        
        missing_indices = np.random.choice(100, 10, replace=False)
        self.sample_df.loc[missing_indices, 'numeric1'] = np.nan
        
        duplicate_row = self.sample_df.iloc[0:1]
        self.sample_df = pd.concat([self.sample_df, duplicate_row], ignore_index=True)
    
    def test_missing_value_handling(self):
        df_clean = self.preprocessor._handle_missing_values(self.sample_df.copy())
        
        self.assertEqual(df_clean.isnull().sum().sum(), 0)
        self.assertIn('missing_values', self.preprocessor.metadata)
    
    def test_duplicate_removal(self):
        df_clean = self.preprocessor._remove_duplicates(self.sample_df.copy())
        
        self.assertEqual(len(df_clean), len(self.sample_df) - 1)
        self.assertIn('duplicates_removed', self.preprocessor.metadata)
        self.assertEqual(self.preprocessor.metadata['duplicates_removed'], 1)
    
    def test_outlier_handling(self):
        df_test = self.sample_df.copy()
        df_test.loc[0, 'numeric1'] = 1000
        
        df_clean = self.preprocessor._handle_outliers(df_test)
        
        self.assertIn('outliers', self.preprocessor.metadata)
        self.assertLess(df_clean['numeric1'].max(), 1000)
    
    def test_categorical_encoding(self):
        df_clean = self.preprocessor._encode_categorical(self.sample_df.copy())
        
        self.assertIn('encoding_map', self.preprocessor.metadata)
        
        self.assertNotIn('category1', df_clean.columns)
        self.assertTrue(any('category1_' in col for col in df_clean.columns))
    
    def test_feature_scaling(self):
        df_test = self.sample_df[['numeric1', 'numeric2', 'target']].copy().dropna()
        df_clean = self.preprocessor._scale_features(df_test)
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            self.assertAlmostEqual(df_clean[col].mean(), 0, places=1)
            self.assertAlmostEqual(df_clean[col].std(), 1, places=1)
    
    def test_full_pipeline(self):
        temp_file = Path('temp_test_data.csv')
        self.sample_df.to_csv(temp_file, index=False)
        
        try:
            df_clean, metadata = self.preprocessor.process(str(temp_file))
            
            self.assertIsInstance(df_clean, pd.DataFrame)
            self.assertIsInstance(metadata, dict)
            
            self.assertIn('original_shape', metadata)
            self.assertIn('final_shape', metadata)
            self.assertIn('column_types', metadata)
            
            self.assertEqual(df_clean.isnull().sum().sum(), 0)
            
        finally:
            if temp_file.exists():
                temp_file.unlink()

class TestDataPreprocessorEdgeCases(unittest.TestCase):
    
    def setUp(self):
        self.config = {
            'missing_strategy': 'median',
            'missing_threshold': 0.5,
            'outlier_method': 'iqr',
            'cap_outliers': True,
            'onehot_threshold': 10,
            'scale_features': True,
            'scaler': 'standard',
            'feature_engineering': False
        }
        
        self.preprocessor = DataPreprocessor(self.config)
    
    def test_all_missing_column(self):
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [np.nan] * 5,
            'col3': [10, 20, 30, 40, 50]
        })
        
        df_clean = self.preprocessor._handle_missing_values(df)
        
        self.assertNotIn('col2', df_clean.columns)
    
    def test_single_value_column(self):
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A'] * 5,
            'col3': [10, 20, 30, 40, 50]
        })
        
        df_clean = self.preprocessor._encode_categorical(df)
        
        self.assertIsInstance(df_clean, pd.DataFrame)
    
    def test_no_numeric_columns(self):
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C', 'D', 'E'],
            'cat2': ['X', 'Y', 'Z', 'X', 'Y']
        })
        
        df_clean = self.preprocessor._scale_features(df)
        
        self.assertIsInstance(df_clean, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()