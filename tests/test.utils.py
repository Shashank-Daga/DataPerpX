import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.utils import (
    load_config, save_config, load_data, save_data,
    detect_column_types, calculate_memory_usage, 
    get_timestamp, TemplateLoader
)

class TestConfigOperations(unittest.TestCase):
    
    def setUp(self):
        self.test_config = {
            'preprocessing': {
                'missing_strategy': 'median',
                'scale_features': True
            },
            'estimation': {
                'test_size': 0.2
            }
        }
    
    def test_save_and_load_yaml(self):
        yaml_path = Path('test_config.yaml')
        
        try:
            save_config(self.test_config, str(yaml_path))
            loaded_config = load_config(str(yaml_path))
            
            self.assertEqual(loaded_config, self.test_config)
            
        finally:
            if yaml_path.exists():
                yaml_path.unlink()
    
    def test_save_and_load_json(self):
        json_path = Path('test_config.json')
        
        try:
            save_config(self.test_config, str(json_path))
            loaded_config = load_config(str(json_path))
            
            self.assertEqual(loaded_config, self.test_config)
            
        finally:
            if json_path.exists():
                json_path.unlink()
    
    def test_load_nonexistent_config(self):
        with self.assertRaises(FileNotFoundError):
            load_config('nonexistent_config.yaml')

class TestDataOperations(unittest.TestCase):
    
    def setUp(self):
        self.test_df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': ['A', 'B', 'C', 'D', 'E'],
            'col3': [1.1, 2.2, 3.3, 4.4, 5.5]
        })
    
    def test_save_and_load_csv(self):
        csv_path = Path('test_data.csv')
        
        try:
            save_data(self.test_df, str(csv_path))
            loaded_df = load_data(str(csv_path))
            
            pd.testing.assert_frame_equal(loaded_df, self.test_df)
            
        finally:
            if csv_path.exists():
                csv_path.unlink()
    
    def test_save_and_load_excel(self):
        xlsx_path = Path('test_data.xlsx')
        
        try:
            save_data(self.test_df, str(xlsx_path))
            loaded_df = load_data(str(xlsx_path))
            
            pd.testing.assert_frame_equal(loaded_df, self.test_df)
            
        finally:
            if xlsx_path.exists():
                xlsx_path.unlink()
    
    def test_load_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            load_data('nonexistent_file.csv')
    
    def test_unsupported_format(self):
        with self.assertRaises(ValueError):
            load_data('data.txt')

class TestColumnTypeDetection(unittest.TestCase):
    
    def test_detect_mixed_types(self):
        df = pd.DataFrame({
            'numeric_int': [1, 2, 3, 4, 5],
            'numeric_float': [1.1, 2.2, 3.3, 4.4, 5.5],
            'categorical': ['A', 'B', 'C', 'D', 'E'],
            'datetime': pd.date_range('2024-01-01', periods=5)
        })
        
        column_types = detect_column_types(df)
        
        self.assertIn('numeric', column_types)
        self.assertIn('categorical', column_types)
        self.assertIn('datetime', column_types)
        
        self.assertIn('numeric_int', column_types['numeric'])
        self.assertIn('numeric_float', column_types['numeric'])
        self.assertIn('categorical', column_types['categorical'])
        self.assertIn('datetime', column_types['datetime'])
    
    def test_detect_only_numeric(self):
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [1.1, 2.2, 3.3]
        })
        
        column_types = detect_column_types(df)
        
        self.assertEqual(len(column_types['numeric']), 2)
        self.assertEqual(len(column_types['categorical']), 0)
        self.assertEqual(len(column_types['datetime']), 0)
    
    def test_detect_only_categorical(self):
        df = pd.DataFrame({
            'col1': ['A', 'B', 'C'],
            'col2': ['X', 'Y', 'Z']
        })
        
        column_types = detect_column_types(df)
        
        self.assertEqual(len(column_types['numeric']), 0)
        self.assertEqual(len(column_types['categorical']), 2)

class TestMemoryCalculation(unittest.TestCase):
    
    def test_memory_calculation(self):
        df = pd.DataFrame({
            'col1': range(1000),
            'col2': ['text'] * 1000,
            'col3': np.random.randn(1000)
        })
        
        memory = calculate_memory_usage(df)
        
        self.assertIn('bytes', memory)
        self.assertIn('kb', memory)
        self.assertIn('mb', memory)
        self.assertIn('gb', memory)
        
        self.assertGreater(memory['bytes'], 0)
        self.assertGreater(memory['kb'], 0)
        self.assertGreater(memory['mb'], 0)
        
        self.assertEqual(memory['bytes'], memory['kb'] * 1024)
        self.assertAlmostEqual(memory['kb'], memory['mb'] * 1024, places=2)

class TestTimestamp(unittest.TestCase):
    
    def test_timestamp_format(self):
        timestamp = get_timestamp()
        
        self.assertEqual(len(timestamp), 15)
        self.assertTrue(timestamp.isdigit() or '_' in timestamp)
        
        parts = timestamp.split('_')
        self.assertEqual(len(parts), 2)
        self.assertEqual(len(parts[0]), 8)
        self.assertEqual(len(parts[1]), 6)

class TestTemplateLoader(unittest.TestCase):
    
    def setUp(self):
        self.template_dir = Path('test_templates')
        self.template_dir.mkdir(exist_ok=True)
        
        self.template_content = "Hello {name}, your score is {score}."
        template_file = self.template_dir / 'greeting.txt'
        
        with open(template_file, 'w') as f:
            f.write(self.template_content)
        
        self.loader = TemplateLoader(str(self.template_dir))
    
    def tearDown(self):
        import shutil
        if self.template_dir.exists():
            shutil.rmtree(self.template_dir)
    
    def test_load_template(self):
        template = self.loader.load('greeting')
        self.assertEqual(template, self.template_content)
    
    def test_render_template(self):
        rendered = self.loader.render('greeting', name='Alice', score=95)
        self.assertEqual(rendered, "Hello Alice, your score is 95.")
    
    def test_load_nonexistent_template(self):
        with self.assertRaises(FileNotFoundError):
            self.loader.load('nonexistent')

if __name__ == '__main__':
    unittest.main()