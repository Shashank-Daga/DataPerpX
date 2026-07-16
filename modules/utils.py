import logging
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Union
import pandas as pd

def setup_logging(log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger('DataPrepX')
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config

def save_config(config: Dict[str, Any], output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        if output_path.suffix in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        elif output_path.suffix == '.json':
            json.dump(config, f, indent=2)

def load_data(file_path: str) -> pd.DataFrame:
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    if file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
    elif file_path.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Supported formats: .csv, .xlsx, .xls")
    
    return df

def save_data(df: pd.DataFrame, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.csv':
        df.to_csv(output_path, index=False)
    elif output_path.suffix in ['.xlsx', '.xls']:
        df.to_excel(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")

def get_timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def detect_column_types(df: pd.DataFrame) -> Dict[str, list]:
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }

def calculate_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    memory_bytes = df.memory_usage(deep=True).sum()
    return {
        'bytes': memory_bytes,
        'kb': memory_bytes / 1024,
        'mb': memory_bytes / (1024 ** 2),
        'gb': memory_bytes / (1024 ** 3)
    }

class TemplateLoader:
    def __init__(self, template_dir: str = 'configs/templates'):
        self.template_dir = Path(template_dir)
    
    def load(self, template_name: str) -> str:
        template_path = self.template_dir / f"{template_name}.txt"
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        with open(template_path, 'r') as f:
            return f.read()
    
    def render(self, template_name: str, **kwargs) -> str:
        template = self.load(template_name)
        return template.format(**kwargs)