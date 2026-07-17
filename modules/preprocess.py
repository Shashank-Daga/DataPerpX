import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from modules.utils import load_data, detect_column_types, setup_logging

logger = setup_logging()


class DataPreprocessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metadata = {}
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def process(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        df = load_data(file_path)
        
        self.metadata['original_shape'] = df.shape
        self.metadata['original_columns'] = df.columns.tolist()
        self.metadata['original_dtypes'] = df.dtypes.to_dict()
        
        logger.info(f"Original data shape: {df.shape}")
        
        df = self._handle_missing_values(df)
        df = self._remove_duplicates(df)
        df = self._handle_outliers(df)
        df = self._encode_categorical(df)
        df = self._scale_features(df)
        df = self._feature_engineering(df)
        
        self.metadata['final_shape'] = df.shape
        self.metadata['final_columns'] = df.columns.tolist()
        self.metadata['column_types'] = detect_column_types(df)
        
        logger.info(f"Final data shape: {df.shape}")
        
        return df, self.metadata
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_summary = df.isnull().sum()
        self.metadata['missing_values'] = missing_summary[missing_summary > 0].to_dict()
        
        if missing_summary.sum() == 0:
            return df
        
        strategy = self.config.get('missing_strategy', 'auto')
        threshold = self.config.get('missing_threshold', 0.5)

        if strategy == 'auto':
            # KNN imputation is more accurate but O(n^2)-ish;
            # only worth it on smaller datasets. Fall back to median/most-frequent otherwise.
            strategy = 'knn' if len(df) <= 2000 else 'median'
            logger.info(f"Auto selected missing value strategy: {strategy}")
        
        high_missing_cols = missing_summary[missing_summary / len(df) > threshold].index.tolist()
        if high_missing_cols:
            logger.info(f"Dropping {len(high_missing_cols)} columns with >{threshold*100}% missing data")
            df = df.drop(columns=high_missing_cols)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if strategy == 'knn' and len(numeric_cols) > 0:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
            self.imputers['numeric'] = imputer
        else:
            if len(numeric_cols) > 0:
                num_imputer = SimpleImputer(strategy='median')
                df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
                self.imputers['numeric'] = num_imputer
        
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            self.imputers['categorical'] = cat_imputer
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_rows = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df)
        
        self.metadata['duplicates_removed'] = duplicates_removed
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        method = self.config.get('outlier_method', 'iqr')

        if method == 'auto':
            # IQR is the more conservative, distribution-agnostic default
            # (z-score assumes near-normal data and drops rows outright).
            method = 'iqr'
            logger.info("Auto selected outlier method: iqr")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        outliers_count = {}
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                outliers = (df[col] < lower) | (df[col] > upper)
                outliers_count[col] = outliers.sum()
                
                if self.config.get('cap_outliers', True):
                    df[col] = df[col].clip(lower=lower, upper=upper)
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = z_scores > 3
                outliers_count[col] = outliers.sum()
                
                if self.config.get('cap_outliers', True):
                    df = df[z_scores <= 3]
        
        self.metadata['outliers'] = outliers_count
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        encoding_map = {}
        
        for col in categorical_cols:
            unique_values = df[col].nunique()
            
            if unique_values <= self.config.get('onehot_threshold', 10):
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
                encoding_map[col] = 'onehot'
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                encoding_map[col] = 'label'
        
        self.metadata['encoding_map'] = encoding_map
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.get('scale_features', True):
            return df
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return df
        
        scaler_type = self.config.get('scaler', 'standard')

        if scaler_type == 'auto':
            # Robust scaling (median/IQR) is preferable when features are
            # heavily skewed / outlier-prone; standard scaling otherwise.
            max_abs_skew = df[numeric_cols].skew().abs().max()
            scaler_type = 'robust' if max_abs_skew > 1.0 else 'standard'
            logger.info(f"Auto selected scaler: {scaler_type} (max abs skew = {max_abs_skew:.2f})")
        
        if scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.scalers['features'] = scaler
        
        return df
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.get('feature_engineering', True):
            return df
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            for i in range(min(3, len(numeric_cols) - 1)):
                col1 = numeric_cols[i]
                col2 = numeric_cols[i + 1]
                
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
        
        for col in numeric_cols[:5]:
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_log'] = np.log1p(np.abs(df[col]))
        
        return df
