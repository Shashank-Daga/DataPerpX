import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from modules.utils import load_data, detect_column_types, setup_logging

logger = setup_logging()


class DataPreprocessor:
    """
    Fits all preprocessing statistics (imputation values, outlier bounds,
    encoding vocabularies, scaler parameters) on ONE dataset via `fit()`,
    then applies them unchanged via `transform()`.

    This split matters: if you fit on train+test combined (or on the whole
    dataset before splitting), information about the test set leaks into
    the preprocessing statistics, which quietly inflates evaluation scores.

    Usage for a train/test workflow (no leakage):
        preprocessor = DataPreprocessor(config)
        df_train_clean, metadata = preprocessor.fit_transform(df_train, target_column='y')
        df_test_clean = preprocessor.transform(df_test)

    Usage for one-off exploration (no train/test split needed):
        preprocessor = DataPreprocessor(config)
        df_clean, metadata = preprocessor.process('data.csv', target_column='y')

    `target_column`, when given, is excluded from every transformation
    (imputation, outlier capping, scaling, feature engineering) and
    reattached unmodified. Without this, engineered features like
    `{target}_squared` or `{target}_x_{other}` would leak the label
    directly into the feature set.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metadata: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}

        self.target_column: Optional[str] = None
        self._fitted = False

        self._high_missing_cols: List[str] = []
        self._numeric_impute_cols: List[str] = []
        self._categorical_impute_cols: List[str] = []
        self._missing_strategy: str = 'median'

        self._outlier_method: str = 'iqr'
        self._outlier_bounds: Dict[str, Dict[str, float]] = {}

        self._categorical_cols: List[str] = []
        self._onehot_categories: Dict[str, List[Any]] = {}
        self._label_encoded_cols: List[str] = []

        self._scaler_type: str = 'standard'
        self._scale_cols: List[str] = []

        self._fe_numeric_cols = None
        self._fe_pairs = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, file_path: str, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Convenience wrapper: load a file and fit_transform it in one shot.
        For train/test workflows, prefer fit_transform(train) + transform(test)
        instead, to avoid fitting statistics on data the model will be tested on."""
        df = load_data(file_path)
        return self.fit_transform(df, target_column)

    def fit_transform(self, df: pd.DataFrame, target_column: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        self.fit(df, target_column)
        return self.transform(df, _is_fit_data=True), self.metadata

    def fit(self, df: pd.DataFrame, target_column: Optional[str] = None) -> 'DataPreprocessor':
        self.target_column = target_column if target_column in df.columns else None

        self.metadata['original_shape'] = df.shape
        self.metadata['original_columns'] = df.columns.tolist()
        self.metadata['original_dtypes'] = df.dtypes.to_dict()
        logger.info(f"Original data shape: {df.shape}")

        features, _ = self._split_target(df)

        features = self._fit_missing_values(features)
        features = self._fit_outliers(features)
        features = self._fit_categorical(features)
        features = self._fit_scaling(features)
        self._fit_feature_engineering()

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, _is_fit_data: bool = False) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("DataPreprocessor.transform() called before fit(). Call fit() or fit_transform() first.")

        features, target = self._split_target(df)

        if _is_fit_data:
            # Duplicate removal is a training-set cleaning decision; it is
            # NOT applied to new/test data (dropping "duplicate" test rows
            # would arbitrarily shrink the evaluation set).
            features, target = self._remove_duplicates(features, target)

        features = self._apply_missing_values(features)
        features = self._apply_outliers(features, allow_row_drop=_is_fit_data)

        if target is not None and _is_fit_data:
            target = target.reindex(features.index)

        features = self._apply_categorical(features)
        features = self._apply_scaling(features)
        features = self._apply_feature_engineering(features)

        result = features
        if target is not None:
            result = features.copy()
            result[self.target_column] = target

        if _is_fit_data:
            self.metadata['final_shape'] = result.shape
            self.metadata['final_columns'] = result.columns.tolist()
            self.metadata['column_types'] = detect_column_types(result)
            logger.info(f"Final data shape: {result.shape}")

        return result

    # ------------------------------------------------------------------
    # Target isolation
    # ------------------------------------------------------------------

    def _split_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        if self.target_column and self.target_column in df.columns:
            target = df[self.target_column].copy()
            features = df.drop(columns=[self.target_column]).copy()

            missing_target = int(target.isna().sum())
            if missing_target > 0:
                valid_idx = target.dropna().index
                logger.info(f"Dropping {missing_target} rows with missing target '{self.target_column}'")
                features = features.loc[valid_idx]
                target = target.loc[valid_idx]

            return features, target
        return df.copy(), None

    def _remove_duplicates(self, features: pd.DataFrame, target: Optional[pd.Series]
                            ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        combined = features if target is None else features.assign(**{self.target_column: target})
        initial_rows = len(combined)
        combined = combined.drop_duplicates()
        duplicates_removed = initial_rows - len(combined)

        self.metadata['duplicates_removed'] = duplicates_removed
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")

        if target is None:
            return combined, None
        return combined.drop(columns=[self.target_column]), combined[self.target_column]

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------

    def _fit_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_summary = df.isnull().sum()
        self.metadata['missing_values'] = missing_summary[missing_summary > 0].to_dict()

        strategy = self.config.get('missing_strategy', 'auto')
        threshold = self.config.get('missing_threshold', 0.5)

        if strategy == 'auto':
            strategy = 'knn' if len(df) <= 2000 else 'median'
            logger.info(f"Auto selected missing value strategy: {strategy}")
        self._missing_strategy = strategy

        self._high_missing_cols = missing_summary[missing_summary / len(df) > threshold].index.tolist()
        if self._high_missing_cols:
            logger.info(f"Dropping {len(self._high_missing_cols)} columns with >{threshold*100}% missing data")
            df = df.drop(columns=self._high_missing_cols)

        if missing_summary.sum() == 0:
            self._numeric_impute_cols = []
            self._categorical_impute_cols = []
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self._numeric_impute_cols = numeric_cols
        self._categorical_impute_cols = categorical_cols

        if strategy == 'knn' and numeric_cols:
            imputer = KNNImputer(n_neighbors=5)
            imputer.fit(df[numeric_cols])
            self.imputers['numeric'] = imputer
        elif numeric_cols:
            imputer = SimpleImputer(strategy='median')
            imputer.fit(df[numeric_cols])
            self.imputers['numeric'] = imputer

        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            cat_imputer.fit(df[categorical_cols])
            self.imputers['categorical'] = cat_imputer

        return df

    def _apply_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [c for c in self._high_missing_cols if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        if 'numeric' in self.imputers and self._numeric_impute_cols:
            cols = [c for c in self._numeric_impute_cols if c in df.columns]
            if cols:
                df[cols] = self.imputers['numeric'].transform(df[cols])

        if 'categorical' in self.imputers and self._categorical_impute_cols:
            cols = [c for c in self._categorical_impute_cols if c in df.columns]
            if cols:
                df[cols] = self.imputers['categorical'].transform(df[cols])

        return df

    # ------------------------------------------------------------------
    # Outliers
    # ------------------------------------------------------------------

    def _fit_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        method = self.config.get('outlier_method', 'iqr')
        if method == 'auto':
            method = 'iqr'
            logger.info("Auto selected outlier method: iqr")
        self._outlier_method = method

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_count = {}
        self._outlier_bounds = {}

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            else:  # zscore
                mean, std = df[col].mean(), df[col].std()
                lower, upper = mean - 3 * std, mean + 3 * std

            outliers_count[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
            self._outlier_bounds[col] = {'lower': lower, 'upper': upper}

        self.metadata['outliers'] = outliers_count
        return df

    def _apply_outliers(self, df: pd.DataFrame, allow_row_drop: bool) -> pd.DataFrame:
        if not self.config.get('cap_outliers', True):
            return df

        for col, bounds in self._outlier_bounds.items():
            if col not in df.columns:
                continue
            lower, upper = bounds['lower'], bounds['upper']

            if self._outlier_method == 'zscore' and allow_row_drop:
                # Row-dropping is only safe on the data being fit (training
                # set); on new/test data we clip instead, so row count and
                # evaluation-set membership stay stable.
                df = df[(df[col] >= lower) & (df[col] <= upper)]
            else:
                df[col] = df[col].clip(lower=lower, upper=upper)

        return df

    # ------------------------------------------------------------------
    # Categorical encoding
    # ------------------------------------------------------------------

    def _fit_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self._categorical_cols = categorical_cols
        self._onehot_categories = {}
        self._label_encoded_cols = []
        encoding_map = {}

        for col in categorical_cols:
            unique_values = df[col].nunique()
            if unique_values <= self.config.get('onehot_threshold', 10):
                categories = sorted(df[col].dropna().unique().tolist())
                self._onehot_categories[col] = categories
                encoding_map[col] = 'onehot'
            else:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.encoders[col] = le
                self._label_encoded_cols.append(col)
                encoding_map[col] = 'label'

        self.metadata['encoding_map'] = encoding_map
        return df

    def _apply_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, categories in self._onehot_categories.items():
            if col not in df.columns:
                continue
            cat_series = pd.Categorical(df[col], categories=categories)
            dummies = pd.get_dummies(cat_series, prefix=col, drop_first=True)
            dummies.index = df.index
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])

        for col in self._label_encoded_cols:
            if col not in df.columns:
                continue
            le = self.encoders[col]
            known = set(le.classes_)
            # Unseen categories fall back to the first known class rather
            # than crashing LabelEncoder.transform().
            safe_values = df[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(safe_values)

        return df

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def _fit_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.get('scale_features', True):
            self._scale_cols = []
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            self._scale_cols = []
            return df

        scaler_type = self.config.get('scaler', 'standard')
        if scaler_type == 'auto':
            max_abs_skew = df[numeric_cols].skew().abs().max()
            scaler_type = 'robust' if max_abs_skew > 1.0 else 'standard'
            logger.info(f"Auto selected scaler: {scaler_type} (max abs skew = {max_abs_skew:.2f})")
        self._scaler_type = scaler_type
        return df

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.get('scale_features', True):
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return df

        # The scaler itself is fit only once: the first time this runs,
        # which is during fit_transform's pass over the training data
        # (encoding has already happened by this point, so the onehot
        # dummy columns are included in what gets scaled).
        if 'features' not in self.scalers:
            scaler = RobustScaler() if self._scaler_type == 'robust' else StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            self.scalers['features'] = scaler
            self._scale_cols = numeric_cols
        else:
            missing = [c for c in self._scale_cols if c not in df.columns]
            for c in missing:
                df[c] = 0  # category seen at fit time, absent from this batch
            df[self._scale_cols] = self.scalers['features'].transform(df[self._scale_cols])

        return df

    # ------------------------------------------------------------------
    # Feature engineering (deterministic; only the column/pair choice
    # needs to be "fit" once, so train and test get identical features)
    # ------------------------------------------------------------------

    def _fit_feature_engineering(self) -> None:
        self._fe_numeric_cols = None
        self._fe_pairs = None

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.config.get('feature_engineering', True):
            return df

        if self._fe_numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self._fe_pairs = [
                (numeric_cols[i], numeric_cols[i + 1])
                for i in range(min(3, len(numeric_cols) - 1))
            ] if len(numeric_cols) >= 2 else []
            self._fe_numeric_cols = numeric_cols[:5]

        for col1, col2 in self._fe_pairs:
            if col1 in df.columns and col2 in df.columns:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)

        for col in self._fe_numeric_cols:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_log'] = np.log1p(np.abs(df[col]))

        return df
