import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, \
    GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             mean_squared_error, mean_absolute_error, r2_score,
                             classification_report, confusion_matrix)
from modules.utils import setup_logging

logger = setup_logging()


class ModelEstimator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_score = -np.inf
        self.task_type = None
        self.n_jobs = -1

    def split_raw(self, df: pd.DataFrame, target_col: str, task: str = 'auto'):
        """Split RAW (unprocessed) data into train/test dataframes, before any
        preprocessing happens. Use this + DataPreprocessor.fit_transform(train)
        + DataPreprocessor.transform(test) + fit_and_evaluate_presplit() for a
        leak-free pipeline. Returns (df_train, df_test, task_type)."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")

        y = df[target_col]
        task_type = self._detect_task_type(y) if task == 'auto' else task
        can_stratify = self._can_stratify(y, task_type)

        try:
            df_train, df_test = train_test_split(
                df,
                test_size=self.config.get('test_size', 0.2),
                random_state=self.config.get('random_state', 42),
                stratify=y if can_stratify else None
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}. Using random split instead.")
            df_train, df_test = train_test_split(
                df,
                test_size=self.config.get('test_size', 0.2),
                random_state=self.config.get('random_state', 42),
                stratify=None
            )

        return df_train, df_test, task_type

    def fit_and_evaluate(self, df: pd.DataFrame, target_col: str, task: str = 'auto') -> Dict[str, Any]:
        """Single-dataframe entry point: splits df into train/test internally.

        NOTE: if df was already preprocessed (imputed/scaled/encoded) as a
        whole before this split, statistics from the test portion have
        leaked into those preprocessing steps. For a leak-free workflow,
        split your raw data first and use fit_and_evaluate_presplit()
        with a DataPreprocessor fit only on the training portion."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        task_type = self._detect_task_type(y) if task == 'auto' else task
        logger.info(f"Task type: {task_type}" + (" (auto-detected)" if task == 'auto' else ""))

        can_stratify = self._can_stratify(y, task_type)

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.get('test_size', 0.2),
                random_state=self.config.get('random_state', 42),
                stratify=y if can_stratify else None
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed: {e}. Using random split instead.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config.get('test_size', 0.2),
                random_state=self.config.get('random_state', 42),
                stratify=None
            )

        return self._fit_and_evaluate_core(X_train, X_test, y_train, y_test, task_type, target_col)

    def fit_and_evaluate_presplit(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                                   target_col: str, task: str = 'auto') -> Dict[str, Any]:
        """Leak-free entry point: pass already-split, already-preprocessed
        train/test dataframes (e.g. from DataPreprocessor.fit_transform(train)
        and DataPreprocessor.transform(test)). No further splitting happens
        here, so preprocessing statistics fit on train never touch test."""
        if target_col not in df_train.columns:
            raise ValueError(f"Target column '{target_col}' not found in training dataframe")
        if target_col not in df_test.columns:
            raise ValueError(f"Target column '{target_col}' not found in test dataframe")

        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]
        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col]

        task_type = self._detect_task_type(y_train) if task == 'auto' else task
        logger.info(f"Task type: {task_type}" + (" (auto-detected)" if task == 'auto' else ""))

        return self._fit_and_evaluate_core(X_train, X_test, y_train, y_test, task_type, target_col)

    def _can_stratify(self, y: pd.Series, task_type: str) -> bool:
        if task_type != 'classification':
            return False
        class_counts = y.value_counts()
        min_class_count = class_counts.min()
        if min_class_count >= 2:
            return True
        logger.warning(
            f"Cannot use stratified split: minimum class has only {min_class_count} sample(s). "
            f"Class distribution: {dict(class_counts)}"
        )
        return False

    def _fit_and_evaluate_core(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.Series, y_test: pd.Series,
                                task_type: str, target_col: str) -> Dict[str, Any]:
        self.task_type = task_type

        # Validate data is sufficient for training
        if len(X_train) < 10:
            raise ValueError(f"Insufficient training data: only {len(X_train)} samples. Need at least 10 samples.")

        # Additional validation for classification
        if self.task_type == 'classification':
            train_class_counts = y_train.value_counts()
            n_classes = len(train_class_counts)
            classes_with_one_sample = train_class_counts[train_class_counts == 1].index.tolist()

            # Check if this looks more like regression data
            if n_classes > len(y_train) * 0.8:
                raise ValueError(
                    f"Classification detected, but target has {n_classes} classes for {len(y_train)} training samples "
                    f"({n_classes / len(y_train) * 100:.1f}% unique). This is definitely a REGRESSION problem, "
                    f"not classification.\n"
                    f"Solution: Set task='regression' explicitly when calling fit_and_evaluate()."
                )

            if len(classes_with_one_sample) > 0:
                logger.warning(
                    f"Found {len(classes_with_one_sample)} classes with only 1 sample in training set. "
                    f"This will cause issues with cross-validation."
                )

                # If too many singleton classes, this is definitely regression
                singleton_ratio = len(classes_with_one_sample) / n_classes
                if singleton_ratio > 0.5:
                    raise ValueError(
                        f"Classification detected, but {len(classes_with_one_sample)}/{n_classes} classes "
                        f"({singleton_ratio * 100:.1f}%) have only 1 sample. This is a REGRESSION problem, "
                        f"not classification.\n"
                        f"Solution: Set task='regression' explicitly when calling fit_and_evaluate()."
                    )

                if singleton_ratio > 0.3:
                    logger.error(
                        f"Over 30% of classes have only 1 sample. This might be a regression problem. "
                        f"Try setting task='regression' explicitly."
                    )

                # Only remove singleton classes if it's a small portion
                if len(classes_with_one_sample) < n_classes * 0.3:
                    logger.warning(f"Removing {len(classes_with_one_sample)} singleton classes to enable training...")
                    valid_mask = ~y_train.isin(classes_with_one_sample)
                    X_train = X_train[valid_mask]
                    y_train = y_train[valid_mask]

                    # Also remove from test set
                    valid_mask_test = ~y_test.isin(classes_with_one_sample)
                    X_test = X_test[valid_mask_test]
                    y_test = y_test[valid_mask_test]

                    logger.info(f"Training set reduced to {len(X_train)} samples after removing singleton classes")

                    if len(X_train) < 10:
                        raise ValueError(
                            f"After removing singleton classes, only {len(X_train)} samples remain. "
                            f"This dataset is too small or imbalanced for reliable classification. "
                            f"Solution: Set task='regression' to treat this as a continuous prediction problem."
                        )
                else:
                    raise ValueError(
                        f"Cannot proceed: {len(classes_with_one_sample)}/{n_classes} classes have only 1 sample. "
                        f"This is too many to remove. This is definitely a REGRESSION problem.\n"
                        f"Solution: Set task='regression' explicitly when calling fit_and_evaluate()."
                    )

        # Determine appropriate CV folds
        cv_folds = self._determine_cv_folds(y_train)
        logger.info(f"Using {cv_folds}-fold cross-validation")

        # Honor the parallel_training toggle: -1 uses all cores, 1 is sequential.
        self.n_jobs = -1 if self.config.get('parallel_training', True) else 1
        tune = self.config.get('enable_hyperparameter_tuning', False)

        models_to_try = self._get_models()

        results = {
            'task_type': self.task_type,
            'target_column': target_col,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'feature_count': X_train.shape[1],
            'models': {}
        }

        # Train and evaluate each model
        for name, model in models_to_try.items():
            try:
                if tune and name in ('RandomForest', 'GradientBoosting'):
                    logger.info(f"Tuning {name} via GridSearchCV...")
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, None] if name == 'RandomForest' else [3, 5, 7],
                    }
                    grid = GridSearchCV(
                        model, param_grid,
                        cv=min(cv_folds, 3),
                        n_jobs=self.n_jobs,
                        scoring='accuracy' if self.task_type == 'classification' else 'r2'
                    )
                    grid.fit(X_train, y_train)
                    model = grid.best_estimator_
                    logger.info(f"{name} best params: {grid.best_params_}")
                else:
                    logger.info(f"Training {name}...")
                    model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                if self.task_type == 'classification':
                    metrics = self._calculate_classification_metrics(y_test, y_pred)
                    score = metrics['accuracy']
                else:
                    metrics = self._calculate_regression_metrics(y_test, y_pred)
                    score = metrics['r2_score']

                # Perform cross-validation with appropriate strategy
                try:
                    if self.task_type == 'classification':
                        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    else:
                        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=self.n_jobs)
                    metrics['cv_mean'] = cv_scores.mean()
                    metrics['cv_std'] = cv_scores.std()
                except Exception as cv_error:
                    logger.warning(f"Cross-validation failed for {name}: {cv_error}. Using single validation score.")
                    metrics['cv_mean'] = score
                    metrics['cv_std'] = 0.0

                results['models'][name] = metrics

                # Track best model
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = name
                    self.models[name] = model

                logger.info(f"{name}: Score={score:.4f}, CV={metrics['cv_mean']:.4f} ±{metrics['cv_std']:.4f}")

            except Exception as e:
                logger.warning(f"Failed to train {name}: {str(e)}")
                continue

        # Ensure at least one model trained successfully
        if not results['models']:
            raise RuntimeError("All models failed to train. Please check your data and try again.")

        results['best_model'] = self.best_model
        results['best_score'] = self.best_score

        # Extract feature importance from best model
        if self.best_model and self.best_model in self.models:
            results['feature_importance'] = self._get_feature_importance(
                self.models[self.best_model],
                X_train.columns.tolist()
            )

        return results

    def _detect_task_type(self, y: pd.Series) -> str:
        """
        Enhanced task detection that considers:
        1. Data type
        2. Number of unique values
        3. Ratio of unique values to total samples
        4. Distribution of values
        5. Practical trainability for classification
        """
        n_samples = len(y)
        n_unique = y.nunique()
        unique_ratio = n_unique / n_samples

        logger.info(f"Analyzing target: {n_samples} samples, {n_unique} unique values ({unique_ratio:.2%} unique)")

        # CRITICAL: If almost all values are unique, this is regression
        if unique_ratio > 0.9:
            logger.info(f"Task: Regression (>90% unique values - continuous target)")
            return 'regression'

        # If more than 50 unique values per 100 samples, likely regression
        if n_unique > n_samples * 0.5:
            logger.info(f"Task: Regression ({unique_ratio:.1%} unique - too many for classification)")
            return 'regression'

        # Check for singleton classes (classes with only 1 sample)
        value_counts = y.value_counts()
        singleton_classes = (value_counts == 1).sum()
        singleton_ratio = singleton_classes / n_unique if n_unique > 0 else 0

        # If more than 50% of classes are singletons, this is regression
        if singleton_ratio > 0.5:
            logger.info(
                f"Task: Regression ({singleton_classes}/{n_unique} classes have only 1 sample - "
                f"too sparse for classification)"
            )
            return 'regression'

        # Check data type
        if y.dtype in ['object', 'category', 'bool']:
            # Even for categorical dtype, check if it's too many categories
            if n_unique > 100:
                logger.warning(
                    f"Categorical dtype with {n_unique} unique values. "
                    f"This is too many for classification. Consider encoding as numeric regression target."
                )
                return 'regression'
            logger.info(f"Task: Classification (categorical dtype, {n_unique} classes)")
            return 'classification'

        # If very few unique values, likely classification
        if n_unique <= 2:
            logger.info(f"Task: Classification (binary: {n_unique} classes)")
            return 'classification'

        if n_unique <= 20:
            logger.info(f"Task: Classification ({n_unique} classes)")
            return 'classification'

        # Check if values are all integers and could be class labels
        if np.all(y == y.astype(int)):
            # If unique ratio is less than 5%, likely classification
            if unique_ratio < 0.05 and n_unique <= 100:
                logger.info(f"Task: Classification ({n_unique} classes, {unique_ratio:.2%} unique)")
                return 'classification'
            # If values are sequential integers starting from 0 or 1 (and not too many)
            unique_vals = sorted(y.unique())
            if unique_vals[0] in [0, 1] and unique_vals[-1] == len(unique_vals) + unique_vals[0] - 1:
                if n_unique <= 50:
                    logger.info(f"Task: Classification (sequential labels 0-{unique_vals[-1]})")
                    return 'classification'
                else:
                    logger.info(f"Task: Regression (sequential 0-{unique_vals[-1]} but too many for classification)")
                    return 'regression'

        # Check if most common value appears frequently enough
        max_count = value_counts.max()
        max_count_ratio = max_count / n_samples

        # If most common value appears in >5% of data and reasonable number of classes
        if max_count_ratio > 0.05 and n_unique <= 50:
            logger.info(f"Task: Classification ({n_unique} classes with repeated values)")
            return 'classification'

        # If we have 20-50 classes but they're all rare
        if 20 < n_unique <= 50:
            # Check if there are enough samples per class on average
            avg_samples_per_class = n_samples / n_unique
            if avg_samples_per_class < 2:
                logger.info(
                    f"Task: Regression ({n_unique} classes but only {avg_samples_per_class:.1f} "
                    f"samples per class on average - too sparse)"
                )
                return 'regression'

        # Default to regression for ambiguous cases
        logger.info(f"Task: Regression (default for continuous-looking data with {n_unique} unique values)")
        return 'regression'

    def _determine_cv_folds(self, y_train: pd.Series) -> int:
        """
        Determine appropriate number of CV folds based on:
        1. Sample size
        2. Class distribution (for classification)
        3. User configuration
        """
        n_samples = len(y_train)
        requested_folds = self.config.get('cv_folds', 5)

        # Minimum samples per fold
        min_samples_per_fold = 2
        max_folds_by_size = n_samples // min_samples_per_fold

        if self.task_type == 'classification':
            # For classification, ensure each class has enough samples
            class_counts = y_train.value_counts()
            min_class_count = class_counts.min()

            # Check for singleton classes (should have been removed earlier, but double-check)
            if min_class_count == 1:
                logger.warning(
                    f"Found class with only 1 sample during CV fold determination. "
                    f"This will cause issues. Falling back to 2-fold CV."
                )
                return 2

            # Each fold needs at least 1 sample from smallest class
            # With k folds, training set is (k-1)/k of data
            # So we need: min_class_count * (k-1)/k >= 1
            # Which gives: k <= min_class_count + 1
            max_folds_by_class = min(min_class_count, requested_folds)

            # For very imbalanced data, be more conservative
            if min_class_count <= 5:
                max_folds_by_class = min(2, min_class_count)
                logger.warning(
                    f"Minimum class has only {min_class_count} samples. "
                    f"Using {max_folds_by_class}-fold CV to ensure stability."
                )

            # Take minimum of all constraints
            cv_folds = min(requested_folds, max_folds_by_size, max_folds_by_class)

            # Warn if we had to reduce folds
            if cv_folds < requested_folds:
                logger.warning(
                    f"Reducing CV folds from {requested_folds} to {cv_folds} "
                    f"(min class has {min_class_count} samples, total {n_samples} samples)"
                )
        else:
            # For regression, only consider sample size
            cv_folds = min(requested_folds, max_folds_by_size)

            if cv_folds < requested_folds:
                logger.warning(
                    f"Reducing CV folds from {requested_folds} to {cv_folds} "
                    f"due to small sample size ({n_samples} samples)"
                )

        # Ensure at least 2 folds
        cv_folds = max(2, cv_folds)

        return cv_folds

    def _get_models(self) -> Dict[str, Any]:
        if self.task_type == 'classification':
            return {
                'RandomForest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=self.n_jobs,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2
                ),
                'GradientBoosting': GradientBoostingClassifier(
                    n_estimators=100,
                    random_state=42,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2
                ),
                'LogisticRegression': LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    solver='lbfgs'
                )
            }
        else:
            return {
                'RandomForest': RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=self.n_jobs,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2
                ),
                'GradientBoosting': GradientBoostingRegressor(
                    n_estimators=100,
                    random_state=42,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2
                ),
                'Ridge': Ridge(random_state=42, alpha=1.0),
                'Lasso': Lasso(random_state=42, alpha=1.0, max_iter=1000)
            }

    def _calculate_classification_metrics(self, y_true, y_pred) -> Dict[str, float]:
        # Determine if binary or multiclass
        n_classes = len(np.unique(y_true))

        if n_classes == 2:
            average = 'binary'
        else:
            average = 'weighted'

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
        }

    def _calculate_regression_metrics(self, y_true, y_pred) -> Dict[str, float]:
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred)
        }

    def _get_feature_importance(self, model, feature_names) -> Dict[str, float]:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]

            return {
                feature_names[i]: float(importances[i])
                for i in indices
            }
        elif hasattr(model, 'coef_'):
            coef = np.abs(model.coef_)
            if len(coef.shape) > 1:
                coef = coef.mean(axis=0)
            indices = np.argsort(coef)[::-1][:20]

            return {
                feature_names[i]: float(coef[i])
                for i in indices
            }
        else:
            return {}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_model is None or self.best_model not in self.models:
            raise ValueError("No model has been trained yet")

        return self.models[self.best_model].predict(X)
