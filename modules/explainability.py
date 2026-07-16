import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional
from pathlib import Path
import shap
from lime import lime_tabular
from modules.utils import setup_logging

logger = setup_logging()

class ExplainabilityAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.shap_values = None
        self.lime_explanations = []
        self.feature_names = None
        
    def analyze(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                task_type: str) -> Dict[str, Any]:
        
        logger.info("Starting explainability analysis...")
        
        self.feature_names = X.columns.tolist()
        
        results = {
            'task_type': task_type,
            'shap_analysis': None,
            'lime_analysis': None,
            'global_importance': None,
            'local_explanations': []
        }
        
        try:
            logger.info("Computing SHAP values...")
            shap_results = self._compute_shap(model, X, task_type)
            results['shap_analysis'] = shap_results
            logger.info("✓ SHAP analysis complete")
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
        
        try:
            logger.info("Computing LIME explanations...")
            lime_results = self._compute_lime(model, X, y, task_type)
            results['lime_analysis'] = lime_results
            logger.info("✓ LIME analysis complete")
        except Exception as e:
            logger.warning(f"LIME analysis failed: {e}")
        
        if results['shap_analysis']:
            results['global_importance'] = self._get_global_importance(
                results['shap_analysis']
            )
        
        return results
    
    def _compute_shap(self, model: Any, X: pd.DataFrame, task_type: str) -> Dict[str, Any]:
        
        sample_size = min(100, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            explainer_type = 'tree'
        except:
            try:
                X_background = shap.sample(X, 50)
                explainer = shap.KernelExplainer(model.predict, X_background)
                shap_values = explainer.shap_values(X_sample)
                explainer_type = 'kernel'
            except Exception as e:
                logger.warning(f"Could not create SHAP explainer: {e}")
                return None
        
        if isinstance(shap_values, list):
            if task_type == 'classification' and len(shap_values) == 2:
                shap_values = shap_values[1]
            else:
                shap_values = shap_values[0]
        
        self.shap_values = shap_values
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(self.feature_names, mean_abs_shap))
        sorted_importance = dict(sorted(feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True))
        
        return {
            'explainer_type': explainer_type,
            'shap_values': shap_values,
            'feature_importance': sorted_importance,
            'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else None,
            'X_sample': X_sample
        }
    
    def _compute_lime(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                     task_type: str) -> Dict[str, Any]:
        
        mode = 'classification' if task_type == 'classification' else 'regression'
        
        if mode == 'classification':
            class_names = [str(c) for c in sorted(y.unique())]
        else:
            class_names = None
        
        explainer = lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=self.feature_names,
            class_names=class_names,
            mode=mode,
            random_state=42
        )
        
        num_samples = min(5, len(X))
        sample_indices = np.random.choice(len(X), num_samples, replace=False)
        
        explanations = []
        
        for idx in sample_indices:
            instance = X.iloc[idx].values
            
            if mode == 'classification':
                if hasattr(model, 'predict_proba'):
                    exp = explainer.explain_instance(
                        instance, 
                        model.predict_proba,
                        num_features=10
                    )
                else:
                    continue
            else:
                exp = explainer.explain_instance(
                    instance,
                    model.predict,
                    num_features=10
                )
            
            exp_dict = {
                'instance_index': int(idx),
                'prediction': float(model.predict([instance])[0]),
                'actual': float(y.iloc[idx]),
                'feature_contributions': dict(exp.as_list())
            }
            
            explanations.append(exp_dict)
        
        return {
            'explanations': explanations,
            'num_samples': num_samples
        }
    
    def _get_global_importance(self, shap_analysis: Dict[str, Any]) -> Dict[str, float]:
        
        return shap_analysis['feature_importance']
    
    def generate_plots(self, output_dir: Path) -> Dict[str, Path]:
        
        plots_dir = output_dir / 'explainability_plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_paths = {}
        
        if self.shap_values is not None:
            try:
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    self.shap_values, 
                    features=self.feature_names,
                    show=False,
                    plot_type='bar'
                )
                path = plots_dir / 'shap_summary_bar.png'
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths['shap_summary_bar'] = path
                
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    self.shap_values,
                    show=False
                )
                path = plots_dir / 'shap_summary_beeswarm.png'
                plt.savefig(path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths['shap_summary_beeswarm'] = path
                
            except Exception as e:
                logger.warning(f"Failed to generate SHAP plots: {e}")
        
        return plot_paths
    
    def get_feature_explanations(self, top_n: int = 10) -> Dict[str, str]:
        
        if not self.shap_values is not None:
            return {}
        
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
        
        explanations = {}
        
        for idx in top_indices:
            feature = self.feature_names[idx]
            importance = mean_abs_shap[idx]
            
            explanation = f"Feature '{feature}' has average SHAP importance of {importance:.4f}. "
            
            feature_values = self.shap_values[:, idx]
            if np.mean(feature_values) > 0:
                explanation += "On average, this feature increases predictions."
            else:
                explanation += "On average, this feature decreases predictions."
            
            explanations[feature] = explanation
        
        return explanations