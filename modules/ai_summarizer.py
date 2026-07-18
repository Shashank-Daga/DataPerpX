# import json
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from modules.utils import setup_logging

logger = setup_logging()


class AISummarizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_url = config.get('api_url', 'http://localhost:1234/v1/chat/completions')
        self.model = config.get('model', 'local-model')

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace('###', '').replace('##', '').replace('**', '')
        text = text.replace('■', '').replace('▪', '').replace('•', '')
        text = text.replace('\u2011', '-').replace('\xd7', 'x')
        text = text.replace('\u2192', '->').replace('\xb2', '²')
        text = text.replace('`', '').replace('*', '')

        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith(('#', '-', '*', '.')):
                line = ' '.join(line.split())
                cleaned_lines.append(line)

        return '\n\n'.join(cleaned_lines)

    def generate_summary(self, df_info: Dict[str, Any], metadata: Dict[str, Any],
                         results: Optional[Dict[str, Any]] = None) -> tuple:
        """Returns (summary_text, source) where source is 'llm' or 'fallback',
        so callers can tell the user whether this was actually AI-generated."""

        prompt = self._build_prompt(df_info, metadata, results)

        try:
            response = self._call_llm(prompt)
            return self._clean_text(response), 'llm'
        except Exception as e:
            logger.warning(f"AI summarization failed: {e}")
            return self._clean_text(self._fallback_summary(df_info, metadata, results)), 'fallback'

    def generate_data_quality_report(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> tuple:

        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
            'missing_values_total': df.isnull().sum().sum(),
            'duplicate_rows': metadata.get('duplicates_removed', 0),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
        }

        prompt = f"""Generate a professional data quality assessment report based on the following statistics:

Dataset Statistics:
- Total Rows: {stats['total_rows']:,}
- Total Columns: {stats['total_columns']}
- Numeric Columns: {stats['numeric_columns']}
- Categorical Columns: {stats['categorical_columns']}
- Total Missing Values: {stats['missing_values_total']}
- Duplicate Rows Removed: {stats['duplicate_rows']}
- Memory Usage: {stats['memory_usage_mb']:.2f} MB

Missing Values by Column:
{self._format_missing_values(metadata.get('missing_values', {}))}

Outliers Detected:
{self._format_outliers(metadata.get('outliers', {}))}

Provide a detailed analysis covering:
1. Overall data quality assessment
2. Data completeness and integrity
3. Potential data quality issues
4. Recommendations for improvement

IMPORTANT: Do not use any markdown formatting, bullet points, asterisks, or special symbols. Write in clear, 
flowing paragraphs only."""

        try:
            response = self._call_llm(prompt)
            return self._clean_text(response), 'llm'
        except Exception as e:
            logger.warning(f"Data quality report generation failed: {e}")
            return self._clean_text(self._fallback_data_quality(stats, metadata)), 'fallback'

    def generate_model_insights(self, results: Dict[str, Any]) -> tuple:

        prompt = f"""Generate professional insights and recommendations based on these machine learning results:

Task Type: {results.get('task_type', 'N/A')}
Best Model: {results.get('best_model', 'N/A')}
Best Score: {results.get('best_score', 0):.4f}

Detailed Model Performance:
{self._format_model_metrics(results.get('models', {}), results.get('task_type', 'classification'))}

Top 10 Important Features:
{self._format_feature_importance(results.get('feature_importance', {}))}

Provide:
1. Model performance analysis and comparison
2. Feature importance interpretation
3. Model selection justification
4. Potential improvements and next steps
5. Production deployment considerations

IMPORTANT: Do not use any markdown formatting, bullet points, asterisks, or special symbols. Write in clear, 
flowing paragraphs only."""

        try:
            response = self._call_llm(prompt)
            return self._clean_text(response), 'llm'
        except Exception as e:
            logger.warning(f"Model insights generation failed: {e}")
            return self._clean_text(self._fallback_model_insights(results)), 'fallback'

    def generate_feature_analysis(self, df: pd.DataFrame, target_col: str) -> tuple:

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        feature_stats = []
        for col in numeric_cols[:10]:
            stats = {
                'name': col,
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
            feature_stats.append(stats)

        prompt = f"""Analyze these feature statistics and provide insights:

Target Variable: {target_col}

Feature Statistics (Top 10 Numeric Features):
{self._format_feature_stats(feature_stats)}

Provide:
1. Distribution analysis for key features
2. Identification of skewed or unusual distributions
3. Feature relationships with target variable
4. Data transformation recommendations

IMPORTANT: Do not use any markdown formatting, bullet points, asterisks, or special symbols. Write in clear, 
flowing paragraphs only."""

        try:
            response = self._call_llm(prompt)
            return self._clean_text(response), 'llm'
        except Exception as e:
            logger.warning(f"Feature analysis failed: {e}")
            return self._clean_text(self._fallback_feature_analysis(feature_stats)), 'fallback'

    def generate_business_recommendations(self, df_info: Dict[str, Any],
                                          metadata: Dict[str, Any],
                                          results: Optional[Dict[str, Any]] = None) -> tuple:

        prompt = f"""Based on this data analysis and modeling project, provide business-focused recommendations:

Project Overview:
- Dataset Size: {metadata['final_shape'][0]:,} rows, {metadata['final_shape'][1]} columns
- Analysis Type: {results.get('task_type', 'exploratory').title() if results else 'Exploratory'}
"""

        if results:
            prompt += f"""- Best Model Performance: {results.get('best_score', 0):.4f}
- Top Contributing Factor: 
{list(results.get('feature_importance', {}).keys())[0] if results.get('feature_importance') else 'N/A'}
"""

        prompt += """
Provide:
1. Key business insights from the analysis
2. Actionable recommendations for stakeholders
3. Risk factors and considerations
4. ROI potential and implementation strategy
5. Monitoring and maintenance recommendations

IMPORTANT: Do not use any markdown formatting, bullet points, asterisks, or special symbols. Write in clear, 
flowing paragraphs only."""

        try:
            response = self._call_llm(prompt)
            return self._clean_text(response), 'llm'
        except Exception as e:
            logger.warning(f"Business recommendations generation failed: {e}")
            return self._clean_text(self._fallback_business_recommendations(results)), 'fallback'

    @staticmethod
    def _format_missing_values(missing_values: Dict[str, int]) -> str:
        if not missing_values:
            return "No missing values detected"

        lines = []
        for col, count in list(missing_values.items())[:10]:
            lines.append(f"{col}: {count} missing values")
        return "\n".join(lines)

    @staticmethod
    def _format_outliers(outliers: Dict[str, int]) -> str:
        if not outliers:
            return "No significant outliers detected"

        lines = []
        for col, count in list(outliers.items())[:10]:
            if count > 0:
                lines.append(f"{col}: {count} outliers")
        return "\n".join(lines) if lines else "No significant outliers detected"

    @staticmethod
    def _format_model_metrics(models: Dict[str, Any], task_type: str) -> str:
        lines = []
        for model_name, metrics in models.items():
            if task_type == 'classification':
                lines.append(f"{model_name}:")
                lines.append(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                lines.append(f"  Precision: {metrics.get('precision', 0):.4f}")
                lines.append(f"  Recall: {metrics.get('recall', 0):.4f}")
                lines.append(f"  F1-Score: {metrics.get('f1_score', 0):.4f}")
                lines.append(f"  Cross-Validation: {metrics.get('cv_mean', 0):.4f} ± {metrics.get('cv_std', 0):.4f}")
            else:
                lines.append(f"{model_name}:")
                lines.append(f"  R² Score: {metrics.get('r2_score', 0):.4f}")
                lines.append(f"  RMSE: {metrics.get('rmse', 0):.2f}")
                lines.append(f"  MAE: {metrics.get('mae', 0):.2f}")
                lines.append(f"  Cross-Validation: {metrics.get('cv_mean', 0):.4f} ± {metrics.get('cv_std', 0):.4f}")
        return "\n".join(lines)

    @staticmethod
    def _format_feature_importance(feature_importance: Dict[str, float]) -> str:
        if not feature_importance:
            return "Feature importance not available"

        lines = []
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
            lines.append(f"{i}. {feature}: {importance:.4f}")
        return "\n".join(lines)

    @staticmethod
    def _format_feature_stats(feature_stats: List[Dict[str, Any]]) -> str:
        lines = []
        for stat in feature_stats:
            lines.append(f"\n{stat['name']}:")
            lines.append(f"  Mean: {stat['mean']:.4f}")
            lines.append(f"  Median: {stat['median']:.4f}")
            lines.append(f"  Std Dev: {stat['std']:.4f}")
            lines.append(f"  Range: [{stat['min']:.4f}, {stat['max']:.4f}]")
            lines.append(f"  Skewness: {stat['skewness']:.4f}")
            lines.append(f"  Kurtosis: {stat['kurtosis']:.4f}")
        return "\n".join(lines)

    @staticmethod
    def _build_prompt(df_info: Dict[str, Any], metadata: Dict[str, Any],
                      results: Optional[Dict[str, Any]]) -> str:

        context = f"""Generate a comprehensive executive summary for this data science project.

Dataset Information:
- Original dimensions: {metadata.get('original_shape', 'N/A')}
- Final dimensions: {metadata.get('final_shape', 'N/A')}
- Features engineered: {metadata.get('final_shape', [0, 0])[1] - metadata.get('original_shape', [0, 0])[1]}
- Data quality improvements: {metadata.get('duplicates_removed', 0)} duplicates removed
"""

        if metadata.get('missing_values'):
            context += f"\nData Cleaning:\n"
            for col, count in list(metadata['missing_values'].items())[:5]:
                context += f"Imputed {count} missing values in {col}\n"

        if results:
            context += f"\nMachine Learning Pipeline:\n"
            context += f"- Objective: {results.get('task_type', 'N/A').title()}\n"
            context += f"- Target Variable: {results.get('target_column', 'N/A')}\n"
            context += f"- Models Evaluated: {len(results.get('models', {}))}\n"
            context += f"- Training Dataset: {results.get('train_size', 0):,} samples\n"
            context += f"- Test Dataset: {results.get('test_size', 0):,} samples\n"
            context += f"- Best Model: {results.get('best_model', 'N/A')}\n"
            context += f"- Performance Score: {results.get('best_score', 0):.4f}\n"

            if results.get('models'):
                context += "\nModel Performance Summary:\n"
                for model_name, metrics in results['models'].items():
                    if results['task_type'] == 'classification':
                        context += f"{model_name}: Accuracy={metrics.get('accuracy', 0):.4f}, "
                        context += f"F1={metrics.get('f1_score', 0):.4f}\n"
                    else:
                        context += f"{model_name}: R²={metrics.get('r2_score', 0):.4f}, "
                        context += f"RMSE={metrics.get('rmse', 0):.2f}\n"

            if results.get('feature_importance'):
                context += f"\nKey Predictive Features (Top 5):\n"
                for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:5], 1):
                    context += f"{i}. {feature}: {importance:.4f}\n"

        context += """\n\nProvide a professional executive summary including:
1. Project overview and objectives
2. Data processing and quality improvements
3. Key findings from the analysis
4. Model performance and reliability
5. Business implications and recommendations
6. Next steps and considerations

IMPORTANT: Do not use any markdown formatting (no ###, **, -, *, bullets, or special symbols). Write in clear, 
flowing paragraphs with proper sentence structure. Use regular numbers (1, 2, 3) for lists if needed, 
but avoid bullet points."""

        return context

    def _call_llm(self, prompt: str) -> str:
        headers = {
            'Content-Type': 'application/json'
        }

        payload = {
            'model': self.model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a senior data scientist and business analyst providing professional insights '
                               'on machine learning projects. Your responses should be clear, actionable, '
                               'and suitable for both technical and business audiences. Never use markdown formatting '
                               'or special symbols. Write in clear, professional prose.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'temperature': 0.7,
            'max_tokens': 2000
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"API call failed with status {response.status_code}")

    @staticmethod
    def _fallback_summary(df_info: Dict[str, Any], metadata: Dict[str, Any],
                          results: Optional[Dict[str, Any]]) -> str:

        summary_parts = [
            "EXECUTIVE SUMMARY",
            "",
            "PROJECT OVERVIEW",
            f"This analysis processed a dataset containing {metadata.get('original_shape', [0, 0])[0]:,} records",
            f"and {metadata.get('original_shape', [0, 0])[1]} initial features. Through comprehensive preprocessing",
            f"and feature engineering, the dataset was enhanced to {metadata.get('final_shape', [0, 0])[1]} features.",
            "",
        ]

        if metadata.get('duplicates_removed', 0) > 0 or metadata.get('missing_values'):
            summary_parts.append("DATA QUALITY IMPROVEMENTS")
            if metadata.get('duplicates_removed', 0) > 0:
                summary_parts.append(f"Removed {metadata.get('duplicates_removed')} duplicate records")
            if metadata.get('missing_values'):
                summary_parts.append(f"Handled missing values in {len(metadata['missing_values'])} columns")
            summary_parts.append("")

        if results:
            summary_parts.append("MACHINE LEARNING RESULTS")
            summary_parts.append(f"Task Type: {results.get('task_type', 'N/A').title()}")
            summary_parts.append(f"Best Model: {results.get('best_model', 'N/A')}")
            summary_parts.append(f"Performance Score: {results.get('best_score', 0):.4f}")
            summary_parts.append("")

            if results.get('models'):
                summary_parts.append("MODEL COMPARISON")
                for model_name, metrics in results['models'].items():
                    if results['task_type'] == 'classification':
                        summary_parts.append(
                            f"{model_name}: Accuracy={metrics.get('accuracy', 0):.4f}, F1="
                            f"{metrics.get('f1_score', 0):.4f}")
                    else:
                        summary_parts.append(
                            f"{model_name}: R²={metrics.get('r2_score', 0):.4f}, RMSE={metrics.get('rmse', 0):.2f}")
                summary_parts.append("")

        summary_parts.append("RECOMMENDATIONS")
        summary_parts.append("Review feature importance to identify key drivers")
        summary_parts.append("Validate model performance on new data before deployment")
        summary_parts.append("Implement monitoring for data quality and model drift")
        summary_parts.append("Consider ensemble methods for improved performance")

        return "\n".join(summary_parts)

    @staticmethod
    def _fallback_data_quality(stats: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        completeness = ((stats['total_rows'] * stats['total_columns'] - stats['missing_values_total'])
                        / (stats['total_rows'] * stats['total_columns']) * 100)
        parts = [
            "DATA QUALITY ASSESSMENT",
            "",
            f"Dataset contains {stats['total_rows']:,} rows and {stats['total_columns']} columns",
            f"Numeric features: {stats['numeric_columns']}, Categorical features: {stats['categorical_columns']}",
            f"Memory footprint: {stats['memory_usage_mb']:.2f} MB",
            "",
            f"Data completeness: {completeness:.2f}%",
        ]
        return "\n".join(parts)

    @staticmethod
    def _fallback_model_insights(results: Dict[str, Any]) -> str:
        parts = [
            "MODEL PERFORMANCE INSIGHTS",
            f"Best performing model: {results.get('best_model', 'N/A')}",
            f"Performance score: {results.get('best_score', 0):.4f}",
            "",
            "This model demonstrates strong predictive capability on the test dataset.",
            "Consider cross-validation results for production deployment assessment.",
        ]
        return "\n".join(parts)

    @staticmethod
    def _fallback_feature_analysis(feature_stats: List[Dict[str, Any]]) -> str:
        parts = ["FEATURE ANALYSIS"]
        parts.extend(f"{stat['name']}: Mean={stat['mean']:.4f}, Std={stat['std']:.4f}"
                     for stat in feature_stats[:5])
        return "\n".join(parts)

    @staticmethod
    def _fallback_business_recommendations(results: Optional[Dict[str, Any]]) -> str:
        parts = ["BUSINESS RECOMMENDATIONS"]

        if results and results.get('best_model'):
            task = results.get('task_type', 'the target variable')
            score = results.get('best_score', 0)
            model = results.get('best_model')

            parts.append(
                f"The {model} model was selected as the best performer for this "
                f"{task} task, achieving a score of {score:.4f}."
            )

            if results.get('task_type') == 'classification' and score < 0.7:
                parts.append(
                    "Model accuracy is moderate; consider gathering more training "
                    "data or engineering additional features before deployment."
                )
            elif results.get('task_type') == 'regression' and score < 0.5:
                parts.append(
                    "The R\u00b2 score suggests the model explains less than half "
                    "of the variance in the target; treat predictions with caution."
                )
            else:
                parts.append(
                    "Model performance appears solid; validate on a held-out or "
                    "more recent dataset before production use."
                )

            if results.get('feature_importance'):
                top_feature = next(iter(results['feature_importance']))
                parts.append(f"'{top_feature}' was the most influential feature and is worth monitoring for drift.")
        else:
            parts.append("No trained model was available; recommendations are limited to data quality.")

        parts.append("Deploy model in controlled environment with monitoring")
        parts.append("Track key performance indicators against baseline")
        parts.append("Establish feedback loop for continuous improvement")
        parts.append("Document assumptions and limitations for stakeholders")
        return "\n".join(parts)
