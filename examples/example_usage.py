import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.utils import setup_logging, load_config
from modules.preprocess import DataPreprocessor
from modules.estimation import ModelEstimator
from modules.report_gen import ReportGenerator

logger = setup_logging()

def example_1_classification():
    logger.info("=" * 60)
    logger.info("Example 1: Classification Task - Loan Approval Prediction")
    logger.info("=" * 60)
    
    config = load_config('configs/default_config.yaml')
    
    preprocessor = DataPreprocessor(config['preprocessing'])
    df_clean, metadata = preprocessor.process('data/loan_approval.csv')
    
    logger.info(f"Data shape after preprocessing: {df_clean.shape}")
    logger.info(f"Columns: {list(df_clean.columns)}")
    
    estimator = ModelEstimator(config['estimation'])
    results = estimator.fit_and_evaluate(df_clean, 'loan_approved', 'classification')
    
    logger.info(f"\nBest Model: {results['best_model']}")
    logger.info(f"Best Accuracy: {results['best_score']:.4f}")
    
    logger.info("\nAll Model Results:")
    for model_name, metrics in results['models'].items():
        logger.info(f"  {model_name}: Accuracy={metrics['accuracy']:.4f}, CV={metrics['cv_mean']:.4f}")
    
    output_dir = Path('output/example1_classification')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_gen = ReportGenerator(config['report'])
    report_paths = report_gen.generate(df_clean, metadata, results, output_dir, 'pdf')
    
    logger.info(f"\nReport generated: {report_paths[0]}")
    logger.info("=" * 60)

def example_2_regression():
    logger.info("=" * 60)
    logger.info("Example 2: Regression Task - Housing Price Prediction")
    logger.info("=" * 60)
    
    config = load_config('configs/default_config.yaml')
    
    preprocessor = DataPreprocessor(config['preprocessing'])
    df_clean, metadata = preprocessor.process('data/housing_prices.csv')
    
    logger.info(f"Data shape after preprocessing: {df_clean.shape}")
    
    estimator = ModelEstimator(config['estimation'])
    results = estimator.fit_and_evaluate(df_clean, 'price', 'regression')
    
    logger.info(f"\nBest Model: {results['best_model']}")
    logger.info(f"Best R² Score: {results['best_score']:.4f}")
    
    logger.info("\nAll Model Results:")
    for model_name, metrics in results['models'].items():
        logger.info(f"  {model_name}: R²={metrics['r2_score']:.4f}, RMSE={metrics['rmse']:.2f}")
    
    output_dir = Path('output/example2_regression')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_gen = ReportGenerator(config['report'])
    report_paths = report_gen.generate(df_clean, metadata, results, output_dir, 'both')
    
    logger.info(f"\nReports generated:")
    for path in report_paths:
        logger.info(f"  - {path}")
    logger.info("=" * 60)

def example_3_custom_config():
    logger.info("=" * 60)
    logger.info("Example 3: Custom Configuration - Advanced Preprocessing")
    logger.info("=" * 60)
    
    custom_config = {
        'preprocessing': {
            'missing_strategy': 'knn',
            'missing_threshold': 0.3,
            'outlier_method': 'zscore',
            'cap_outliers': True,
            'onehot_threshold': 5,
            'scale_features': True,
            'scaler': 'robust',
            'feature_engineering': True
        },
        'estimation': {
            'test_size': 0.25,
            'random_state': 123,
            'cv_folds': 10
        },
        'report': {
            'include_charts': True,
            'chart_style': 'darkgrid',
            'dpi': 300
        }
    }
    
    preprocessor = DataPreprocessor(custom_config['preprocessing'])
    df_clean, metadata = preprocessor.process('data/housing_prices.csv')
    
    logger.info(f"Original columns: {metadata['original_shape'][1]}")
    logger.info(f"Final columns (with feature engineering): {metadata['final_shape'][1]}")
    
    estimator = ModelEstimator(custom_config['estimation'])
    results = estimator.fit_and_evaluate(df_clean, 'price', 'regression')
    
    logger.info(f"\nWith 10-fold CV:")
    logger.info(f"Best Model: {results['best_model']}")
    logger.info(f"Best R² Score: {results['best_score']:.4f}")
    
    if results['feature_importance']:
        logger.info("\nTop 5 Most Important Features:")
        for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:5], 1):
            logger.info(f"  {i}. {feature}: {importance:.4f}")
    
    output_dir = Path('output/example3_custom')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_gen = ReportGenerator(custom_config['report'])
    report_paths = report_gen.generate(df_clean, metadata, results, output_dir, 'pdf')
    
    logger.info(f"\nReport generated: {report_paths[0]}")
    logger.info("=" * 60)

def example_4_data_exploration_only():
    logger.info("=" * 60)
    logger.info("Example 4: Data Exploration Without Modeling")
    logger.info("=" * 60)
    
    config = load_config('configs/default_config.yaml')
    
    preprocessor = DataPreprocessor(config['preprocessing'])
    df_clean, metadata = preprocessor.process('data/loan_approval.csv')
    
    logger.info(f"Data Overview:")
    logger.info(f"  Rows: {metadata['final_shape'][0]}")
    logger.info(f"  Columns: {metadata['final_shape'][1]}")
    logger.info(f"  Duplicates removed: {metadata.get('duplicates_removed', 0)}")
    
    if metadata.get('missing_values'):
        logger.info(f"\nMissing values handled:")
        for col, count in list(metadata['missing_values'].items())[:5]:
            logger.info(f"  {col}: {count} values")
    
    output_dir = Path('output/example4_exploration')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_gen = ReportGenerator(config['report'])
    report_paths = report_gen.generate(df_clean, metadata, None, output_dir, 'pdf')
    
    logger.info(f"\nExploration report generated: {report_paths[0]}")
    logger.info("=" * 60)

def example_5_excel_input():
    logger.info("=" * 60)
    logger.info("Example 5: Processing Excel File")
    logger.info("=" * 60)
    
    config = load_config('configs/default_config.yaml')
    
    preprocessor = DataPreprocessor(config['preprocessing'])
    df_clean, metadata = preprocessor.process('data/housing_prices.xlsx')
    
    logger.info(f"Successfully loaded Excel file")
    logger.info(f"Data shape: {df_clean.shape}")
    
    estimator = ModelEstimator(config['estimation'])
    results = estimator.fit_and_evaluate(df_clean, 'price', 'auto')
    
    logger.info(f"\nAuto-detected task type: {results['task_type']}")
    logger.info(f"Best Model: {results['best_model']}")
    logger.info(f"Best Score: {results['best_score']:.4f}")
    
    output_dir = Path('output/example5_excel')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_gen = ReportGenerator(config['report'])
    report_paths = report_gen.generate(df_clean, metadata, results, output_dir, 'docx')
    
    logger.info(f"\nDOCX report generated: {report_paths[0]}")
    logger.info("=" * 60)

def run_all_examples():
    try:
        example_1_classification()
        print("\n")
        
        example_2_regression()
        print("\n")
        
        example_3_custom_config()
        print("\n")
        
        example_4_data_exploration_only()
        print("\n")
        
        example_5_excel_input()
        
        logger.info("\n" + "=" * 60)
        logger.info("All examples completed successfully!")
        logger.info("Check the 'output' directory for generated reports")
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"Sample data not found: {e}")
        logger.error("Please run: python data/generate_sample_data.py")
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DataPrepX Usage Examples')
    parser.add_argument('--example', type=int, choices=[1, 2, 3, 4, 5], 
                       help='Run specific example (1-5)')
    parser.add_argument('--all', action='store_true', 
                       help='Run all examples')
    
    args = parser.parse_args()
    
    if args.all:
        run_all_examples()
    elif args.example == 1:
        example_1_classification()
    elif args.example == 2:
        example_2_regression()
    elif args.example == 3:
        example_3_custom_config()
    elif args.example == 4:
        example_4_data_exploration_only()
    elif args.example == 5:
        example_5_excel_input()
    else:
        print("Usage:")
        print("  python examples/example_usage.py --all")
        print("  python examples/example_usage.py --example 1")
        print("\nAvailable examples:")
        print("  1: Classification Task - Loan Approval")
        print("  2: Regression Task - Housing Prices")
        print("  3: Custom Configuration")
        print("  4: Data Exploration Only")
        print("  5: Excel File Processing")