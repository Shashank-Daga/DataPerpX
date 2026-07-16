import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def run_web_gui():
    import matplotlib
    matplotlib.use('Agg')
    
    from flask import Flask, render_template, request, jsonify, send_file
    import pandas as pd
    import threading
    import uuid
    import shutil
    from modules.utils import setup_logging, load_config, load_data
    from modules.preprocess import DataPreprocessor
    from modules.estimation import ModelEstimator
    from modules.explainability import ExplainabilityAnalyzer
    from modules.report_gen import ReportGenerator
    from modules.ai_summarizer import AISummarizer
    
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
    
    Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
    
    jobs = {}
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/upload', methods=['POST'])
    def upload_file():
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            return jsonify({'error': 'Invalid file type. Use CSV or Excel files'}), 400
        
        try:
            job_id = str(uuid.uuid4())
            file_path = Path(app.config['UPLOAD_FOLDER']) / f"{job_id}_{file.filename}"
            file.save(str(file_path))
            
            df = load_data(str(file_path))
            columns = df.columns.tolist()
            
            return jsonify({
                'job_id': job_id,
                'filename': file.filename,
                'columns': columns,
                'rows': len(df),
                'default_target': columns[-1]
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/analyze', methods=['POST'])
    def analyze():
        data = request.json
        job_id = data.get('job_id')
        target_column = data.get('target')
        task_type = data.get('task', 'auto')
        report_format = data.get('report_format', 'pdf')
        enable_explain = data.get('explain', False)
        enable_tune = data.get('tune', False)
        
        if not job_id or not target_column:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        jobs[job_id] = {
            'status': 'running',
            'progress': 0,
            'message': 'Starting analysis...',
            'logs': []
        }
        
        thread = threading.Thread(
            target=run_analysis,
            args=(job_id, target_column, task_type, report_format, enable_explain, enable_tune)
        )
        thread.start()
        
        return jsonify({'job_id': job_id})
    
    @app.route('/status/<job_id>')
    def get_status(job_id):
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404
        return jsonify(jobs[job_id])
    
    @app.route('/download/<job_id>/<filename>')
    def download_file(job_id, filename):
        file_path = Path('output') / filename
        if file_path.exists():
            return send_file(str(file_path), as_attachment=True)
        return jsonify({'error': 'File not found'}), 404
    
    def run_analysis(job_id, target_column, task_type, report_format, enable_explain, enable_tune):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff()
        
        logger = setup_logging()
        
        try:
            upload_files = list(Path(app.config['UPLOAD_FOLDER']).glob(f"{job_id}_*"))
            if not upload_files:
                raise FileNotFoundError("Uploaded file not found")
            
            file_path = upload_files[0]
            
            jobs[job_id]['progress'] = 10
            jobs[job_id]['message'] = 'Loading data...'
            jobs[job_id]['logs'].append('Loading data...')
            
            df_original = load_data(str(file_path))
            
            if target_column not in df_original.columns:
                raise ValueError(f"Target column '{target_column}' not found")
            
            jobs[job_id]['progress'] = 20
            jobs[job_id]['logs'].append(f'Loaded {len(df_original)} rows, {len(df_original.columns)} columns')
            
            config = load_config('config/default_config.yaml')
            
            if enable_tune:
                config['estimation']['enable_hyperparameter_tuning'] = True
            
            target_data = df_original[target_column].copy()
            df_features = df_original.drop(columns=[target_column])
            
            temp_file = Path('temp_features.csv')
            df_features.to_csv(temp_file, index=False)
            
            jobs[job_id]['progress'] = 30
            jobs[job_id]['message'] = 'Preprocessing data...'
            jobs[job_id]['logs'].append('Preprocessing data...')
            
            preprocessor = DataPreprocessor(config['preprocessing'])
            df_clean, metadata = preprocessor.process(str(temp_file))
            temp_file.unlink()
            
            df_clean[target_column] = target_data.values
            
            jobs[job_id]['progress'] = 50
            jobs[job_id]['logs'].append(f'Preprocessed to {df_clean.shape[0]} rows, {df_clean.shape[1]} columns')
            
            jobs[job_id]['message'] = 'Training ML models...'
            jobs[job_id]['logs'].append('Training ML models...')
            
            estimator = ModelEstimator(config['estimation'])
            results = estimator.fit_and_evaluate(df_clean, target_column, task_type)
            
            jobs[job_id]['progress'] = 70
            jobs[job_id]['logs'].append(f"Best Model: {results['best_model']} (Score: {results['best_score']:.4f})")
            
            if enable_explain and results['best_model']:
                jobs[job_id]['message'] = 'Generating explainability...'
                jobs[job_id]['logs'].append('Generating explainability analysis...')
                
                explainer = ExplainabilityAnalyzer(config.get('explainability', {}))
                explainer.analyze(
                    estimator.models[results['best_model']],
                    df_clean.drop(columns=[target_column]),
                    df_clean[target_column],
                    results['task_type']
                )
            
            jobs[job_id]['progress'] = 80
            jobs[job_id]['message'] = 'Generating AI summary...'
            jobs[job_id]['logs'].append('Generating AI summary...')
            
            ai_config = {
                'api_url': 'http://localhost:1234/v1/chat/completions',
                'model': 'openai/gpt-oss-20b'
            }
            
            summarizer = AISummarizer(ai_config)
            df_info = {
                'columns': df_clean.columns.tolist(),
                'dtypes': df_clean.dtypes.to_dict(),
                'shape': df_clean.shape
            }
            
            ai_summary = summarizer.generate_summary(df_info, metadata, results)
            results['ai_summary'] = ai_summary
            
            jobs[job_id]['progress'] = 90
            jobs[job_id]['message'] = 'Generating reports...'
            jobs[job_id]['logs'].append('Generating reports...')
            
            output_dir = Path('output')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            report_gen = ReportGenerator(config['report'])
            report_paths = report_gen.generate(
                df_clean, metadata, results,
                output_dir, report_format
            )
            
            jobs[job_id]['progress'] = 100
            jobs[job_id]['status'] = 'completed'
            jobs[job_id]['message'] = 'Analysis complete!'
            jobs[job_id]['logs'].append('Analysis complete!')
            jobs[job_id]['reports'] = [str(Path(p).name) for p in report_paths]
            
            file_path.unlink()
            
        except Exception as e:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['message'] = f'Error: {str(e)}'
            jobs[job_id]['logs'].append(f'Error: {str(e)}')
    
    print("\n" + "="*60)
    print("DataPrepX Web GUI is running!")
    print("="*60)
    print("\nOpen your browser and go to:")
    print("  👉 http://localhost:5000")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=False, port=5000)

def run_cli():
    from modules.utils import setup_logging, load_config, load_data
    from modules.preprocess import DataPreprocessor
    from modules.estimation import ModelEstimator
    from modules.explainability import ExplainabilityAnalyzer
    from modules.report_gen import ReportGenerator
    from modules.ai_summarizer import AISummarizer
    
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(
        description='DataPrepX v1.1 - AI-Enabled Data Processing and ML Pipeline'
    )
    
    parser.add_argument('--input', '-i', type=str, default='dataset.csv')
    parser.add_argument('--target', '-t', type=str, default=None)
    parser.add_argument('--task', choices=['auto', 'classification', 'regression'], default='auto')
    parser.add_argument('--config', '-c', type=str, default='config/default_config.yaml')
    parser.add_argument('--output', '-o', type=str, default='output')
    parser.add_argument('--report-format', choices=['pdf', 'docx', 'both'], default='pdf')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--explain', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(10)
    
    logger.info("=" * 70)
    logger.info("DataPrepX v1.1 - Starting Pipeline")
    logger.info("=" * 70)
    
    try:
        config = load_config(args.config)
        
        if args.tune:
            config['estimation']['enable_hyperparameter_tuning'] = True
        
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading data from {args.input}...")
        df_original = load_data(args.input)
        
        if args.target is None:
            args.target = df_original.columns[-1]
            logger.info(f"Auto-detected target column: {args.target}")
        
        if args.target not in df_original.columns:
            available_columns = ', '.join(df_original.columns.tolist())
            raise ValueError(
                f"Target column '{args.target}' not found.\n"
                f"Available columns: {available_columns}"
            )
        
        target_column = df_original[args.target].copy()
        df_features = df_original.drop(columns=[args.target])
        
        temp_file = Path('temp_features.csv')
        df_features.to_csv(temp_file, index=False)
        
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor(config['preprocessing'])
        df_clean, metadata = preprocessor.process(str(temp_file))
        temp_file.unlink()
        
        df_clean[args.target] = target_column.values
        logger.info(f"Preprocessed: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        
        logger.info(f"Training models (Task: {args.task})...")
        estimator = ModelEstimator(config['estimation'])
        results = estimator.fit_and_evaluate(df_clean, args.target, args.task)
        
        if results['best_model'] is None:
            logger.error("No models trained successfully")
            return 1
        
        logger.info(f"Best Model: {results['best_model']} (Score: {results['best_score']:.4f})")
        
        if args.explain and results['best_model']:
            logger.info("Generating explainability...")
            explainer = ExplainabilityAnalyzer(config.get('explainability', {}))
            explainer.analyze(
                estimator.models[results['best_model']],
                df_clean.drop(columns=[args.target]),
                df_clean[args.target],
                results['task_type']
            )
        
        logger.info("Generating AI summary...")
        ai_config = {
            'api_url': 'http://localhost:1234/v1/chat/completions',
            'model': 'openai/gpt-oss-20b'
        }
        
        summarizer = AISummarizer(ai_config)
        df_info = {
            'columns': df_clean.columns.tolist(),
            'dtypes': df_clean.dtypes.to_dict(),
            'shape': df_clean.shape
        }
        
        ai_summary = summarizer.generate_summary(df_info, metadata, results)
        logger.info("\n" + "=" * 70)
        logger.info("AI SUMMARY")
        logger.info("=" * 70)
        logger.info(ai_summary)
        logger.info("=" * 70 + "\n")
        
        results['ai_summary'] = ai_summary
        
        logger.info("Generating reports...")
        report_gen = ReportGenerator(config['report'])
        report_paths = report_gen.generate(df_clean, metadata, results, output_dir, args.report_format)
        
        logger.info("=" * 70)
        logger.info("Pipeline Complete!")
        logger.info("Generated Reports:")
        for path in report_paths:
            logger.info(f"  {path}")
        logger.info("=" * 70)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1

def main():
    if len(sys.argv) == 1:
        run_web_gui()
    else:
        exit(run_cli())

if __name__ == '__main__':
    main()