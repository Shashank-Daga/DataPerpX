**Overview**

DataPrepX is an advanced AI-based data preparation tool designed to simplify and accelerate the process of transforming raw data into clean, structured, and analytics-ready datasets. It combines traditional data processing techniques with intelligent automation to make data preprocessing efficient, reproducible, and extensible for both analysts and engineers.

Key capabilities include:

- Automated detection of data quality issues (missing values, inconsistent types, outliers)
- Intelligent suggestions for preprocessing operations
- Modular transformation pipeline support
- Integration with machine learning workflows
- AI-generated executive summaries and business insights via a local LLM

**Features**

- AI-Assisted Data Processing: Core engine that uses intelligent heuristics (and optionally ML models) to recommend or perform data preparation tasks
- Modular Design: Organized into clearly defined modules for loading, cleaning, transforming, and exporting data
- Flexible Pipelines: Build custom preprocessing pipelines that can be reused across projects
- Example Scripts: Ready-to-use examples demonstrating typical workflows
- Unit Tests: Automated tests to ensure reliability and correctness
- Explainability: SHAP and LIME-based model explanations
- Report Generation: PDF and DOCX report output

**Project Structure**

```
DataPrepX/
├── app.py                 # Streamlit web application (main entry point)
├── setup.py                # Package metadata
├── Makefile                 # Common dev commands (see `make help`)
├── config/
│   └── default_config.yaml  # Default pipeline configuration
├── data/                    # Sample datasets (generate via `make generate-data`)
├── examples/
│   └── example_usage.py     # Scripted usage examples
├── modules/
│   ├── utils.py              # Config/data I/O helpers
│   ├── preprocess.py         # Missing values, outliers, encoding, scaling
│   ├── estimation.py         # Model training and evaluation
│   ├── explainability.py     # SHAP/LIME analysis
│   ├── report_gen.py         # PDF/DOCX report generation
│   └── ai_summarizer.py      # LLM-based summaries (with template fallback)
├── output/                  # Generated reports land here
└── tests/                   # Unit tests (pytest)
```

**Running DataPrepX**

DataPrepX is a Streamlit application. To launch it:

```
make install
make generate-data   # optional: creates sample datasets in data/
make run             # streamlit run app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`) and upload a CSV or Excel file.

**Configuration**

DataPrepX uses configuration files to define preprocessing pipelines and settings. See `config/default_config.yaml` for the default pipeline configuration. Config values marked `'auto'` (e.g. `outlier_method`, `scaler`, `missing_strategy`) are resolved automatically based on simple heuristics over the input data (dataset size, skewness) rather than a fixed choice.

**AI Summaries**

DataPrepX can generate an executive summary and supplementary insights via a local, OpenAI-compatible LLM endpoint (e.g. LM Studio) configured in the app sidebar. If the endpoint isn't reachable, DataPrepX falls back to a template-based summary and clearly labels it as such in both the app and generated reports.

**Testing**

```
make test        # run the test suite
make test-cov     # run with coverage report
```