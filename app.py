import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import yaml
import json
import time
from datetime import datetime
import io

# Import your existing modules
from modules.utils import setup_logging, load_config, detect_column_types
from modules.preprocess import DataPreprocessor
from modules.estimation import ModelEstimator
from modules.explainability import ExplainabilityAnalyzer
from modules.report_gen import ReportGenerator
from modules.ai_summarizer import AISummarizer  # AI Integration

# Page configuration
st.set_page_config(
    page_title="DataPrepX - AI-Powered ML Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stAlert {
        border-radius: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        font-size: 1rem;
        font-weight: 600;
    }
    .ai-summary-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'ai_summary' not in st.session_state:
    st.session_state.ai_summary = None

# Header
st.markdown('<h1 class="main-header">🤖 DataPrepX</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Enabled Data Processing & Machine Learning Pipeline</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
    st.title("⚙️ Configuration")

    # File upload
    st.subheader("📁 Data Upload")
    uploaded_file = st.file_uploader(
        "Drop your CSV or XLSX file here",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your dataset for processing"
    )

    if uploaded_file:
        st.success(f"✅ Loaded: {uploaded_file.name}")

        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.original_data = df

            st.metric("Rows", f"{len(df):,}")
            st.metric("Columns", f"{len(df.columns):,}")
        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.divider()

    # Configuration options
    st.subheader("🎯 Model Configuration")

    if st.session_state.original_data is not None:
        target_column = st.selectbox(
            "Target Column",
            options=st.session_state.original_data.columns.tolist(),
            help="Select the column you want to predict"
        )
    else:
        target_column = st.text_input("Target Column", "")

    task_type = st.selectbox(
        "Task Type",
        options=['auto', 'classification', 'regression'],
        help="Auto-detect or manually specify the task type"
    )

    st.divider()

    # AI Configuration Section
    st.subheader("🤖 AI Configuration")
    
    with st.expander("AI Settings", expanded=False):
        ai_api_url = st.text_input(
            "AI API URL",
            value="http://localhost:1234/v1/chat/completions",
            help="LLM Studio API endpoint"
        )
        
        ai_model = st.text_input(
            "Model Name",
            value="openai/gpt-oss-20b",
            help="Model identifier for LLM Studio"
        )
        
        enable_ai_summary = st.checkbox("Generate AI Summary", value=True)
        enable_ai_insights = st.checkbox("Generate AI Insights", value=True)

    st.divider()

    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        enable_explain = st.checkbox("Generate Explainability Analysis", True)
        parallel_training = st.checkbox("Parallel Model Training", True)

        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)

        report_format = st.selectbox(
            "Report Format",
            options=['pdf', 'docx', 'both']
        )

    st.divider()

    # Process button
    process_button = st.button("🚀 Start Processing", type="primary", use_container_width=True)

# Main content area
if st.session_state.original_data is None and not uploaded_file:
    # Welcome screen
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📊 Data Preprocessing
        - Handle missing values
        - Remove duplicates
        - Detect & cap outliers
        - Feature scaling
        - Auto encoding
        """)

    with col2:
        st.markdown("""
        ### 🤖 ML Training
        - Auto model selection
        - Multiple algorithms
        - Cross-validation
        - Hyperparameter tuning
        - Performance metrics
        """)

    with col3:
        st.markdown("""
        ### 📈 Explainability
        - SHAP analysis
        - LIME explanations
        - Feature importance
        - Model insights
        - Visual reports
        """)

    st.info("👆 Upload a dataset from the sidebar to get started!")

    # Sample data section
    st.subheader("📦 Or Try Sample Datasets")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💳 Loan Approval Dataset", use_container_width=True):
            st.info("Load sample data: data/loan_approval.csv")

    with col2:
        if st.button("🏠 Housing Prices Dataset", use_container_width=True):
            st.info("Load sample data: data/housing_prices.csv")

    with col3:
        if st.button("📈 Sales Timeseries Dataset", use_container_width=True):
            st.info("Load sample data: data/sales_timeseries.csv")

elif st.session_state.original_data is not None:
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Explorer",
        "🔄 Processing",
        "🤖 AI Insights",
        "📈 Model Results",
        "💡 Explainability",
        "📄 Reports"
    ])

    # Tab 1: Data Explorer
    with tab1:
        st.subheader("📊 Dataset Overview")

        df = st.session_state.original_data

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", f"{len(df.columns):,}")
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        with col4:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
            st.metric("Memory Usage", f"{memory_mb:.2f} MB")

        st.divider()

        # Data preview
        st.subheader("🔍 Data Preview")
        st.dataframe(df.head(100), use_container_width=True, height=400)

        # Column statistics
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Numerical Columns")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("No numerical columns found")

        with col2:
            st.subheader("📝 Categorical Columns")
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                cat_summary = pd.DataFrame({
                    'Column': cat_cols,
                    'Unique Values': [df[col].nunique() for col in cat_cols],
                    'Most Common': [df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A' for col in cat_cols]
                })
                st.dataframe(cat_summary, use_container_width=True)
            else:
                st.info("No categorical columns found")

        st.divider()

        # Visualizations
        st.subheader("📊 Data Visualizations")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Missing values heatmap
            if df.isnull().sum().sum() > 0:
                missing_data = df.isnull().sum()
                missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

                fig = px.bar(
                    x=missing_data.values,
                    y=missing_data.index,
                    orientation='h',
                    title="Missing Values by Column",
                    labels={'x': 'Count', 'y': 'Column'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values in dataset!")

        with viz_col2:
            # Correlation heatmap
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    title="Correlation Matrix",
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Distribution plots
        if numeric_cols:
            st.subheader("📈 Distribution Plots")
            selected_col = st.selectbox("Select column to visualize:", numeric_cols)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)

    # Tab 2: Processing
    with tab2:
        if process_button and target_column:
            with st.spinner("🔄 Processing your data..."):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    # Load config
                    status_text.text("Loading configuration...")
                    progress_bar.progress(10)
                    config = load_config('config/default_config.yaml')

                    # Update config with user selections
                    config['estimation']['test_size'] = test_size
                    config['estimation']['cv_folds'] = cv_folds
                    config['estimation']['parallel_training'] = parallel_training

                    # Preprocessing
                    status_text.text("🔄 Preprocessing data...")
                    progress_bar.progress(30)
                    time.sleep(0.5)

                    # Save original data temporarily
                    target_data = df[target_column].copy()
                    df_features = df.drop(columns=[target_column])

                    temp_file = Path('temp_upload.csv')
                    df_features.to_csv(temp_file, index=False)

                    preprocessor = DataPreprocessor(config['preprocessing'])
                    df_clean, metadata = preprocessor.process(str(temp_file))

                    temp_file.unlink()
                    df_clean[target_column] = target_data.values

                    st.session_state.processed_data = df_clean
                    st.session_state.metadata = metadata

                    # Model Training
                    status_text.text("🤖 Training ML models...")
                    progress_bar.progress(60)
                    time.sleep(0.5)

                    estimator = ModelEstimator(config['estimation'])
                    results = estimator.fit_and_evaluate(df_clean, target_column, task_type)
                    st.session_state.results = results
                    st.session_state.estimator = estimator

                    # Explainability
                    if enable_explain and results['best_model']:
                        status_text.text("💡 Generating explainability analysis...")
                        progress_bar.progress(75)
                        time.sleep(0.5)

                        explainer = ExplainabilityAnalyzer(config.get('explainability', {}))
                        explainability_results = explainer.analyze(
                            estimator.models[results['best_model']],
                            df_clean.drop(columns=[target_column]),
                            df_clean[target_column],
                            results['task_type']
                        )
                        st.session_state.explainability = explainability_results

                    # AI Summary Generation
                    if enable_ai_summary:
                        status_text.text("🤖 Generating AI summary...")
                        progress_bar.progress(85)
                        time.sleep(0.5)

                        try:
                            ai_config = {
                                'api_url': ai_api_url,
                                'model': ai_model
                            }
                            
                            summarizer = AISummarizer(ai_config)
                            df_info = {
                                'columns': df_clean.columns.tolist(),
                                'dtypes': df_clean.dtypes.to_dict(),
                                'shape': df_clean.shape
                            }
                            
                            ai_summary = summarizer.generate_summary(df_info, metadata, results)
                            st.session_state.ai_summary = ai_summary
                            results['ai_summary'] = ai_summary
                            
                            # Generate additional AI insights if enabled
                            if enable_ai_insights:
                                data_quality_report = summarizer.generate_data_quality_report(df_clean, metadata)
                                model_insights = summarizer.generate_model_insights(results)
                                business_recommendations = summarizer.generate_business_recommendations(
                                    df_info, metadata, results
                                )
                                
                                st.session_state.ai_insights = {
                                    'data_quality': data_quality_report,
                                    'model_insights': model_insights,
                                    'business_recommendations': business_recommendations
                                }
                        except Exception as e:
                            st.warning(f"⚠️ AI summary generation failed: {str(e)}")
                            st.info("💡 Make sure LLM Studio is running at the configured URL")

                    # Generate Reports
                    status_text.text("📄 Generating reports...")
                    progress_bar.progress(95)
                    time.sleep(0.5)

                    output_dir = Path('output')
                    output_dir.mkdir(exist_ok=True)

                    report_gen = ReportGenerator(config['report'])
                    report_paths = report_gen.generate(
                        df_clean, metadata, results,
                        output_dir, report_format
                    )
                    st.session_state.report_paths = report_paths

                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    st.session_state.processing_complete = True

                    time.sleep(1)
                    st.success("🎉 Pipeline completed successfully!")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)

        if st.session_state.processing_complete:
            st.success("✅ Data processing completed successfully!")

            metadata = st.session_state.metadata

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Original Rows",
                    f"{metadata['original_shape'][0]:,}",
                    delta=f"{metadata['final_shape'][0] - metadata['original_shape'][0]:,}"
                )

            with col2:
                st.metric(
                    "Final Columns",
                    f"{metadata['final_shape'][1]:,}",
                    delta=f"{metadata['final_shape'][1] - metadata['original_shape'][1]:,}"
                )

            with col3:
                st.metric(
                    "Duplicates Removed",
                    f"{metadata.get('duplicates_removed', 0):,}"
                )

            st.divider()

            # Preprocessing details
            with st.expander("🔍 View Preprocessing Details"):
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Missing Values Handled")
                    if metadata.get('missing_values'):
                        missing_df = pd.DataFrame(
                            list(metadata['missing_values'].items()),
                            columns=['Column', 'Missing Count']
                        )
                        st.dataframe(missing_df, use_container_width=True)
                    else:
                        st.info("No missing values found")

                with col2:
                    st.subheader("Encoding Applied")
                    if metadata.get('encoding_map'):
                        encoding_df = pd.DataFrame(
                            list(metadata['encoding_map'].items()),
                            columns=['Column', 'Encoding Type']
                        )
                        st.dataframe(encoding_df, use_container_width=True)
                    else:
                        st.info("No encoding applied")
        else:
            st.info("👆 Click 'Start Processing' in the sidebar to begin the ML pipeline")

    # Tab 3: AI Insights (NEW)
    with tab3:
        if st.session_state.ai_summary:
            st.subheader("🤖 AI-Generated Executive Summary")
            
            st.markdown(f"""
            <div class="ai-summary-box">
                {st.session_state.ai_summary.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # Additional AI Insights if available
            if hasattr(st.session_state, 'ai_insights') and st.session_state.ai_insights:
                insights = st.session_state.ai_insights
                
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("📊 Data Quality Report", expanded=True):
                        st.markdown(insights['data_quality'])
                
                with col2:
                    with st.expander("🎯 Model Insights", expanded=True):
                        st.markdown(insights['model_insights'])
                
                st.divider()
                
                with st.expander("💼 Business Recommendations", expanded=True):
                    st.markdown(insights['business_recommendations'])
        else:
            st.info("👆 Process your data with AI enabled to see AI-generated insights")
            
            st.markdown("""
            ### 🤖 AI-Powered Analysis
            
            When you enable AI analysis, you'll get:
            
            - **Executive Summary**: High-level overview of your data and model performance
            - **Data Quality Report**: Detailed assessment of data completeness and integrity
            - **Model Insights**: AI interpretation of model performance and feature importance
            - **Business Recommendations**: Actionable insights for stakeholders
            
            **Setup Requirements:**
            1. Install and run LLM Studio on your local machine
            2. Configure the API URL in the sidebar (default: http://localhost:1234)
            3. Select your preferred model
            4. Enable AI summary generation
            """)

    # Tab 4: Model Results
    with tab4:
        if st.session_state.results:
            results = st.session_state.results

            # Best model highlight
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 10px; color: white; text-align: center;'>
                <h2>🏆 Best Model: {results['best_model']}</h2>
                <h1>{results['best_score']:.4f}</h1>
                <p>Task: {results['task_type'].title()}</p>
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # Model comparison
            st.subheader("📊 Model Comparison")

            model_data = []
            for model_name, metrics in results['models'].items():
                if results['task_type'] == 'classification':
                    score = metrics.get('accuracy', 0)
                    metric_name = 'Accuracy'
                else:
                    score = metrics.get('r2_score', 0)
                    metric_name = 'R² Score'

                model_data.append({
                    'Model': model_name,
                    metric_name: score,
                    'CV Mean': metrics.get('cv_mean', 0),
                    'CV Std': metrics.get('cv_std', 0)
                })

            model_df = pd.DataFrame(model_data)

            # Bar chart
            fig = px.bar(
                model_df,
                x='Model',
                y=metric_name,
                title=f"Model Performance Comparison ({metric_name})",
                color=metric_name,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Detailed metrics table
            st.dataframe(model_df, use_container_width=True)

            st.divider()

            # Feature importance
            if results.get('feature_importance'):
                st.subheader("🎯 Feature Importance")

                importance_data = results['feature_importance']
                importance_df = pd.DataFrame(
                    list(importance_data.items()),
                    columns=['Feature', 'Importance']
                ).sort_values('Importance', ascending=False)

                fig = px.bar(
                    importance_df.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 15 Most Important Features',
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)

            # Detailed metrics per model
            with st.expander("📈 Detailed Metrics for All Models"):
                for model_name, metrics in results['models'].items():
                    st.subheader(f"**{model_name}**")

                    metrics_cols = st.columns(len(metrics) - 2)
                    idx = 0
                    for key, value in metrics.items():
                        if key not in ['cv_mean', 'cv_std']:
                            with metrics_cols[idx]:
                                st.metric(key.replace('_', ' ').title(), f"{value:.4f}")
                            idx += 1

                    st.divider()
        else:
            st.info("👆 Process your data first to see model results")

    # Tab 5: Explainability
    with tab5:
        if hasattr(st.session_state, 'explainability') and st.session_state.explainability:
            explain_results = st.session_state.explainability

            st.subheader("💡 Model Explainability Analysis")

            # Global importance
            if explain_results.get('global_importance'):
                st.markdown("### 🌍 Global Feature Importance")

                importance_data = explain_results['global_importance']
                importance_df = pd.DataFrame(
                    list(importance_data.items()),
                    columns=['Feature', 'SHAP Value']
                ).sort_values('SHAP Value', ascending=False).head(20)

                fig = px.bar(
                    importance_df,
                    x='SHAP Value',
                    y='Feature',
                    orientation='h',
                    title='SHAP Feature Importance (Top 20)',
                    color='SHAP Value',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # LIME explanations
            if explain_results.get('lime_analysis') and explain_results['lime_analysis'].get('explanations'):
                st.markdown("### 🔍 LIME Local Explanations")

                explanations = explain_results['lime_analysis']['explanations']

                for i, exp in enumerate(explanations):
                    with st.expander(
                            f"Instance {i + 1} - Prediction: {exp['prediction']:.4f} | Actual: {exp['actual']:.4f}"):
                        contrib_df = pd.DataFrame(
                            list(exp['feature_contributions'].items()),
                            columns=['Feature', 'Contribution']
                        ).sort_values('Contribution', key=lambda x: abs(x), ascending=False)

                        fig = px.bar(
                            contrib_df.head(10),
                            x='Contribution',
                            y='Feature',
                            orientation='h',
                            title='Top 10 Feature Contributions',
                            color='Contribution',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # Interpretation guide
            with st.expander("ℹ️ How to Interpret These Results"):
                st.markdown("""
                **SHAP (SHapley Additive exPlanations)**
                - Shows how much each feature contributes to the model's predictions
                - Positive values increase the prediction, negative values decrease it
                - Larger absolute values indicate more important features

                **LIME (Local Interpretable Model-agnostic Explanations)**
                - Explains individual predictions by approximating the model locally
                - Shows which features pushed the prediction up or down for specific instances
                - Helps understand why the model made a particular prediction
                """)
        else:
            st.info("👆 Enable explainability analysis in the sidebar and process your data to see insights")

    # Tab 6: Reports
    with tab6:
        if hasattr(st.session_state, 'report_paths') and st.session_state.report_paths:
            st.subheader("📄 Generated Reports")

            st.success("✅ Reports have been generated successfully!")

            for report_path in st.session_state.report_paths:
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.text(f"📄 {report_path.name}")

                with col2:
                    file_size = report_path.stat().st_size / 1024
                    st.text(f"{file_size:.1f} KB")

                with col3:
                    with open(report_path, "rb") as file:
                        st.download_button(
                            label="Download",
                            data=file,
                            file_name=report_path.name,
                            mime="application/octet-stream",
                            use_container_width=True
                        )

            st.divider()

            # Report preview info
            st.info("""
            📋 **Report Contents:**
            - Executive Summary (including AI-generated insights)
            - Data Overview & Statistics
            - Preprocessing Details
            - Model Performance Metrics
            - Feature Importance Analysis
            - Visualizations & Charts
            - AI-Generated Business Recommendations
            """)

        else:
            st.info("👆 Process your data first to generate reports")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>DataPrepX v1.1 : Enhance With AI</p>
    <p>©️ 2025 DataPrepX Team</p>
</div>
""", unsafe_allow_html=True)
