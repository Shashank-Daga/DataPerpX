.PHONY: help install test clean run-examples generate-data lint format run quick-start

help:
	@echo "DataPrepX - Available Commands"
	@echo "================================"
	@echo "make install        - Install dependencies"
	@echo "make install-dev    - Install dev dependencies"
	@echo "make test           - Run all tests"
	@echo "make test-cov       - Run tests with coverage"
	@echo "make lint           - Run code linting"
	@echo "make format         - Format code"
	@echo "make clean          - Clean generated files"
	@echo "make generate-data  - Generate sample datasets"
	@echo "make run-examples   - Run all examples"
	@echo "make run            - Launch the Streamlit app"
	@echo "make quick-start    - Generate sample data, then launch the app"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=modules --cov-report=html --cov-report=term

test-unit:
	python -m pytest tests/ -m unit -v

test-integration:
	python -m pytest tests/ -m integration -v

lint:
	flake8 modules/ app.py --max-line-length=120
	mypy modules/ --ignore-missing-imports

format:
	black modules/ app.py tests/ --line-length=120

clean:
	rm -rf output/*
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf **/__pycache__/
	rm -rf **/*.pyc
	rm -rf .coverage
	rm -rf *.egg-info/
	rm -rf dist/
	rm -rf build/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

generate-data:
	python data/generate_sample_data.py

run-examples:
	python examples/example_usage.py --all

quick-start: generate-data
	@echo "Launching DataPrepX web app..."
	@echo "Upload data/loan_approval.csv and select 'loan_approved' as the target column."
	streamlit run app.py

run:
	streamlit run app.py

setup: install generate-data
	@echo "✓ Setup complete! Ready to use DataPrepX"
	@echo "Try: make quick-start"