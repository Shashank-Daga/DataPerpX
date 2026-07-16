**Overview**


DataPrepX is an advanced AI-based data preparation tool designed to simplify and accelerate the process of transforming raw data into clean, structured, and analytics-ready datasets. It combines traditional data processing techniques with intelligent automation to make data preprocessing efficient, reproducible, and extensible for both analysts and engineers.

Key capabilities include:


- Automated detection of data quality issues (missing values, inconsistent types, outliers)
- Intelligent suggestions for preprocessing operations
- Modular transformation pipeline support
- Integration with machine learning workflows

**Features**


- AI-Assisted Data Processing: Core engine that uses intelligent heuristics (and optionally ML models) to recommend or perform data preparation tasks
- Modular Design: Organized into clearly defined modules for loading, cleaning, transforming, and exporting data.
- Flexible Pipelines: Build custom preprocessing pipelines that can be reused across projects.
- Example Scripts: Ready-to-use examples demonstrating typical workflows.
- Unit Tests: Automated tests to ensure reliability and correctness.


**Project FLow Diagram**
<p align = "center">
	<img width="570" height="980" alt="image" src="https://github.com/user-attachments/assets/2fdec84a-0502-4e3d-afae-3b29eab660a7" />
</p>

**UI Screenshots**
<p align = "center">
	<img width="1499" height="855" alt="image" src="https://github.com/user-attachments/assets/b7faf5d9-586e-4418-80e0-02efdf718ee8" />
</p>

<p align = "center">
<img width="1499" height="855" alt="image" src="https://github.com/user-attachments/assets/a3fa13a8-d6f3-4d7f-8752-59ff1d496e30" />
</p>

**Configuration**

DataPrepX uses configuration files to define preprocessing pipelines and settings. See the config/ directory for examples and templates. You can define pipelines in YAML or JSON formats that describe the sequence of operations to apply to a dataset.

