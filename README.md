# AI Chatbot Paper - Python Code Repository

This repository contains all Python code and Jupyter notebooks for the AI Chatbot research paper.

## Repository Structure

```
├── config/               # Configuration files
│   └── data_config.py   # Data processing configuration
├── notebooks/           # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_results_visualization.ipynb
├── scripts/             # Python scripts
│   ├── automated_tagging.py      # Automated data tagging
│   ├── combine_all_datasets.py   # Dataset combination utilities
│   ├── data_preprocessing.py     # Data preprocessing pipeline
│   ├── data_validation.py        # Data validation tools
│   └── run_phase2_pipeline.py    # Main pipeline runner
└── requirements.txt     # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nouf/ai-chatbot-paper-python.git
cd ai-chatbot-paper-python
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Pipeline

```bash
python scripts/run_phase2_pipeline.py
```

### Individual Scripts

- **Data Preprocessing**: `python scripts/data_preprocessing.py`
- **Data Validation**: `python scripts/data_validation.py`
- **Automated Tagging**: `python scripts/automated_tagging.py`
- **Combine Datasets**: `python scripts/combine_all_datasets.py`

### Jupyter Notebooks

Launch Jupyter to explore the notebooks:
```bash
jupyter notebook
```

## Data Processing Pipeline

The pipeline consists of several phases:

1. **Data Collection**: Gather phishing and legitimate email datasets
2. **Preprocessing**: Clean and standardize data formats
3. **Feature Engineering**: Extract relevant features for ML models
4. **Model Training**: Train various ML models for classification
5. **Evaluation**: Assess model performance and generate results

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for complete list of dependencies

## License

This project is part of academic research. Please cite appropriately if using this code.

## Contact

For questions or issues, please contact the research team.