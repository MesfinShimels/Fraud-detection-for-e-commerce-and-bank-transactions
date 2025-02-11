# Fraud Detection Project

## Overview
This project improves the detection of fraud in e-commerce and bank transactions using advanced data preprocessing, feature engineering, multiple machine learning models, explainability tools (SHAP & LIME), and real-time API deployment.

## Project Structure
- **data/**: Contains raw datasets.
- **notebooks/**: Jupyter notebooks for each task:
  - **Task1_Data_Analysis_Preprocessing.ipynb**
  - **Task2_Model_Building_Training.ipynb**
  - **Task3_Model_Explainability.ipynb**
  - **Task4_Model_Deployment_API.ipynb**
  - **Task5_Dashboard_Flask_Dash.ipynb**
- **scripts/**: Reusable Python scripts for:
  - Data preprocessing (`data_preprocessing.py`)
  - Feature engineering (`feature_engineering.py`)
  - Model training (`model_training.py`)
  - Model explainability (`model_explainability.py`)
  - Model serving API (`serve_model.py`)
  - Dashboard API (`dashboard_api.py`)
- **Dockerfile**: For containerizing the Flask API.
- **requirements.txt**: Lists required Python libraries.
- **README.md**: This documentation.

## Setup Instructions
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
