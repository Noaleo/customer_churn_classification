# Churn Prediction using XGBoost

### Overview

    This project aims to predict customer churn using time-series data and machine learning techniques. 
    The model is trained on transactional data and user behavior, leveraging feature engineering, categorical encoding,
    and hyperparameter tuning. 
    The pipeline automates data processing, model training, evaluation, and prediction generation.
    
 
### Repository Structure  
    |-- data/                      # Directory for raw and processed data
    |   |-- churn_data.csv         # Input dataset
    |-- output/                    # Directory for model outputs
    |   |-- model_1M.pkl           # Trained model for 1-month prediction
    |   |-- model_2M.pkl           # Trained model for 2-month prediction
    |   |-- predictions.csv        # Final predictions
    |   |-- metrics_1M.json        # Model evaluation metrics for 1M shift
    |   |-- metrics_2M.json        # Model evaluation metrics for 2M shift
    |-- atidot_task.py                    # Main script to execute the pipeline
    |-- README.md    
    
### Setup

#### Prerequisites

    Ensure you have Python 3 installed along with the following dependencies:
    
    pip install -r requirements.txt
    
### Usage

    Run the following command to train and evaluate the churn prediction model:

    python atidot_task.py

    By default, the script will:
    
    1. Load and preprocess the dataset.
    
    2. Engineer features and encode categorical variables.
    
    3. Split the data into training, validation, and test sets.
    
    4. Train an XGBoost model using GridSearchCV for hyperparameter tuning.
    
    5. Evaluate model performance and save the results.
    
    6. Generate final churn predictions.
    
    - To run the pipeline for a specific shift window (e.g., predicting churn 2 months ahead), use:
    
        python main.py --shift 2

### Output

    1. Trained Model: Saved as a .pkl file in the output/ directory.
    
    2. Evaluation Metrics: Precision & recall metrics are saved in output/metrics_2M.json file.
    
    3. Predictions: The final predictions for customer churn are stored in output/predictions.csv.

#### Author

    Developed by Noa Leonard for a data science task focused on customer churn analysis using machine learning.

