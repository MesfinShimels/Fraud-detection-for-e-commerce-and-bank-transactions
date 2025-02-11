"""
model_training.py

This script contains functions to:
- Prepare the dataset (feature and target separation, train-test split)
- Train multiple machine learning models:
    * Logistic Regression
    * Decision Tree
    * Random Forest
    * Gradient Boosting
    * Multi-Layer Perceptron (MLP)
- Evaluate model performance using standard metrics.
- Log experiments with MLflow.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import mlflow
import mlflow.sklearn

def prepare_data(df, target_col):
    """
    Separates features and target variable and splits data into training and testing sets.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_mlp(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints accuracy, confusion matrix, and classification report.
    """
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return y_pred

if __name__ == "__main__":
    # Example usage for Fraud_Data
    fraud_file = "../data/Fraud_Data_Featured.csv"
    # For this example, we assume 'class' is the target column in Fraud_Data.
    df_fraud = pd.read_csv(fraud_file, parse_dates=['purchase_time'])
    
    # Select a subset of features for demo purposes.
    # Make sure these columns exist and are numeric; in a real case, encode categorical features properly.
    selected_columns = ['purchase_value', 'age', 'hour_of_day', 'day_of_week', 'transaction_count', 'time_diff', 'class']
    df_fraud = df_fraud[selected_columns].dropna()
    
    X_train, X_test, y_train, y_test = prepare_data(df_fraud, 'class')
    
    # Set up MLflow experiment logging
    mlflow.set_experiment("Fraud_Detection_Fraud_Data")
    
    models = {
        "LogisticRegression": train_logistic_regression,
        "DecisionTree": train_decision_tree,
        "RandomForest": train_random_forest,
        "GradientBoosting": train_gradient_boosting,
        "MLP": train_mlp
    }
    
    for model_name, train_func in models.items():
        with mlflow.start_run(run_name=model_name):
            model = train_func(X_train, y_train)
            evaluate_model(model, X_test, y_test)
            mlflow.sklearn.log_model(model, model_name)
            mlflow.log_params({"model": model_name})
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("accuracy", accuracy)
            print(f"{model_name} training and evaluation complete.")
    
    # Optionally, save one model for deployment (here we save the RandomForest model)
    import pickle
    with open("../scripts/trained_model.pkl", "wb") as f:
        pickle.dump(models["RandomForest"](X_train, y_train), f)
    print("Random Forest model saved as trained_model.pkl")
