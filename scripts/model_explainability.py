"""
model_explainability.py

This script demonstrates how to interpret a trained model using:
- SHAP (Shapley Additive exPlanations) for global and local feature importance.
- LIME (Local Interpretable Model-agnostic Explanations) for explaining individual predictions.
"""

import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular

def explain_model_shap(model, X_train, X_test, feature_names):
    """
    Uses SHAP to generate a summary plot and a force plot for the model.
    """
    # Create a Kernel SHAP explainer (works for any model with a predict function)
    explainer = shap.KernelExplainer(model.predict, X_train)
    shap_values = explainer.shap_values(X_test, nsamples=100)
    
    # Generate summary plot for feature importance
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
    
    # Generate a force plot for the first prediction in X_test
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], feature_names=feature_names, matplotlib=True)
    plt.show()
    
def explain_model_lime(model, X_train, X_test, feature_names):
    """
    Uses LIME to explain a single prediction.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=['Non-Fraud', 'Fraud'],
        mode='classification'
    )
    
    # Explain the first instance in the test set.
    i = 0
    exp = explainer.explain_instance(X_test.iloc[i], model.predict_proba, num_features=5)
    print("LIME explanation for the first instance:")
    print(exp.as_list())
    exp.as_pyplot_figure()
    
if __name__ == "__main__":
    # Dummy example using iris dataset to demonstrate functionality.
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    feature_names = iris.feature_names
    explain_model_shap(model, X_train, X_test, feature_names)
    explain_model_lime(model, X_train, X_test, feature_names)
