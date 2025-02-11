"""
serve_model.py

This script sets up a Flask API to serve the trained fraud detection model.
Endpoint:
- /predict: Accepts POST requests with JSON input and returns model predictions.
"""

from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging

# Initialize the Flask application and logging.
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load the trained model (ensure that the model file exists in the specified path).
MODEL_PATH = "trained_model.pkl"

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    app.logger.info("Model loaded successfully.")
except Exception as e:
    app.logger.error("Error loading model: " + str(e))
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON in the following format:
    {
        "data": [
            [feature1, feature2, ..., featureN],
            [feature1, feature2, ..., featureN]
        ]
    }
    Returns a JSON response with predictions.
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json(force=True)
        input_data = data.get("data")
        if input_data is None:
            return jsonify({"error": "No data provided"}), 400
        
        # Convert input list to numpy array for prediction.
        input_array = np.array(input_data)
        predictions = model.predict(input_array)
        predictions_list = predictions.tolist()
        
        app.logger.info("Prediction made for input: {}".format(input_data))
        return jsonify({"predictions": predictions_list})
    except Exception as e:
        app.logger.error("Error during prediction: " + str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the Flask API on host 0.0.0.0 and port 5000.
    app.run(host='0.0.0.0', port=5000)
