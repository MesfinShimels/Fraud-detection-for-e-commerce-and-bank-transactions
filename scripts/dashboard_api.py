"""
dashboard_api.py

This script sets up a Flask API to serve fraud detection statistics for dashboard visualization.
Endpoints:
- /fraud_stats: Returns summary statistics (total transactions, fraud cases, fraud percentage).
- /fraud_trends: Returns fraud trends over time (daily fraud counts).
"""

from flask import Flask, jsonify
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Path to the featured fraud data file.
FRAUD_DATA_PATH = "../data/Fraud_Data_Featured.csv"

def get_fraud_statistics():
    """
    Reads the fraud dataset and computes summary statistics.
    """
    df = pd.read_csv(FRAUD_DATA_PATH, parse_dates=['purchase_time'])
    total_transactions = len(df)
    total_fraud_cases = df['class'].sum()  # Assumes fraudulent transactions are marked as 1.
    fraud_percentage = (total_fraud_cases / total_transactions) * 100 if total_transactions > 0 else 0
    return {
        "total_transactions": total_transactions,
        "total_fraud_cases": int(total_fraud_cases),
        "fraud_percentage": fraud_percentage
    }

def get_fraud_trends():
    """
    Computes daily fraud counts for trend analysis.
    """
    df = pd.read_csv(FRAUD_DATA_PATH, parse_dates=['purchase_time'])
    df['date'] = df['purchase_time'].dt.date
    daily_counts = df.groupby('date')['class'].sum().reset_index()
    trends = daily_counts.to_dict(orient='records')
    return trends

@app.route('/fraud_stats', methods=['GET'])
def fraud_stats():
    try:
        stats = get_fraud_statistics()
        app.logger.info("Fraud statistics requested.")
        return jsonify(stats)
    except Exception as e:
        app.logger.error("Error getting fraud statistics: " + str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/fraud_trends', methods=['GET'])
def fraud_trends():
    try:
        trends = get_fraud_trends()
        app.logger.info("Fraud trends requested.")
        return jsonify(trends)
    except Exception as e:
        app.logger.error("Error getting fraud trends: " + str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run the dashboard API on port 5001.
    app.run(host='0.0.0.0', port=5001)
