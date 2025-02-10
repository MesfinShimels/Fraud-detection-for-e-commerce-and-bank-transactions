"""
feature_engineering.py

This script includes functions to engineer additional features for fraud detection:
- Time-based features (hour of day, day of week)
- Transaction frequency per user
- Transaction velocity (time difference between consecutive transactions)
"""

import pandas as pd

def add_time_features(fraud_df):
    """
    Adds new time-based features:
    - hour_of_day: the hour when the purchase was made.
    - day_of_week: day of week (0=Monday, 6=Sunday).
    """
    fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
    fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
    return fraud_df

def add_transaction_frequency(fraud_df):
    """
    Adds a 'transaction_count' feature indicating the total number of transactions per user.
    """
    transaction_counts = fraud_df.groupby('user_id').size().reset_index(name='transaction_count')
    fraud_df = fraud_df.merge(transaction_counts, on='user_id', how='left')
    return fraud_df

def add_transaction_velocity(fraud_df):
    """
    Computes transaction velocity as the time difference (in seconds) between consecutive transactions for each user.
    """
    fraud_df = fraud_df.sort_values(by=['user_id', 'purchase_time'])
    fraud_df['time_diff'] = fraud_df.groupby('user_id')['purchase_time'].diff().dt.total_seconds()
    fraud_df['time_diff'] = fraud_df['time_diff'].fillna(99999)  # Use a large number for the first transaction.
    return fraud_df

# For testing purpose, run directly.
if __name__ == "__main__":
    import pandas as pd
    fraud_file = "../data/Fraud_Data_Preprocessed.csv"
    fraud_df = pd.read_csv(fraud_file, parse_dates=['purchase_time'])
    
    fraud_df = add_time_features(fraud_df)
    fraud_df = add_transaction_frequency(fraud_df)
    fraud_df = add_transaction_velocity(fraud_df)
    
    fraud_df.to_csv("../data/Fraud_Data_Featured.csv", index=False)
    print("Feature engineering complete.")
