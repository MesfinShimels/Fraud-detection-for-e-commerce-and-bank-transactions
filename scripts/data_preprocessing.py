"""
data_preprocessing.py

This script contains functions for:
- Cleaning transaction data (handling missing values, converting date columns, removing duplicates)
- Converting IP addresses to integers
- Merging e-commerce transaction data with IP-to-country mapping for geolocation analysis.
"""

import pandas as pd
import numpy as np

def clean_fraud_data(fraud_df):
    """
    Clean the fraud data by:
    - Removing duplicates.
    - Converting 'signup_time' and 'purchase_time' columns to datetime objects.
    - Dropping rows missing critical values (user_id, purchase_time, purchase_value).
    """
    fraud_df = fraud_df.drop_duplicates()

    # Convert time columns with error coercion (invalid parsing will become NaT)
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'], errors='coerce')
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'], errors='coerce')

    # Drop rows where critical values are missing
    fraud_df = fraud_df.dropna(subset=['user_id', 'purchase_time', 'purchase_value'])
    
    return fraud_df

def clean_creditcard_data(creditcard_df):
    """
    Clean the credit card dataset by removing duplicates and rows with missing values.
    """
    creditcard_df = creditcard_df.drop_duplicates()
    creditcard_df = creditcard_df.dropna()
    return creditcard_df

def ip_to_int(ip):
    """
    Converts an IP address string (e.g., '192.168.1.1') into its integer representation.
    If conversion fails, returns np.nan.
    """
    try:
        parts = ip.split('.')
        return int(parts[0]) * (256**3) + int(parts[1]) * (256**2) + int(parts[2]) * 256 + int(parts[3])
    except Exception as e:
        return np.nan

def merge_ip_data(fraud_df, ip_mapping_df):
    """
    Merges the fraud transaction data with the IP mapping data.
    Converts IP addresses to integers and then determines the country by checking
    which range (lower_bound to upper_bound) the IP falls in.
    """
    # Convert the IP addresses in fraud data to integer format.
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)

    # Convert lower and upper IP bounds in mapping data to integer.
    ip_mapping_df['lower_bound'] = ip_mapping_df['lower_bound_ip_address'].apply(ip_to_int)
    ip_mapping_df['upper_bound'] = ip_mapping_df['upper_bound_ip_address'].apply(ip_to_int)

    # Define a helper function to lookup country for a given integer IP.
    def get_country(ip_int):
        match = ip_mapping_df[(ip_mapping_df['lower_bound'] <= ip_int) &
                              (ip_mapping_df['upper_bound'] >= ip_int)]
        if not match.empty:
            return match.iloc[0]['country']
        else:
            return 'Unknown'
    
    fraud_df['country'] = fraud_df['ip_int'].apply(get_country)
    return fraud_df

# For testing purpose, you can run this script directly.
if __name__ == "__main__":
    fraud_file = "../data/Fraud_Data.csv"
    ip_file = "../data/IpAddress_to_Country.csv"
    
    fraud_df = pd.read_csv(fraud_file)
    ip_mapping_df = pd.read_csv(ip_file)
    
    fraud_df = clean_fraud_data(fraud_df)
    fraud_df = merge_ip_data(fraud_df, ip_mapping_df)
    
    # Save the preprocessed file
    fraud_df.to_csv("../data/Fraud_Data_Preprocessed.csv", index=False)
    print("Fraud data preprocessing complete.")
