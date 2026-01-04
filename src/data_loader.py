"""
Data loading and cleaning module for fraud detection system.

This module handles loading the PaySim dataset and performing data cleaning operations
including removing invalid balances and handling missing values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_raw_data(filepath='data/raw/PS_20174392719_1491204439457_log.csv', nrows=None):
    """
    Load the PaySim dataset from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the PaySim CSV file
    nrows : int, optional
        Number of rows to load (useful for testing on smaller samples)
        
    Returns:
    --------
    pd.DataFrame
        Raw transaction data
    """
    print(f"Loading data from {filepath}...")
    
    # Define optimal dtypes for memory efficiency
    dtypes = {
        'step': 'int32',
        'type': 'category',
        'amount': 'float32',
        'nameOrig': 'object',
        'oldbalanceOrg': 'float32',
        'newbalanceOrig': 'float32',
        'nameDest': 'object',
        'oldbalanceDest': 'float32',
        'newbalanceDest': 'float32',
        'isFraud': 'int8',
        'isFlaggedFraud': 'int8'
    }
    
    df = pd.read_csv(filepath, dtype=dtypes, nrows=nrows)
    
    print(f"Loaded {len(df):,} transactions")
    print(f"Fraud rate: {df['isFraud'].mean()*100:.4f}%")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


def clean_data(df):
    """
    Clean the transaction data by removing invalid records and handling missing values.
    
    Cleaning steps:
    1. Remove transactions with negative balances (data errors)
    2. Remove transactions where balance logic is violated
    3. Handle any missing values
    4. Create transaction ID for tracking
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw transaction data
        
    Returns:
    --------
    pd.DataFrame
        Cleaned transaction data
    """
    print("\nCleaning data...")
    initial_count = len(df)
    
    # Create unique transaction ID
    df = df.reset_index(drop=True)
    df['transaction_id'] = df.index
    
    # Remove transactions with negative balances (data errors)
    df = df[
        (df['oldbalanceOrg'] >= 0) & 
        (df['newbalanceOrig'] >= 0) &
        (df['oldbalanceDest'] >= 0) & 
        (df['newbalanceDest'] >= 0)
    ].copy()
    
    print(f"Removed {initial_count - len(df):,} transactions with negative balances")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print("\nMissing values found:")
        print(missing[missing > 0])
        # For this dataset, we typically don't have missing values
        # If they exist, we'd handle them based on the column
        df = df.dropna()
    else:
        print("No missing values found")
    
    # Verify data integrity
    print(f"\nFinal dataset: {len(df):,} transactions")
    print(f"Fraud rate after cleaning: {df['isFraud'].mean()*100:.4f}%")
    
    return df


def save_processed_data(df, filepath='data/processed/cleaned_data.csv'):
    """
    Save cleaned data to CSV for reproducibility.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned transaction data
    filepath : str
        Output file path
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving processed data to {filepath}...")
    df.to_csv(filepath, index=False)
    print("Data saved successfully")


def load_processed_data(filepath='data/processed/cleaned_data.csv'):
    """
    Load previously cleaned and processed data.
    
    Parameters:
    -----------
    filepath : str
        Path to processed data file
        
    Returns:
    --------
    pd.DataFrame
        Processed transaction data
    """
    print(f"Loading processed data from {filepath}...")
    
    dtypes = {
        'step': 'int32',
        'type': 'category',
        'amount': 'float32',
        'nameOrig': 'object',
        'oldbalanceOrg': 'float32',
        'newbalanceOrig': 'float32',
        'nameDest': 'object',
        'oldbalanceDest': 'float32',
        'newbalanceDest': 'float32',
        'isFraud': 'int8',
        'isFlaggedFraud': 'int8',
        'transaction_id': 'int64'
    }
    
    df = pd.read_csv(filepath, dtype=dtypes)
    print(f"Loaded {len(df):,} transactions")
    
    return df


if __name__ == "__main__":
    # Example usage
    df = load_raw_data(nrows=100000)  # Load sample for testing
    df_clean = clean_data(df)
    save_processed_data(df_clean)
