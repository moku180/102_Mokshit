"""Data loading and cleaning for fraud detection."""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_raw_data(filepath='data/raw/PS_20174392719_1491204439457_log.csv', nrows=None):
    """Load PaySim dataset from CSV."""
    print(f"Loading data from {filepath}...")
    
    # Optimize memory with specific dtypes
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
    """Remove invalid records and handle missing values."""
    print("\nCleaning data...")
    initial_count = len(df)
    
    # Add transaction ID
    df = df.reset_index(drop=True)
    df['transaction_id'] = df.index
    
    # Remove negative balances (data errors)
    df = df[
        (df['oldbalanceOrg'] >= 0) & 
        (df['newbalanceOrig'] >= 0) &
        (df['oldbalanceDest'] >= 0) & 
        (df['newbalanceDest'] >= 0)
    ].copy()
    
    print(f"Removed {initial_count - len(df):,} transactions with negative balances")
    
    # Check missing values
    missing = df.isnull().sum()
    if missing.any():
        print("\nMissing values found:")
        print(missing[missing > 0])
        df = df.dropna()
    else:
        print("No missing values found")
    
    print(f"\nFinal dataset: {len(df):,} transactions")
    print(f"Fraud rate after cleaning: {df['isFraud'].mean()*100:.4f}%")
    
    return df


def save_processed_data(df, filepath='data/processed/cleaned_data.csv'):
    """Save cleaned data to CSV."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving processed data to {filepath}...")
    df.to_csv(filepath, index=False)
    print("Data saved successfully")


def load_processed_data(filepath='data/processed/cleaned_data.csv'):
    """Load previously cleaned data."""
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
    df = load_raw_data(nrows=100000)
    df_clean = clean_data(df)
    save_processed_data(df_clean)
