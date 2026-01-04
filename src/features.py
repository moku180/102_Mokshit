"""
Feature engineering module for fraud detection.

This module creates features that help identify fraudulent transactions:
1. Balance inconsistency features (key fraud indicators)
2. Amount-based features (normalization, statistical features)
3. Transaction behavior flags
4. Categorical encoding
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def create_balance_features(df):
    """
    Create balance inconsistency features.
    
    These are critical fraud indicators - fraudulent transactions often have
    balance inconsistencies where the balance changes don't match the transaction amount.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction data with balance columns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with new balance features
    """
    df = df.copy()
    
    # Origin account balance error
    # Expected: newbalanceOrig = oldbalanceOrg - amount
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    
    # Destination account balance error
    # Expected: newbalanceDest = oldbalanceDest + amount
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    # Absolute errors (magnitude of inconsistency)
    df['absErrorBalanceOrig'] = np.abs(df['errorBalanceOrig'])
    df['absErrorBalanceDest'] = np.abs(df['errorBalanceDest'])
    
    # Combined error indicator
    df['totalBalanceError'] = df['absErrorBalanceOrig'] + df['absErrorBalanceDest']
    
    # Binary flags for any balance error
    df['hasBalanceErrorOrig'] = (df['absErrorBalanceOrig'] > 0.01).astype('int8')
    df['hasBalanceErrorDest'] = (df['absErrorBalanceDest'] > 0.01).astype('int8')
    
    # Zero balance flags (suspicious patterns)
    df['origBalanceZero'] = (df['oldbalanceOrg'] == 0).astype('int8')
    df['destBalanceZero'] = (df['oldbalanceDest'] == 0).astype('int8')
    df['newOrigBalanceZero'] = (df['newbalanceOrig'] == 0).astype('int8')
    df['newDestBalanceZero'] = (df['newbalanceDest'] == 0).astype('int8')
    
    return df


def create_amount_features(df):
    """
    Create amount-based features.
    
    Features include normalization, log transforms, and statistical indicators
    that help identify unusual transaction amounts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction data with amount column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with new amount features
    """
    df = df.copy()
    
    # Log transform of amount (handles skewness)
    df['log_amount'] = np.log1p(df['amount'])
    
    # Amount relative to origin balance
    df['amount_to_orig_balance_ratio'] = np.where(
        df['oldbalanceOrg'] > 0,
        df['amount'] / df['oldbalanceOrg'],
        0
    )
    
    # Amount relative to destination balance
    df['amount_to_dest_balance_ratio'] = np.where(
        df['oldbalanceDest'] > 0,
        df['amount'] / df['oldbalanceDest'],
        0
    )
    
    # Flag for transactions that drain the origin account
    df['drains_origin_account'] = (
        (df['newbalanceOrig'] == 0) & 
        (df['oldbalanceOrg'] > 0)
    ).astype('int8')
    
    # Flag for large transactions (top 5%)
    amount_95th = df['amount'].quantile(0.95)
    df['is_large_transaction'] = (df['amount'] > amount_95th).astype('int8')
    
    # Round number flag (fraudsters often use round numbers)
    df['is_round_amount'] = (df['amount'] % 1000 == 0).astype('int8')
    
    return df


def create_transaction_features(df):
    """
    Create transaction type and behavior features.
    
    Different transaction types have different fraud patterns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction data with type column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with encoded transaction features
    """
    df = df.copy()
    
    # One-hot encode transaction type
    type_dummies = pd.get_dummies(df['type'], prefix='type', dtype='int8')
    df = pd.concat([df, type_dummies], axis=1)
    
    # Transaction types that can have fraud (TRANSFER and CASH_OUT only)
    df['can_be_fraud'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype('int8')
    
    # Time-based features (step represents hours)
    df['hour'] = df['step'] % 24
    df['day'] = df['step'] // 24
    
    # Time of day categories
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype('int8')
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype('int8')
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype('int8')
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype('int8')
    
    return df


def engineer_features(df, fit_encoder=True, encoder=None):
    """
    Main feature engineering pipeline.
    
    Applies all feature engineering steps in the correct order.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw transaction data
    fit_encoder : bool
        Whether to fit a new encoder (True for training, False for test)
    encoder : LabelEncoder, optional
        Pre-fitted encoder for transaction types (use for test set)
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all engineered features
    dict
        Dictionary containing fitted encoders (for use on test set)
    """
    print("Engineering features...")
    
    # Apply all feature engineering functions
    df = create_balance_features(df)
    df = create_amount_features(df)
    df = create_transaction_features(df)
    
    # Store encoders for later use
    encoders = {}
    
    print(f"Feature engineering complete. Total features: {len(df.columns)}")
    print(f"New features created: {len(df.columns) - 11}")  # Original has 11 columns
    
    return df, encoders


def get_feature_columns(df):
    """
    Get list of feature columns for modeling.
    
    Excludes ID columns, target variable, and raw categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with engineered features
        
    Returns:
    --------
    list
        List of feature column names
    """
    # Columns to exclude from features
    exclude_cols = [
        'transaction_id',
        'isFraud',  # Target variable
        'isFlaggedFraud',  # Not used as feature
        'nameOrig',  # High cardinality, not useful
        'nameDest',  # High cardinality, not useful
        'type',  # Categorical, we use one-hot encoded version
    ]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    return feature_cols


if __name__ == "__main__":
    # Example usage
    from data_loader import load_raw_data, clean_data
    
    # Load sample data
    df = load_raw_data(nrows=10000)
    df = clean_data(df)
    
    # Engineer features
    df_features, encoders = engineer_features(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df_features)
    
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols:
        print(f"  - {col}")
