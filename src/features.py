"""Feature engineering for fraud detection."""

import pandas as pd
import numpy as np


def engineer_features(df):
    """Create fraud detection features from transaction data."""
    print("Engineering features...")
    df = df.copy()
    
    # Balance error features (key fraud indicators)
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['absErrorBalanceOrig'] = np.abs(df['errorBalanceOrig'])
    df['absErrorBalanceDest'] = np.abs(df['errorBalanceDest'])
    df['totalBalanceError'] = df['absErrorBalanceOrig'] + df['absErrorBalanceDest']
    
    # Amount-based features
    df['log_amount'] = np.log1p(df['amount'])
    df['amount_to_orig_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['amount_to_dest_balance_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1)
    df['drains_origin_account'] = (df['newbalanceOrig'] == 0).astype(int)
    df['fills_dest_account'] = (df['oldbalanceDest'] == 0).astype(int)
    
    # Transaction type features
    df = pd.get_dummies(df, columns=['type'], prefix='type', drop_first=False)
    df['can_be_fraud'] = ((df['type_TRANSFER'] == 1) | (df['type_CASH_OUT'] == 1)).astype(int)
    
    # Time features
    df['hour'] = df['step'] % 24
    df['day'] = df['step'] // 24
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    feature_count = len([col for col in df.columns if col not in ['nameOrig', 'nameDest', 'transaction_id', 'isFraud', 'isFlaggedFraud']])
    print(f"Feature engineering complete. Total features: {feature_count}")
    
    feature_info = {
        'balance_features': ['errorBalanceOrig', 'errorBalanceDest', 'absErrorBalanceOrig', 
                            'absErrorBalanceDest', 'totalBalanceError'],
        'amount_features': ['log_amount', 'amount_to_orig_balance_ratio', 
                           'amount_to_dest_balance_ratio', 'drains_origin_account', 'fills_dest_account'],
        'type_features': [col for col in df.columns if col.startswith('type_')],
        'time_features': ['hour', 'day', 'is_night']
    }
    
    return df, feature_info


def get_feature_columns(df):
    """Get list of feature columns for modeling."""
    exclude_cols = ['nameOrig', 'nameDest', 'transaction_id', 'isFraud', 'isFlaggedFraud']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


if __name__ == "__main__":
    from data_loader import load_raw_data, clean_data
    
    df = load_raw_data(nrows=50000)
    df = clean_data(df)
    df_featured, info = engineer_features(df)
    
    print(f"\nFeature columns: {get_feature_columns(df_featured)}")
    print(f"\nDataset shape: {df_featured.shape}")
