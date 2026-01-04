"""
Model training module for fraud detection.

This module handles:
1. Time-based train/test split (no data leakage)
2. Training baseline and advanced models
3. Handling class imbalance with weights and SMOTE
4. Model persistence
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def time_based_split(df, test_size=0.2, time_column='step'):
    
    print(f"Performing time-based split with {test_size*100}% test size...")
    
    # Sort by time
    df_sorted = df.sort_values(time_column).reset_index(drop=True)
    
    # Calculate split point
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    print(f"Train set: {len(train_df):,} transactions ({train_df['isFraud'].sum():,} frauds)")
    print(f"Test set: {len(test_df):,} transactions ({test_df['isFraud'].sum():,} frauds)")
    print(f"Train fraud rate: {train_df['isFraud'].mean()*100:.4f}%")
    print(f"Test fraud rate: {test_df['isFraud'].mean()*100:.4f}%")
    
    return train_df, test_df


def train_baseline(X_train, y_train, X_test, y_test, use_class_weights=True):
    """
    Train baseline Logistic Regression model.
    
    Logistic Regression serves as a simple, interpretable baseline.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    use_class_weights : bool
        Whether to use class weights to handle imbalance
        
    Returns:
    --------
    LogisticRegression
        Trained model
    """
    print("\n" + "="*60)
    print("Training Baseline: Logistic Regression")
    print("="*60)
    
    if use_class_weights:
        # Compute class weights to handle imbalance
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"Class weights: {class_weight_dict}")
    else:
        class_weight_dict = None
    
    # Train model
    model = LogisticRegression(
        class_weight=class_weight_dict,
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training...")
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("Training complete!")
    
    return model


def train_lightgbm(X_train, y_train, X_test, y_test, use_class_weights=True):
    """
    Train LightGBM model with hyperparameters optimized for fraud detection.
    
    LightGBM is chosen for:
    - Excellent performance on tabular data
    - Handles imbalanced data well
    - Fast training and prediction
    - Built-in feature importance
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    use_class_weights : bool
        Whether to use class weights to handle imbalance
        
    Returns:
    --------
    lgb.Booster
        Trained LightGBM model
    """
    print("\n" + "="*60)
    print("Training Main Model: LightGBM")
    print("="*60)
    
    # Calculate scale_pos_weight for imbalanced data
    if use_class_weights:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = 1.0
    
    # LightGBM parameters optimized for fraud detection
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 7,
        'min_child_samples': 20,
        'scale_pos_weight': scale_pos_weight,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    print("Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Train model
    print("\nTraining...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50)
        ]
    )
    
    print(f"\nBest iteration: {model.best_iteration}")
    print("Training complete!")
    
    return model


def train_with_smote(X_train, y_train, X_test, y_test, model_type='lightgbm'):
    """
    Train model with SMOTE oversampling on training set.
    
    SMOTE creates synthetic samples of the minority class.
    Only applied to training set to avoid data leakage.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    model_type : str
        'logistic' or 'lightgbm'
        
    Returns:
    --------
    model
        Trained model
    """
    print("\n" + "="*60)
    print(f"Training with SMOTE: {model_type.upper()}")
    print("="*60)
    
    print(f"Original training set: {len(y_train):,} samples")
    print(f"  Class 0 (legitimate): {(y_train == 0).sum():,}")
    print(f"  Class 1 (fraud): {(y_train == 1).sum():,}")
    
    # Apply SMOTE
    # Use sampling_strategy to avoid extreme oversampling
    # We'll oversample to 10% of majority class (still imbalanced but more balanced)
    fraud_count = (y_train == 1).sum()
    legit_count = (y_train == 0).sum()
    target_fraud_count = int(legit_count * 0.1)
    
    if target_fraud_count > fraud_count:
        sampling_strategy = target_fraud_count / legit_count
        print(f"\nApplying SMOTE with sampling_strategy={sampling_strategy:.4f}...")
        
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, n_jobs=-1)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"Resampled training set: {len(y_train_resampled):,} samples")
        print(f"  Class 0 (legitimate): {(y_train_resampled == 0).sum():,}")
        print(f"  Class 1 (fraud): {(y_train_resampled == 1).sum():,}")
    else:
        print("Skipping SMOTE - fraud count already sufficient")
        X_train_resampled = X_train
        y_train_resampled = y_train
    
    # Train model based on type
    if model_type == 'logistic':
        model = train_baseline(
            X_train_resampled, y_train_resampled, 
            X_test, y_test, 
            use_class_weights=False  # SMOTE already balanced
        )
    else:  # lightgbm
        model = train_lightgbm(
            X_train_resampled, y_train_resampled, 
            X_test, y_test, 
            use_class_weights=False  # SMOTE already balanced
        )
    
    return model


def save_model(model, filepath, model_type='lightgbm'):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : model object
        Trained model
    filepath : str
        Output file path
    model_type : str
        'lightgbm' or 'sklearn'
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving model to {filepath}...")
    
    if model_type == 'lightgbm':
        model.save_model(filepath)
    else:
        joblib.dump(model, filepath)
    
    print("Model saved successfully!")


def load_model(filepath, model_type='lightgbm'):
    """
    Load trained model from disk.
    
    Parameters:
    -----------
    filepath : str
        Model file path
    model_type : str
        'lightgbm' or 'sklearn'
        
    Returns:
    --------
    model
        Loaded model
    """
    print(f"Loading model from {filepath}...")
    
    if model_type == 'lightgbm':
        model = lgb.Booster(model_file=filepath)
    else:
        model = joblib.load(filepath)
    
    print("Model loaded successfully!")
    return model


if __name__ == "__main__":
    # Example usage
    from data_loader import load_raw_data, clean_data
    from features import engineer_features, get_feature_columns
    
    # Load and prepare data
    print("Loading sample data...")
    df = load_raw_data(nrows=50000)
    df = clean_data(df)
    df, _ = engineer_features(df)
    
    # Get features
    feature_cols = get_feature_columns(df)
    
    # Time-based split
    train_df, test_df = time_based_split(df, test_size=0.2)
    
    X_train = train_df[feature_cols]
    y_train = train_df['isFraud']
    X_test = test_df[feature_cols]
    y_test = test_df['isFraud']
    
    # Train models
    baseline_model = train_baseline(X_train, y_train, X_test, y_test)
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test)
    
    # Save models
    save_model(baseline_model, 'models/baseline_lr.pkl', model_type='sklearn')
    save_model(lgb_model, 'models/lightgbm.txt', model_type='lightgbm')
