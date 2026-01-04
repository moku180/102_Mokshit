"""Model training for fraud detection."""

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
    """Split data chronologically to prevent data leakage."""
    print(f"Performing time-based split with {test_size*100}% test size...")
    
    df_sorted = df.sort_values(time_column).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    
    print(f"Train set: {len(train_df):,} transactions ({train_df['isFraud'].sum():,} frauds)")
    print(f"Test set: {len(test_df):,} transactions ({test_df['isFraud'].sum():,} frauds)")
    print(f"Train fraud rate: {train_df['isFraud'].mean()*100:.4f}%")
    print(f"Test fraud rate: {test_df['isFraud'].mean()*100:.4f}%")
    
    return train_df, test_df


def train_baseline(X_train, y_train, X_test, y_test, use_class_weights=True):
    """Train baseline Logistic Regression model."""
    print("\n" + "="*60)
    print("Training Baseline: Logistic Regression")
    print("="*60)
    
    if use_class_weights:
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        print(f"Class weights: {class_weight_dict}")
    else:
        class_weight_dict = None
    
    model = LogisticRegression(
        class_weight=class_weight_dict,
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    return model


def train_lightgbm(X_train, y_train, X_test, y_test, use_class_weights=True):
    """Train LightGBM model optimized for fraud detection."""
    print("\n" + "="*60)
    print("Training Main Model: LightGBM")
    print("="*60)
    
    # Calculate class weight for imbalanced data
    if use_class_weights:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = 1.0
    
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
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
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


def save_model(model, filepath, model_type='lightgbm'):
    """Save trained model to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving model to {filepath}...")
    
    if model_type == 'lightgbm':
        model.save_model(filepath)
    else:
        joblib.dump(model, filepath)
    
    print("Model saved successfully!")


def load_model(filepath, model_type='lightgbm'):
    """Load trained model from disk."""
    print(f"Loading model from {filepath}...")
    
    if model_type == 'lightgbm':
        model = lgb.Booster(model_file=filepath)
    else:
        model = joblib.load(filepath)
    
    print("Model loaded successfully!")
    return model


if __name__ == "__main__":
    from data_loader import load_raw_data, clean_data
    from features import engineer_features, get_feature_columns
    
    df = load_raw_data(nrows=50000)
    df = clean_data(df)
    df, _ = engineer_features(df)
    
    feature_cols = get_feature_columns(df)
    train_df, test_df = time_based_split(df, test_size=0.2)
    
    X_train = train_df[feature_cols]
    y_train = train_df['isFraud']
    X_test = test_df[feature_cols]
    y_test = test_df['isFraud']
    
    baseline_model = train_baseline(X_train, y_train, X_test, y_test)
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test)
    
    save_model(baseline_model, 'models/baseline_lr.pkl', model_type='sklearn')
    save_model(lgb_model, 'models/lightgbm.txt', model_type='lightgbm')
