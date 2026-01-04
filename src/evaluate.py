"""Model evaluation for fraud detection with PR-AUC and Precision@K metrics."""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, auc, confusion_matrix,
    roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def calculate_pr_auc(y_true, y_pred_proba):
    """Calculate Precision-Recall AUC (better for imbalanced data)."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    return pr_auc, avg_precision


def precision_at_k(y_true, y_pred_proba, k):
    """Calculate precision in top K predictions."""
    top_k_indices = np.argsort(y_pred_proba)[-k:]
    precision = y_true.iloc[top_k_indices].sum() / k
    return precision


def recall_at_k(y_true, y_pred_proba, k):
    """Calculate recall in top K predictions."""
    top_k_indices = np.argsort(y_pred_proba)[-k:]
    total_frauds = y_true.sum()
    frauds_in_top_k = y_true.iloc[top_k_indices].sum()
    recall = frauds_in_top_k / total_frauds if total_frauds > 0 else 0
    return recall


def plot_pr_curve(y_true, y_pred_proba, model_name='Model', save_path=None):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc, avg_precision = calculate_pr_auc(y_true, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(recall, precision, linewidth=2, label=f'{model_name} (AP={avg_precision:.4f})')
    ax.axhline(y=y_true.mean(), color='r', linestyle='--', 
               label=f'Baseline (Random): {y_true.mean():.4f}')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PR curve to {save_path}")
    
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name='Model', save_path=None):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    return fig


def evaluate_model(model, X_test, y_test, model_name='Model', threshold=0.5, 
                   k_values=[100, 500, 1000], model_type='lightgbm'):
    """Comprehensive model evaluation with fraud-specific metrics."""
    print("\n" + "="*70)
    print(f"EVALUATION RESULTS: {model_name}")
    print("="*70)
    
    # Get predictions
    if model_type == 'lightgbm':
        y_pred_proba = model.predict(X_test)
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    pr_auc, avg_precision = calculate_pr_auc(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  ROC-AUC:              {roc_auc:.4f}")
    print(f"  PR-AUC:               {pr_auc:.4f}")
    print(f"  Average Precision:    {avg_precision:.4f}")
    
    # Precision@K and Recall@K
    print(f"\n🎯 Precision@K and Recall@K:")
    for k in k_values:
        prec_k = precision_at_k(y_test, y_pred_proba, k)
        rec_k = recall_at_k(y_test, y_pred_proba, k)
        print(f"  K={k:5d}  →  Precision: {prec_k:.4f}  |  Recall: {rec_k:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n📋 Confusion Matrix (threshold={threshold}):")
    print(f"  True Negatives:  {tn:8,}")
    print(f"  False Positives: {fp:8,}")
    print(f"  False Negatives: {fn:8,}")
    print(f"  True Positives:  {tp:8,}")
    
    # Additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n📈 Classification Metrics (threshold={threshold}):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Store results
    results = {
        'model_name': model_name,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'avg_precision': avg_precision,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    }
    
    # Add Precision@K and Recall@K
    for k in k_values:
        results[f'precision@{k}'] = precision_at_k(y_test, y_pred_proba, k)
        results[f'recall@{k}'] = recall_at_k(y_test, y_pred_proba, k)
    
    print("\n" + "="*70)
    
    return results


if __name__ == "__main__":
    from data_loader import load_raw_data, clean_data
    from features import engineer_features, get_feature_columns
    from train import time_based_split, train_baseline, train_lightgbm
    
    df = load_raw_data(nrows=50000)
    df = clean_data(df)
    df, _ = engineer_features(df)
    
    feature_cols = get_feature_columns(df)
    train_df, test_df = time_based_split(df, test_size=0.2)
    
    X_train = train_df[feature_cols]
    y_train = train_df['isFraud']
    X_test = test_df[feature_cols]
    y_test = test_df['isFraud']
    
    lgb_model = train_lightgbm(X_train, y_train, X_test, y_test)
    results = evaluate_model(lgb_model, X_test, y_test, model_name='LightGBM', model_type='lightgbm')
