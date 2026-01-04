"""Complete fraud detection pipeline - runs data loading through model evaluation."""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_loader import load_raw_data, clean_data
from features import engineer_features, get_feature_columns
from train import time_based_split, train_lightgbm, save_model
from evaluate import evaluate_model

def main():
    """Run complete fraud detection pipeline."""
    
    print("=" * 80)
    print("FRAUD DETECTION PIPELINE - COMPLETE EXECUTION")
    print("=" * 80)
    
    # Step 1: Load and Clean Data
    print("\n[1/5] Loading and cleaning data...")
    print("-" * 80)
    df = load_raw_data()
    print(f"✓ Loaded {len(df):,} transactions")
    
    df_clean = clean_data(df)
    print(f"✓ Cleaned data: {len(df_clean):,} transactions remaining")
    print(f"  - Fraud rate: {df_clean['isFraud'].mean() * 100:.3f}%")
    print(f"  - Fraudulent transactions: {df_clean['isFraud'].sum():,}")
    
    # Step 2: Engineer Features
    print("\n[2/5] Engineering features...")
    print("-" * 80)
    df_featured, feature_info = engineer_features(df_clean)
    feature_cols = get_feature_columns(df_featured)
    print(f"✓ Created {len(feature_cols)} features")
    
    # Step 3: Train/Test Split
    print("\n[3/5] Splitting data (time-based)...")
    print("-" * 80)
    train_df, test_df = time_based_split(df_featured)
    print(f"✓ Train set: {len(train_df):,} transactions ({train_df['isFraud'].sum():,} frauds)")
    print(f"✓ Test set:  {len(test_df):,} transactions ({test_df['isFraud'].sum():,} frauds)")
    
    X_train = train_df[feature_cols]
    y_train = train_df['isFraud']
    X_test = test_df[feature_cols]
    y_test = test_df['isFraud']
    
    # Step 4: Train Model
    print("\n[4/5] Training LightGBM model...")
    print("-" * 80)
    model = train_lightgbm(X_train, y_train, X_test, y_test)
    print("✓ Model training complete")
    
    # Save model
    model_path = 'models/lightgbm_best.txt'
    save_model(model, model_path, model_type='lightgbm')
    
    # Step 5: Evaluate Model
    print("\n[5/5] Evaluating model performance...")
    print("-" * 80)
    results = evaluate_model(model, X_test, y_test, model_type='lightgbm')
    
    # Print key metrics
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"\n📊 Overall Performance:")
    print(f"  - PR-AUC:     {results['pr_auc']:.4f}")
    print(f"  - ROC-AUC:    {results['roc_auc']:.4f}")
    
    print(f"\n🎯 Precision@K (Quality of Top Predictions):")
    for k in [100, 500, 1000]:
        if f'precision@{k}' in results:
            print(f"  - Precision@{k:4d}: {results[f'precision@{k}']:.4f}")
    
    print(f"\n📈 Recall@K (Coverage of Frauds):")
    for k in [100, 500, 1000]:
        if f'recall@{k}' in results:
            print(f"  - Recall@{k:4d}:    {results[f'recall@{k}']:.4f}")
    
    # Generate predictions
    print("\n" + "=" * 80)
    print("Generating fraud predictions...")
    print("=" * 80)
    
    y_pred_proba = model.predict(X_test)
    
    predictions_df = pd.DataFrame({
        'transaction_index': test_df.index,
        'fraud_score': y_pred_proba,
        'actual_fraud': y_test.values
    })
    
    predictions_df = predictions_df.sort_values('fraud_score', ascending=False).reset_index(drop=True)
    predictions_df['rank'] = range(1, len(predictions_df) + 1)
    
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / 'fraud_score.csv'
    predictions_df.to_csv(output_file, index=False)
    print(f"✓ Saved predictions to: {output_file}")
    print(f"  - Total predictions: {len(predictions_df):,}")
    print(f"  - Top 100 precision: {predictions_df.head(100)['actual_fraud'].mean():.4f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PIPELINE EXECUTION COMPLETE!")
    print("=" * 80)
    print(f"\n📁 Outputs saved to:")
    print(f"  - Model:       models/lightgbm_best.txt")
    print(f"  - Predictions: outputs/fraud_score.csv")
    
    print(f"\n🎉 Success! The fraud detection system is ready for use.")
    print("=" * 80)
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
