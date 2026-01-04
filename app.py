"""Flask API for Fraud Detection Dashboard with SHAP explanations."""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from features import engineer_features, get_feature_columns

app = Flask(__name__, static_folder='dashboard', static_url_path='')
CORS(app)

# Load model and data on startup
print("Loading model and data...")
model = lgb.Booster(model_file='models/lightgbm_best.txt')
predictions_df = pd.read_csv('outputs/fraud_score.csv')

# Load test data for SHAP
print("Loading test data for SHAP analysis...")
test_data = pd.read_csv('data/raw/PS_20174392719_1491204439457_log.csv', nrows=100000)
from data_loader import clean_data
test_data = clean_data(test_data)
test_data, _ = engineer_features(test_data)
feature_cols = get_feature_columns(test_data)

# Initialize SHAP explainer
print("Initializing SHAP explainer...")
explainer = shap.TreeExplainer(model)
print("Dashboard API ready!")


@app.route('/')
def index():
    """Serve dashboard HTML."""
    return send_from_directory('dashboard', 'index.html')


@app.route('/api/alerts')
def get_alerts():
    """Get recent high-risk fraud alerts."""
    threshold = float(request.args.get('threshold', 0.5))
    limit = int(request.args.get('limit', 50))
    
    # Filter high-risk transactions
    alerts = predictions_df[predictions_df['fraud_score'] >= threshold].head(limit)
    
    # Add risk level
    def get_risk_level(score):
        if score >= 0.9:
            return 'critical'
        elif score >= 0.7:
            return 'high'
        elif score >= 0.5:
            return 'medium'
        return 'low'
    
    alerts_list = []
    for _, row in alerts.iterrows():
        alerts_list.append({
            'transaction_id': int(row['transaction_index']),
            'fraud_score': float(row['fraud_score']),
            'actual_fraud': int(row['actual_fraud']),
            'rank': int(row['rank']),
            'risk_level': get_risk_level(row['fraud_score'])
        })
    
    return jsonify({
        'alerts': alerts_list,
        'total': len(alerts_list)
    })


@app.route('/api/metrics')
def get_metrics():
    """Get dashboard metrics."""
    high_risk = predictions_df[predictions_df['fraud_score'] >= 0.7]
    
    # Calculate precision at different thresholds
    precision_100 = predictions_df.head(100)['actual_fraud'].mean()
    precision_500 = predictions_df.head(500)['actual_fraud'].mean()
    
    metrics = {
        'total_alerts': len(high_risk),
        'total_transactions': len(predictions_df),
        'avg_fraud_score': float(high_risk['fraud_score'].mean()) if len(high_risk) > 0 else 0,
        'precision_at_100': float(precision_100),
        'precision_at_500': float(precision_500),
        'true_frauds_caught': int(high_risk['actual_fraud'].sum()),
        'alert_rate': float(len(high_risk) / len(predictions_df) * 100)
    }
    
    return jsonify(metrics)


@app.route('/api/explain/<int:transaction_id>')
def explain_transaction(transaction_id):
    """Get SHAP explanation for a specific transaction."""
    try:
        # Find transaction in test data
        if transaction_id >= len(test_data):
            return jsonify({'error': 'Transaction not found'}), 404
        
        transaction = test_data.iloc[transaction_id:transaction_id+1]
        X = transaction[feature_cols]
        
        # Get SHAP values
        shap_values = explainer.shap_values(X)
        
        # Get base value and prediction
        base_value = explainer.expected_value
        prediction = model.predict(X)[0]
        
        # Create feature contributions
        contributions = []
        for i, feature in enumerate(feature_cols):
            contributions.append({
                'feature': feature,
                'value': float(X[feature].values[0]),
                'contribution': float(shap_values[0][i])
            })
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        return jsonify({
            'transaction_id': transaction_id,
            'prediction': float(prediction),
            'base_value': float(base_value),
            'contributions': contributions[:10],  # Top 10 features
            'explanation': f"Base fraud probability: {base_value:.4f}. "
                          f"After considering features, final score: {prediction:.4f}"
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/distribution')
def get_distribution():
    """Get fraud score distribution for visualization."""
    bins = np.linspace(0, 1, 21)
    hist, edges = np.histogram(predictions_df['fraud_score'], bins=bins)
    
    distribution = []
    for i in range(len(hist)):
        distribution.append({
            'range': f"{edges[i]:.2f}-{edges[i+1]:.2f}",
            'count': int(hist[i]),
            'midpoint': float((edges[i] + edges[i+1]) / 2)
        })
    
    return jsonify(distribution)


@app.route('/api/timeline')
def get_timeline():
    """Get alerts over time (simulated with ranks)."""
    # Group by rank ranges to simulate time
    timeline = []
    for i in range(0, 1000, 50):
        chunk = predictions_df.iloc[i:i+50]
        high_risk = chunk[chunk['fraud_score'] >= 0.7]
        timeline.append({
            'batch': i // 50,
            'alerts': len(high_risk),
            'avg_score': float(high_risk['fraud_score'].mean()) if len(high_risk) > 0 else 0
        })
    
    return jsonify(timeline)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Fraud Detection Dashboard Starting...")
    print("="*60)
    print(f"📊 Loaded {len(predictions_df):,} predictions")
    print(f"🎯 High-risk alerts: {len(predictions_df[predictions_df['fraud_score'] >= 0.7]):,}")
    print(f"🌐 Dashboard: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
