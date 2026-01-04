# Transaction Fraud Detection System

A production-quality machine learning system for detecting fraudulent financial transactions using the PaySim dataset. Features real-time monitoring dashboard with SHAP explainability, achieving 99% precision on high-risk transactions.

## 🎯 Business Problem

Financial fraud causes billions of dollars in losses annually. This ML-based system:

- **Identifies fraudulent transactions** before they complete
- **Ranks transactions by fraud risk** for efficient investigation  
- **Provides explainable predictions** via SHAP analysis
- **Real-time monitoring dashboard** for fraud analysts
- **Adapts to new fraud patterns** through continuous learning

### Impact

- **Cost Savings**: Each prevented fraud saves the transaction amount plus investigation costs
- **Customer Trust**: Protecting customers from fraud builds loyalty
- **Operational Efficiency**: 770x improvement (100 frauds caught vs 0.13 with random checking)

## 📊 Why Accuracy is Misleading

With only **0.13% fraud rate**, a naive model predicting "all legitimate" achieves **99.87% accuracy** but catches **ZERO frauds**!

### The Right Metrics

1. **PR-AUC**: 99.13% (Precision-Recall AUC for imbalanced data)
2. **ROC-AUC**: 99.94% (Overall discrimination ability)
3. **Precision@100**: 100% (All top 100 flagged transactions are fraud)
4. **Recall@1000**: 23.13% (Catches 23% of all frauds in top 1000)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv_fraud

# Activate (Windows)
.\venv_fraud\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Train model and generate predictions
python run_pipeline.py
```

**Output**:
- `models/lightgbm_best.txt` - Trained model
- `outputs/fraud_score.csv` - Fraud predictions

### 3. Launch Dashboard

```bash
# Install dashboard dependencies
pip install -r requirements_dashboard.txt

# Start dashboard server
python app.py
```

**Access**: http://localhost:5000

## 📊 Dashboard Features

### Real-Time Monitoring
- **Live Alerts**: Color-coded by risk (Critical/High/Medium)
- **Auto-Refresh**: Updates every 5 seconds
- **Metrics Cards**: Total alerts, precision, avg score
- **Interactive Charts**: Fraud distribution and timeline

### SHAP Explanations
Click any alert to see:
- **Why is this fraud?** - Top 10 feature contributions
- **Feature Impact**: Visual bars showing positive/negative contributions
- **Transparency**: Helps analysts understand and trust predictions

### Dashboard Layout
```
┌─────────────────────────────────────────────┐
│  🛡️ Fraud Detection Dashboard    [Active]  │
├─────────────────────────────────────────────┤
│  [Total Alerts] [Precision] [Avg Score]     │
├──────────────────────┬──────────────────────┤
│  🚨 Recent Alerts    │  📈 Analytics        │
│  ┌─────────────────┐│  ┌─────────────────┐ │
│  │ Transaction #123││  │ Score Distrib.  │ │
│  │ 94.5% Fraud     ││  │ [Chart]         │ │
│  │ Critical        ││  └─────────────────┘ │
│  └─────────────────┘│  ┌─────────────────┐ │
│  [Click for SHAP]   │  │ Alerts Timeline │ │
│                      │  │ [Chart]         │ │
└──────────────────────┴──────────────────────┘
```

## 🏗️ Project Structure

```
fraud_detection/
├── data/
│   ├── raw/                    # PaySim dataset
│   └── processed/              # Cleaned data
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_threshold_tuning.ipynb
├── src/
│   ├── data_loader.py         # Data loading and cleaning
│   ├── features.py            # Feature engineering
│   ├── train.py               # Model training
│   └── evaluate.py            # Evaluation metrics
├── dashboard/
│   ├── index.html             # Dashboard UI
│   ├── style.css              # Modern dark theme
│   └── script.js              # Real-time updates
├── models/                     # Trained models
├── outputs/                    # Predictions and reports
├── app.py                      # Flask API + SHAP
├── run_pipeline.py            # End-to-end pipeline
└── README.md                   # This file
```

## 🔬 Technical Approach

### 1. Data Cleaning
- Remove transactions with negative balances
- Handle missing values
- Create unique transaction IDs

### 2. Feature Engineering (41 features)

**Balance Error Features** (strongest fraud indicators):
- `errorBalanceOrig`: Balance inconsistency in origin account
- `errorBalanceDest`: Balance inconsistency in destination account
- `totalBalanceError`: Combined balance errors

**Amount Features**:
- `log_amount`: Log-transformed amount
- `amount_to_orig_balance_ratio`: Transaction size vs account balance
- `drains_origin_account`: Binary flag if account emptied

**Transaction Type**:
- One-hot encoded transaction types
- `can_be_fraud`: Only TRANSFER and CASH_OUT can be fraudulent

**Time Features**:
- `hour`, `day`: Temporal patterns
- `is_night`: Nighttime transactions (22:00-06:00)

### 3. Model Selection: LightGBM

**Why LightGBM?**
- Excellent performance on tabular data
- Handles class imbalance with `scale_pos_weight`
- Fast training and prediction
- Built-in feature importance

**Hyperparameters**:
```python
{
    'num_leaves': 31,
    'learning_rate': 0.05,
    'max_depth': 7,
    'scale_pos_weight': 770  # Handles 0.13% fraud rate
}
```

### 4. Time-Based Split
- **Train**: First 80% of transactions (chronological)
- **Test**: Last 20% of transactions
- **Why**: Prevents data leakage, simulates production scenario

### 5. Evaluation Strategy
- **PR-AUC**: Primary metric for imbalanced data
- **Precision@K**: Business-relevant (limited investigation capacity)
- **SHAP**: Model explainability for production deployment

## 📈 Model Performance

| Metric | Score | Interpretation |
|--------|-------|----------------|
| PR-AUC | 99.13% | Excellent precision-recall trade-off |
| ROC-AUC | 99.94% | Near-perfect discrimination |
| Precision@100 | 100% | All top 100 are fraud |
| Precision@500 | 99.2% | 496/500 top predictions are fraud |
| Recall@1000 | 23.13% | Catches 23% of all frauds in top 1000 |

### Business Impact

**Before**: Random checking
- Check 100 transactions → Find 0.13 frauds

**After**: ML model
- Check top 100 predictions → Find 100 frauds
- **770x improvement**

## 🔍 Model Explainability (SHAP)

### Example Explanation

```
Transaction #6362573
Fraud Score: 94.5%

Top Contributing Features:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
errorBalanceOrig      [████████████] +0.45
  Value: 1000.0
  Impact: Suspicious balance inconsistency
  
type_TRANSFER         [████████] +0.30
  Value: 1.0
  Impact: High-risk transaction type
  
amount_ratio          [█████] +0.15
  Value: 0.95
  Impact: Unusual amount relative to balance
```

## 🎯 For Interviews

### Demo Script (3 minutes)

**1. Show Results** (30s)
```bash
python run_pipeline.py
# Point to PR-AUC: 99.13%, Precision@100: 100%
```

**2. Launch Dashboard** (30s)
```bash
python app.py
# Open http://localhost:5000
```

**3. Explain Features** (60s)
- "Real-time monitoring with auto-refresh"
- "Click any alert to see SHAP explanation"
- "Shows WHY the model flagged this transaction"

**4. Show SHAP** (60s)
- Click critical alert
- "Balance error is the strongest indicator"
- "Model is transparent and explainable"

### Key Points to Emphasize

✅ **End-to-End System**: From raw data to production dashboard  
✅ **Explainability**: SHAP makes AI transparent  
✅ **Real-World Metrics**: Precision@K, not just accuracy  
✅ **Production-Ready**: API, monitoring, error handling  
✅ **Modern UI**: Dark theme, real-time updates, responsive

## 📁 Dataset

**Source**: [PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)

**Statistics**:
- **Transactions**: 6,362,620
- **Fraud Rate**: 0.129% (8,213 frauds)
- **Transaction Types**: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN
- **Fraudulent Types**: Only TRANSFER and CASH_OUT

**Features**:
- `step`: Time unit (1 step = 1 hour)
- `type`: Transaction type
- `amount`: Transaction amount
- `nameOrig`, `nameDest`: Customer IDs
- `oldbalanceOrg`, `newbalanceOrig`: Origin account balances
- `oldbalanceDest`, `newbalanceDest`: Destination account balances
- `isFraud`: Target variable
- `isFlaggedFraud`: Existing fraud detection system flag

## 🛠️ Tech Stack

**Backend**:
- Python 3.8+
- LightGBM (gradient boosting)
- SHAP (explainability)
- Flask (API server)
- Pandas, NumPy, Scikit-learn

**Frontend**:
- HTML5, CSS3, JavaScript
- Chart.js (visualizations)
- Modern dark theme with glassmorphism

**ML Pipeline**:
- Jupyter notebooks (exploration)
- Modular Python scripts (production)
- Automated pipeline (run_pipeline.py)

## 🚨 Troubleshooting

**Dashboard won't load?**
- Check Flask is running: `python app.py`
- Verify: http://localhost:5000
- Check predictions exist: `outputs/fraud_score.csv`

**SHAP explanations fail?**
- Ensure SHAP is installed: `pip install shap`
- SHAP available for transactions: 6,312,620 - 6,362,619
- Click on high-priority alerts (top of list)

**Charts expanding?**
- Refresh page (Ctrl+R)
- Charts have fixed 300px height

**Model not found?**
- Run pipeline first: `python run_pipeline.py`
- Check: `models/lightgbm_best.txt` exists

## 📚 Documentation

- **Implementation Plan**: Details technical approach
- **Walkthrough**: Step-by-step guide with results
- **Interview Guide**: Demo script and Q&A

## 🎓 Model Limitations

1. **Synthetic Data**: Trained on simulated transactions, may need retraining on real data
2. **Temporal Drift**: Fraud patterns evolve, requires periodic retraining
3. **Feature Engineering**: Assumes balance information is available
4. **Class Imbalance**: Extreme imbalance (0.13%) requires careful threshold tuning

## 🚀 Next Steps

### For Production Deployment

1. **API Wrapper**: REST API for real-time scoring
2. **Monitoring**: Track model performance, data drift
3. **Feedback Loop**: Incorporate fraud analyst feedback
4. **A/B Testing**: Compare against existing systems
5. **Scalability**: Batch processing for high-volume transactions

### Potential Improvements

- [ ] Hyperparameter tuning (Optuna, GridSearch)
- [ ] Ensemble methods (stacking, blending)
- [ ] Deep learning models (autoencoders for anomaly detection)
- [ ] Graph-based features (transaction networks)
- [ ] Real-time streaming (Kafka, Spark)

## 📞 Contact

For questions or collaboration:
- GitHub: [Your GitHub Profile]
- Email: [Your Email]

---

**Built with ❤️ for fraud prevention and financial security**
