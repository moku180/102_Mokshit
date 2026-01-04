# Fraud Detection Dashboard

## 🎯 Overview

Real-time fraud detection monitoring dashboard with:
- **Live Alerts**: High-risk transactions with fraud scores
- **SHAP Explanations**: Understand why each transaction is flagged
- **Analytics**: Fraud trends, score distribution, precision metrics
- **Modern UI**: Dark theme with glassmorphism and smooth animations

## 🚀 Quick Start

### 1. Install Dashboard Dependencies

```bash
# Activate virtual environment
.\venv_fraud\Scripts\activate

# Install Flask and SHAP
pip install -r requirements_dashboard.txt
```

### 2. Run the Dashboard

```bash
python app.py
```

### 3. Open Dashboard

Navigate to: **http://localhost:5000**

## 📊 Features

### Real-Time Alert Feed
- Color-coded risk levels (Critical/High/Medium)
- Transaction details and fraud scores
- Click any alert to see SHAP explanation

### SHAP Explanations
- **Why is this fraud?** - Feature contributions
- Top 10 features driving the prediction
- Visual breakdown with contribution bars
- Positive (purple) = increases fraud risk
- Negative (red) = decreases fraud risk

### Dashboard Metrics
- **Total Alerts**: High-risk transactions count
- **Precision@100**: Model accuracy on top predictions
- **Avg Fraud Score**: Mean risk level
- **True Frauds**: Confirmed fraud cases

### Visualizations
- **Fraud Score Distribution**: Histogram of risk scores
- **Alerts Timeline**: Fraud detections over time

## 🎨 Dashboard Layout

```
┌─────────────────────────────────────────────┐
│  🛡️ Fraud Detection Dashboard    [Active]  │
├─────────────────────────────────────────────┤
│  [Total Alerts] [Precision] [Avg Score] [...│
├──────────────────────┬──────────────────────┤
│  🚨 Recent Alerts    │  📈 Analytics        │
│  ┌─────────────────┐│  ┌─────────────────┐ │
│  │ Transaction #123││  │ Score Distrib.  │ │
│  │ 94.5% Fraud     ││  │ [Chart]         │ │
│  │ Critical        ││  └─────────────────┘ │
│  └─────────────────┘│  ┌─────────────────┐ │
│  [Click for SHAP]   │  │ Alerts Timeline │ │
│                      │  │ [Chart]         │ │
│                      │  └─────────────────┘ │
└──────────────────────┴──────────────────────┘
```

## 🔍 How to Use

### Monitor Alerts
1. Dashboard auto-refreshes every 5 seconds
2. Filter by risk level (dropdown)
3. Click any alert to see why it's flagged

### Understand SHAP Explanations
- **Base Value**: Starting fraud probability
- **Feature Contributions**: How each feature changes the score
- **Final Score**: Predicted fraud probability

Example:
```
Base: 0.13% (average fraud rate)
+ Balance Error: +0.45 (suspicious)
+ Transaction Type: +0.30 (TRANSFER)
+ Amount Ratio: +0.15 (unusual)
= Final Score: 94.5% (HIGH RISK)
```

## 📁 Files

```
fraud_detection/
├── app.py                      # Flask API server
├── dashboard/
│   ├── index.html             # Dashboard UI
│   ├── style.css              # Modern styling
│   └── script.js              # Real-time updates
├── requirements_dashboard.txt  # Dashboard dependencies
└── README_DASHBOARD.md        # This file
```

## 🛠️ API Endpoints

- `GET /` - Dashboard UI
- `GET /api/alerts` - Get fraud alerts
- `GET /api/metrics` - Dashboard metrics
- `GET /api/explain/<id>` - SHAP explanation
- `GET /api/distribution` - Score distribution
- `GET /api/timeline` - Alerts over time

## 🎯 For Interviews

**Demo Flow**:
1. Start dashboard: `python app.py`
2. Show real-time alerts with risk levels
3. Click alert → SHAP explanation appears
4. Explain: "This shows WHY the model flagged this transaction"
5. Point out top contributing features
6. Show visualizations updating in real-time

**Key Points**:
- "Real-time monitoring with auto-refresh"
- "SHAP provides explainability - crucial for production"
- "Modern, professional UI built from scratch"
- "Fully functional end-to-end system"

## 🚨 Troubleshooting

**Dashboard won't load?**
- Check Flask is running: `python app.py`
- Verify predictions exist: `outputs/fraud_score.csv`
- Check model exists: `models/lightgbm_best.txt`

**SHAP explanations fail?**
- Ensure test data is available
- Check SHAP is installed: `pip install shap`

**Charts not rendering?**
- Check browser console for errors
- Verify Chart.js CDN is accessible

## 💡 Next Steps

- Add user authentication
- Implement alert acknowledgment
- Export reports to PDF
- Add email notifications
- Deploy to cloud (AWS/Azure)
