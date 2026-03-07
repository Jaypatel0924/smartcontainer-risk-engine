# SmartContainer Risk Engine - Project Documentation

## 📋 Project Overview

The **SmartContainer Risk Engine** is an AI/ML-based system designed to analyze container shipment data and predict risk levels. The system detects anomalies, assigns risk scores (0-100), and provides explainable predictions to help port authorities prioritize inspections and improve operational efficiency.

### Key Features
- 🤖 **Advanced ML Models**: Random Forest + Isolation Forest for robust predictions
- ⚖️ **SMOTE Resampling**: Handles imbalanced datasets effectively
- 📊 **Feature Engineering**: 15+ engineered features from raw data
- 🎯 **Risk Categorization**: Critical (≥50), Low Risk (20-49), Clear (<20)
- 💡 **Explainability**: SHAP-style rule-based explanations for every prediction
- 📈 **Interactive Dashboard**: Streamlit-based visualization and analysis
- 🔍 **Anomaly Detection**: Isolation Forest for outlier detection

## 🗂️ Project Structure

```
SmartContainer Risk Engine/
├── data/
│   ├── historical_data.csv      # Training data (54,000 records)
│   └── realtime_data.csv        # Inference data
├── models/                       # Trained model artifacts
│   ├── random_forest_model.pkl
│   ├── isolation_forest_model.pkl
│   └── feature_engineer.pkl
├── output/                       # Generated predictions
│   └── risk_predictions.csv
├── utils.py                      # Utility functions
├── eda_analysis.py              # Exploratory data analysis
├── feature_engineering.py       # Feature creation & transformation
├── model_training.py            # ML model training with SMOTE
├── predict.py                   # Prediction pipeline
├── dashboard.py                 # Streamlit dashboard
├── main.py                      # Main execution script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 📊 Data Schema

### Input Features
| Column | Type | Description |
|--------|------|-------------|
| Container_ID | String | Unique container identifier |
| Declaration_Date | Date | Import/export declaration date |
| Declaration_Time | Time | Declaration time |
| Trade_Regime | Category | Import/Export/Transit |
| Origin_Country | Category | Country of origin (ISO code) |
| Destination_Port | Category | Port destination code |
| Destination_Country | Category | Destination country |
| HS_Code | Category | Harmonized System code |
| Declared_Value | Numeric | Declared shipment value (USD) |
| Declared_Weight | Numeric | Declared weight (kg) |
| Measured_Weight | Numeric | Actual measured weight (kg) |
| Dwell_Time_Hours | Numeric | Container dwell time in hours |
| Clearance_Status | Category | Target: Clear / Low Risk / Critical |

### Output Features
| Column | Type | Description |
|--------|------|-------------|
| Container_ID | String | Original container ID |
| Risk_Score | Numeric | Predicted risk score (0-100) |
| Risk_Level | Category | Critical / Low Risk / Clear |
| Explanation | String | Rule-based explanation |
| Confidence | Numeric | Model confidence (%) |

## 🔧 Installation

### 1. Clone/Setup Project
```bash
cd "Hackathone 2"
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
# Using venv
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Option 1: Run Full Pipeline
```bash
python main.py --all
```

### Option 2: Run Individual Steps
```bash
# Step 1: Exploratory Data Analysis
python main.py --eda

# Step 2: Train ML Models
python main.py --train

# Step 3: Generate Predictions
python main.py --predict

# Step 4: Launch Interactive Dashboard
python main.py --dashboard
```

### Option 3: Direct Script Execution
```bash
# Run EDA
python eda_analysis.py

# Train models
python model_training.py

# Generate predictions
python predict.py

# Launch dashboard
streamlit run dashboard.py
```

## 📈 Model Architecture

### Random Forest Classifier
- **Estimators**: 150
- **Max Depth**: 20
- **Min Samples Split**: 5
- **Class Weight**: Balanced
- **Purpose**: Multi-class risk classification

### Isolation Forest
- **Contamination**: 3%
- **Estimators**: 100
- **Purpose**: Anomaly detection for outlier flagging

### Class Imbalance Handling: SMOTE
```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
```

**Why SMOTE?**
- Handles severe class imbalance (e.g., 97% Clear, 1% Critical)
- Generates synthetic minority samples intelligently
- Prevents model bias towards majority class
- Improves recall for critical risk class

## 🎯 Feature Engineering Details

### Core Features
1. **Weight_Diff_%**: Percentage difference between declared and measured weight
2. **Weight_Ratio**: Measured to declared weight ratio
3. **Value_Weight_Ratio**: Declared value per kilogram
4. **Dwell_Time_Hours**: Container dwell time
5. **High_Dwell**: Binary flag for dwell > 72h
6. **Very_High_Dwell**: Binary flag for dwell > 120h

### Domain-Specific Features
7. **HS_Code_Risk_Score**: Commodity-based risk (0-25)
8. **Origin_Risk_Score**: Country-based risk assessment
9. **Port_Risk_Score**: Port vulnerability scoring
10. **Trade_Risk_Score**: Trade regime risk weighting
11. **Anomaly_Score**: Composite anomaly indicator

### Feature Scaling
- StandardScaler applied to numerical features
- MinMaxScaler (0-100) for risk scores
- Label encoding for categorical variables

## 📊 Risk Scoring Methodology

### Risk Score Calculation
```python
# 70% from Random Forest predictions
rf_component = (P(Critical) * 100 + P(Low Risk) * 50) / 1.5

# 30% from Isolation Forest anomaly detection
if_component = normalized_anomaly_score * 30

# Final Score (0-100)
risk_score = (rf_component * 0.7 + if_component * 0.3)
```

### Risk Thresholds
- 🔴 **Critical Risk**: Risk_Score ≥ 50
- 🟡 **Low Risk**: 20 ≤ Risk_Score < 50
- 🟢 **Clear**: Risk_Score < 20

## 💡 Explainability

Each prediction includes rule-based explanations:

**Critical Risk Example:**
- "Critical weight discrepancy (35.2%); Excessive dwell time (145h)"

**Low Risk Example:**
- "Weight discrepancy detected (12.5%); High dwell time (89h)"

**Clear Example:**
- "Normal shipping pattern"

Explanations are generated by analyzing:
1. Weight discrepancies (>10%, >20%, >30%)
2. Dwell time violations (>72h, >120h)
3. Value-weight ratio anomalies
4. High-risk port/origin combinations
5. Composite risk indicators

## 📊 Dashboard Features

### Main Dashboard
- **Key Metrics**: Total containers, critical/low-risk/clear counts
- **Risk Distribution**: Pie chart and histograms
- **Top Critical Containers**: Bar chart of highest-risk items
- **Statistics**: Mean, median, std dev of risk scores

### Prediction Interface
- CSV file upload for batch predictions
- Real-time risk scoring
- CSV export of results

### Analytics
- Risk analysis by region
- Confidence vs. risk scatter plots
- Detailed statistical breakdowns

### Settings
- Model configuration details
- Feature engineering documentation
- Data processing options

## 🔄 Model Training Process

### 1. Data Loading & Preprocessing
```python
df = load_data('data/historical_data.csv')
# 54,000 containers with 15 features
```

### 2. Feature Engineering
- Extract 15+ features from raw data
- Handle missing values
- Create anomaly indicators

### 3. Train-Test Split
- 80% training, 20% test
- Stratified sampling by target class

### 4. SMOTE Application
```python
# Before: Clear=51,450 | Low Risk=1,025 | Critical=525 (imbalanced)
# After: All classes ~equal (SMOTE-resampled)
```

### 5. Model Training
- Random Forest: Balanced for classification
- Isolation Forest: Fitted on entire training set

### 6. Evaluation
- 5-fold cross-validation (F1-weighted)
- Classification metrics: Accuracy, Precision, Recall, F1
- Confusion matrix analysis

## 📈 Expected Performance

Based on the configuration image and historical data:

```
Classification Performance:
- Overall Accuracy: ~99.82%
- F1-Score (Critical): 96.68%
- F1-Score (Low Risk): 99.62%
- Records: 54,000 containers
- Test Split: 80/20

Risk Distribution (Output):
- Critical: 559 containers (1.0%)
- Low Risk: 1,073 containers (2.0%)
- Clear: 52,368 containers (97.0%)
```

## 🔍 Top Feature Importance

1. **Weight Diff %** (39.2%) - Most critical feature
2. **Weight Ratio** (26.0%)
3. **Dwell Time in Hours** (15.2%)
4. **Very High Dwell Flag** (5.1%)
5. **High Dwell Flag** (3.5%)
6. **HS Code Category** (2.5%)
7. **Declared Value** (1.6%)
8. **Measured Weight** (1.4%)

## 🎯 High-Risk Port Analysis

Top 10 ports by average risk score:

| Port | Avg Risk Score | Critical Count |
|------|---|---|
| PORT_71 | 9.08 | 173 |
| PORT_130 | 7.78 | 165 |
| PORT_131 | 7.65 | 158 |
| PORT_120 | 7.58 | 142 |
| PORT_90 | 7.00 | 128 |
| PORT_37 | 6.77 | 115 |
| PORT_62 | 6.38 | 98 |
| PORT_17 | 6.36 | 85 |

## 🔄 Critical Origins Analysis

Highest-risk countries:

| Country | Critical Cases | Risk Score |
|---------|---|---|
| CN | 173 | 25 |
| US | 80 | 15 |
| JP | 61 | 15 |
| VN | 36 | 20 |
| DE | 31 | 5 |

## 💾 Output Files

### 1. Model Files (in `models/`)
- `random_forest_model.pkl` - Trained RF classifier
- `isolation_forest_model.pkl` - Trained anomaly detector
- `feature_engineer.pkl` - Feature transformation pipeline

### 2. Predictions (in `output/`)
- `risk_predictions.csv` - Full predictions with explanations

Format:
```csv
Container_ID,Risk_Score,Risk_Level,Explanation,Confidence
97061800,15.42,Clear,Normal shipping pattern,98.5
85945189,72.31,Critical,Critical weight discrepancy (33.2%); High dwell time (51.7h),95.2
```

## 🐛 Troubleshooting

### Issue: Models not found
**Solution**: Run training first
```bash
python main.py --train
```

### Issue: Memory error on large datasets
**Solution**: Process in batches or reduce `SMOTE k_neighbors`

### Issue: Streamlit connection error
**Solution**: Ensure port 8501 is available
```bash
streamlit run dashboard.py --server.port 8502
```

### Issue: Import errors
**Solution**: Verify all dependencies installed
```bash
pip install -r requirements.txt --upgrade
```

## 📚 Documentation References

- [Scikit-learn RandomForest](https://scikit-learn.org/stable/modules/ensemble.html#forests)
- [SMOTE Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [Streamlit Docs](https://docs.streamlit.io)

## 👥 Team Contributors

- **Project**: HackaMINEd-2026 Hackathon
- **Solution**: SmartContainer Risk Engine
- **Focus**: Port Security & Container Risk Assessment

## 📄 License

This project is part of the HackaMINEd-2026 competition.

## 🚢 Contact & Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the code comments in relevant modules
3. Consult the feature engineering documentation

---

**Last Updated**: March 2026
**Model Version**: 1.0
**Data Version**: Historical + Real-time

