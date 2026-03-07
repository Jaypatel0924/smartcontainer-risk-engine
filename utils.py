"""
Utility functions for SmartContainer Risk Engine
"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load CSV data"""
    df = pd.read_csv(filepath)
    return df

def preprocess_datetime(df):
    """Extract hour from Declaration_Time"""
    df['Declaration_Hour'] = pd.to_datetime(df['Declaration_Time'], format='%H:%M:%S').dt.hour
    return df

def calculate_weight_discrepancy(df):
    """Calculate weight difference percentage"""
    df['Weight_Diff_%'] = ((df['Measured_Weight'] - df['Declared_Weight']) / 
                           df['Declared_Weight'].replace(0, 1)).abs() * 100
    df['Weight_Diff_%'] = df['Weight_Diff_%'].fillna(0)
    df['Weight_Diff_%'] = df['Weight_Diff_%'].replace([np.inf, -np.inf], 0)
    return df

def calculate_weight_ratio(df):
    """Calculate measured to declared weight ratio"""
    df['Weight_Ratio'] = (df['Measured_Weight'] / 
                          df['Declared_Weight'].replace(0, 1))
    df['Weight_Ratio'] = df['Weight_Ratio'].fillna(1)
    df['Weight_Ratio'] = df['Weight_Ratio'].replace([np.inf, -np.inf], 1)
    return df

def calculate_value_weight_ratio(df):
    """Calculate value to weight ratio (anomaly indicator)"""
    df['Value_Weight_Ratio'] = (df['Declared_Value'] / 
                                df['Measured_Weight'].replace(0, 1))
    df['Value_Weight_Ratio'] = df['Value_Weight_Ratio'].fillna(0)
    df['Value_Weight_Ratio'] = df['Value_Weight_Ratio'].replace([np.inf, -np.inf], 0)
    return df

def create_dwell_time_flags(df):
    """Create flags for high dwell times"""
    df['High_Dwell'] = (df['Dwell_Time_Hours'] > 72).astype(int)
    df['Very_High_Dwell'] = (df['Dwell_Time_Hours'] > 120).astype(int)
    return df

def encode_categorical(df, categorical_cols, fit_encoder=None):
    """Encode categorical variables"""
    from sklearn.preprocessing import LabelEncoder
    
    encoders = {}
    if fit_encoder is None:
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in categorical_cols:
            # Handle unknown categories in test data
            try:
                df[col] = fit_encoder[col].transform(df[col].astype(str))
            except ValueError:
                # For unknown labels, assign default value (0)
                df[col] = df[col].astype(str).map(
                    {c: i for i, c in enumerate(fit_encoder[col].classes_)}
                ).fillna(-1).astype(int)
                # Replace -1 with 0 (unknown categories get 0)
                df[col] = df[col].replace(-1, 0)
    
    return df, encoders

def get_explanation(row, feature_importance, top_n=2):
    """Generate explanation for prediction"""
    explanations = []
    
    # Weight discrepancy explanation
    if row['Weight_Diff_%'] > 20:
        explanations.append(f"Weight discrepancy {row['Weight_Diff_%']:.1f}%")
    
    # Dwell time explanation
    if row['Dwell_Time_Hours'] > 72:
        explanations.append(f"High dwell time {row['Dwell_Time_Hours']:.0f}h")
    
    # Value anomaly
    if row['Value_Weight_Ratio'] > row['Value_Weight_Ratio'].quantile(0.95):
        explanations.append("Unusual value-weight ratio")
    
    # High risk origin
    critical_origins = ['CN', 'RO', 'VN']  # Example origins
    if row.get('Origin_Country') in critical_origins and row.get('Risk_Score', 0) > 50:
        explanations.append("High-risk origin country")
    
    # High-risk port
    critical_ports = ['PORT_71', 'PORT_130', 'PORT_131']
    if row.get('Destination_Port') in critical_ports:
        explanations.append("High-risk destination port")
    
    # Return top 2 explanations
    if len(explanations) == 0:
        return "Normal shipping pattern"
    return "; ".join(explanations[:top_n])

def create_risk_level(score):
    """Convert score (0-100) to risk level"""
    if score >= 50:
        return "Critical"
    elif score >= 20:
        return "Low Risk"
    else:
        return "Clear"

def generate_report_stats(df_pred):
    """Generate summary statistics for dashboard"""
    stats = {
        'total_containers': len(df_pred),
        'critical_count': len(df_pred[df_pred['Risk_Level'] == 'Critical']),
        'low_risk_count': len(df_pred[df_pred['Risk_Level'] == 'Low Risk']),
        'clear_count': len(df_pred[df_pred['Risk_Level'] == 'Clear']),
        'critical_pct': (len(df_pred[df_pred['Risk_Level'] == 'Critical']) / len(df_pred) * 100) if len(df_pred) > 0 else 0,
        'low_risk_pct': (len(df_pred[df_pred['Risk_Level'] == 'Low Risk']) / len(df_pred) * 100) if len(df_pred) > 0 else 0,
        'clear_pct': (len(df_pred[df_pred['Risk_Level'] == 'Clear']) / len(df_pred) * 100) if len(df_pred) > 0 else 0,
        'avg_risk_score': df_pred['Risk_Score'].mean(),
    }
    return stats
