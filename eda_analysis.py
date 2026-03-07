"""
EDA and Data Analysis for SmartContainer Risk Engine
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, preprocess_datetime, calculate_weight_discrepancy, calculate_weight_ratio
from utils import calculate_value_weight_ratio, create_dwell_time_flags
import warnings
warnings.filterwarnings('ignore')

def exploratory_data_analysis(filepath):
    """Perform comprehensive EDA"""
    print("=" * 80)
    print("SMARTCONTAINER RISK ENGINE - EDA ANALYSIS")
    print("=" * 80)
    
    df = load_data(filepath)
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\n" + "=" * 80)
    print("1. DATA OVERVIEW")
    print("=" * 80)
    print("\nColumn Names and Types:")
    print(df.dtypes)
    
    print("\n2. MISSING VALUES")
    print("=" * 80)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("✓ No missing values found")
    
    print("\n3. TARGET VARIABLE DISTRIBUTION (Clearance_Status)")
    print("=" * 80)
    target_dist = df['Clearance_Status'].value_counts()
    print(target_dist)
    print(f"\nClass Distribution (%):")
    for class_name, count in target_dist.items():
        pct = (count / len(df)) * 100
        print(f"   {class_name}: {count} ({pct:.2f}%)")
    
    # Check for imbalance
    if target_dist.min() < target_dist.max() * 0.1:
        print("\n⚠️  IMBALANCED DATA DETECTED!")
        print("   -> Will apply SMOTE for handling class imbalance")
    
    print("\n4. NUMERICAL FEATURES STATISTICS")
    print("=" * 80)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numerical_cols].describe())
    
    print("\n5. CATEGORICAL FEATURES")
    print("=" * 80)
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"\n{col}:")
        print(f"   Unique values: {df[col].nunique()}")
        print(f"   Top 5 values: {dict(df[col].value_counts().head(5))}")
    
    print("\n6. DUPLICATE CONTAINERS")
    print("=" * 80)
    duplicates = df.duplicated(subset=['Container_ID']).sum()
    print(f"   Duplicate Container_IDs: {duplicates}")
    
    # Feature Engineering Analysis
    print("\n7. FEATURE ENGINEERING ANALYSIS")
    print("=" * 80)
    
    df = preprocess_datetime(df)
    df = calculate_weight_discrepancy(df)
    df = calculate_weight_ratio(df)
    df = calculate_value_weight_ratio(df)
    df = create_dwell_time_flags(df)
    
    print("\nGenerated Features:")
    print(f"   Weight_Diff_%: {df['Weight_Diff_%'].describe()}")
    print(f"\n   Weight_Ratio: {df['Weight_Ratio'].describe()}")
    print(f"\n   Value_Weight_Ratio: {df['Value_Weight_Ratio'].describe()}")
    print(f"\n   High_Dwell (>72h): {df['High_Dwell'].sum()} containers")
    print(f"   Very_High_Dwell (>120h): {df['Very_High_Dwell'].sum()} containers")
    
    print("\n8. ANOMALY INDICATORS")
    print("=" * 80)
    print(f"   Weight discrepancy > 10%: {(df['Weight_Diff_%'] > 10).sum()} containers")
    print(f"   Weight discrepancy > 20%: {(df['Weight_Diff_%'] > 20).sum()} containers")
    print(f"   Weight ratio anomalies (> 1.2 or < 0.8): {((df['Weight_Ratio'] > 1.2) | (df['Weight_Ratio'] < 0.8)).sum()} containers")
    
    print("\n9. ORIGIN AND DESTINATION ANALYSIS")
    print("=" * 80)
    print("\nTop 10 Origins:")
    print(df['Origin_Country'].value_counts().head(10))
    
    print("\nTop 10 Destination Ports:")
    print(df['Destination_Port'].value_counts().head(10))
    
    # Correlation with target
    print("\n10. CORRELATION WITH TARGET (Clearance_Status)")
    print("=" * 80)
    df_encoded = df.copy()
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df_encoded['Clearance_Status_encoded'] = le.fit_transform(df_encoded['Clearance_Status'])
    
    numeric_features = ['Weight_Diff_%', 'Weight_Ratio', 'Value_Weight_Ratio', 
                       'Dwell_Time_Hours', 'High_Dwell', 'Very_High_Dwell',
                       'Declared_Value', 'Measured_Weight', 'Declaration_Hour']
    
    correlations = df_encoded[numeric_features + ['Clearance_Status_encoded']].corr()['Clearance_Status_encoded'].sort_values(ascending=False)
    print("\nFeature Correlations with Risk Level:")
    print(correlations)
    
    print("\n" + "=" * 80)
    print("EDA ANALYSIS COMPLETE")
    print("=" * 80)
    
    return df, df_encoded

if __name__ == "__main__":
    # Run EDA on historical data
    hist_data, hist_encoded = exploratory_data_analysis('data/historical_data.csv')
    
    # Also check realtime data
    print("\n\n" + "=" * 80)
    print("REALTIME DATA EDA")
    print("=" * 80)
    realtime_data, realtime_encoded = exploratory_data_analysis('data/realtime_data.csv')
