"""
Feature Engineering Module for SmartContainer Risk Engine
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import *
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Comprehensive feature engineering for container risk prediction"""
    
    def __init__(self):
        self.scalar = StandardScaler()
        self.encoders = {}
        self.categorical_cols = ['Trade_Regime (Import / Export / Transit)', 'Origin_Country', 
                                'Destination_Port', 'Destination_Country', 'HS_Code', 'Shipping_Line']
        self.numeric_features = []
        
    def fit_transform(self, df):
        """Fit encoders and transform training data"""
        df = df.copy()
        
        # Datetime features
        df = preprocess_datetime(df)
        
        # Weight-based features
        df = calculate_weight_discrepancy(df)
        df = calculate_weight_ratio(df)
        df = calculate_value_weight_ratio(df)
        
        # Dwell time flags
        df = create_dwell_time_flags(df)
        
        # Additional features
        df = self._create_additional_features(df)
        
        # Encode categorical features
        df_encoded, self.encoders = encode_categorical(df, self.categorical_cols)
        
        # Select and scale numeric features
        self.numeric_features = ['Weight_Diff_%', 'Weight_Ratio', 'Value_Weight_Ratio',
                                'Dwell_Time_Hours', 'High_Dwell', 'Very_High_Dwell',
                                'Declared_Value', 'Measured_Weight', 'Declaration_Hour',
                                'HS_Code_Risk_Score', 'Origin_Risk_Score', 'Port_Risk_Score']
        
        # Add encoded categorical features
        self.numeric_features.extend(self.categorical_cols)
        
        # Scale numeric features (except encoded categoricals)
        numeric_only = ['Weight_Diff_%', 'Weight_Ratio', 'Value_Weight_Ratio',
                       'Dwell_Time_Hours', 'High_Dwell', 'Very_High_Dwell',
                       'Declared_Value', 'Measured_Weight', 'Declaration_Hour']
        
        df_encoded[numeric_only] = self.scalar.fit_transform(df_encoded[numeric_only])
        
        return df_encoded
    
    def transform(self, df):
        """Transform test/inference data"""
        df = df.copy()
        
        # Apply same transformations
        df = preprocess_datetime(df)
        df = calculate_weight_discrepancy(df)
        df = calculate_weight_ratio(df)
        df = calculate_value_weight_ratio(df)
        df = create_dwell_time_flags(df)
        df = self._create_additional_features(df)
        
        # Encode using fitted encoders
        df, _ = encode_categorical(df, self.categorical_cols, self.encoders)
        
        # Scale using fitted scaler
        numeric_only = ['Weight_Diff_%', 'Weight_Ratio', 'Value_Weight_Ratio',
                       'Dwell_Time_Hours', 'High_Dwell', 'Very_High_Dwell',
                       'Declared_Value', 'Measured_Weight', 'Declaration_Hour']
        
        df[numeric_only] = self.scalar.transform(df[numeric_only])
        
        return df
    
    def _create_additional_features(self, df):
        """Create domain-specific risk indicators"""
        
        # HS Code risk scoring (based on historical patterns)
        hs_code_risk = {
            '390690': 15, '620640': 12, '940360': 18, '851712': 20,
            '620463': 14, '841320': 11, '851660': 19, '620822': 13,
            '690722': 16, '440890': 10, '854442': 25, '852580': 22,
        }
        
        df['HS_Code_Risk_Score'] = df['HS_Code'].map(hs_code_risk).fillna(10)
        
        # Origin country risk scoring
        high_risk_origins = {'CN': 25, 'RO': 22, 'VN': 20, 'ID': 18, 'JP': 15}
        medium_risk = {'US': 8, 'IT': 7, 'DE': 5, 'UK': 5, 'CA': 6}
        
        origin_risk = {**high_risk_origins, **medium_risk}
        df['Origin_Risk_Score'] = df['Origin_Country'].map(origin_risk).fillna(10)
        
        # Port risk scoring
        high_risk_ports = {
            'PORT_71': 25, 'PORT_130': 22, 'PORT_131': 20, 'PORT_120': 18,
            'PORT_90': 15, 'PORT_37': 14, 'PORT_62': 13, 'PORT_17': 12,
            'PORT_40': 10, 'PORT_20': 8, 'PORT_30': 7, 'PORT_10': 5
        }
        
        df['Port_Risk_Score'] = df['Destination_Port'].map(high_risk_ports).fillna(5)
        
        # Trade regime risk
        trade_risk = {'Import': 15, 'Export': 8, 'Transit': 12}
        df['Trade_Risk_Score'] = df['Trade_Regime (Import / Export / Transit)'].map(trade_risk).fillna(10)
        
        # Composite anomaly score
        df['Anomaly_Score'] = (
            0.35 * df['Weight_Diff_%'] + 
            0.25 * df['HS_Code_Risk_Score'] + 
            0.20 * df['Origin_Risk_Score'] + 
            0.15 * df['Port_Risk_Score'] + 
            0.05 * df['High_Dwell']
        )
        
        return df
    
    def get_feature_list(self):
        """Return list of engineered features"""
        return self.numeric_features

def prepare_training_data(df):
    """Prepare data for model training"""
    
    # Drop unnecessary columns
    cols_to_drop = ['Container_ID', 'Declaration_Date (YYYY-MM-DD)', 'Declaration_Time',
                   'Importer_ID', 'Exporter_ID']
    
    X = df.drop(columns=cols_to_drop + ['Clearance_Status'], errors='ignore')
    y = df['Clearance_Status'].map({'Clear': 0, 'Low Risk': 1, 'Critical': 2})
    
    return X, y

if __name__ == "__main__":
    from utils import load_data
    
    print("Testing Feature Engineering...")
    df = load_data('data/historical_data.csv')
    
    fe = FeatureEngineer()
    X_transformed = fe.fit_transform(df)
    
    print(f"Original shape: {df.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    print(f"Features created: {fe.numeric_features}")
    print(f"\nFeature statistics:")
    print(X_transformed.describe())
