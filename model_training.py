"""
Model Training Module with SMOTE for SmartContainer Risk Engine
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score)
from imblearn.over_sampling import SMOTE
from utils import load_data
from feature_engineering import FeatureEngineer, prepare_training_data
import warnings
warnings.filterwarnings('ignore')

class SmartContainerRiskModel:
    """ML Model for SmartContainer Risk Prediction with SMOTE"""
    
    def __init__(self):
        self.rf_model = None
        self.if_model = None
        self.feature_engineer = None
        self.label_encoder = None
        self.feature_names = None
        self.is_trained = False
        
    def prepare_data(self, df):
        """Prepare and engineer features"""
        print("\n🔧 Preparing data and engineering features...")
        
        self.feature_engineer = FeatureEngineer()
        X = self.feature_engineer.fit_transform(df)
        
        # Drop non-feature columns
        cols_to_drop = ['Container_ID', 'Declaration_Date (YYYY-MM-DD)', 
                       'Declaration_Time', 'Importer_ID', 'Exporter_ID', 'Clearance_Status']
        X = X.drop(columns=cols_to_drop, errors='ignore')
        
        # Ensure all columns are numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0)
        
        # Prepare target variable
        y = df['Clearance_Status'].map({'Clear': 0, 'Low Risk': 1, 'Critical': 2})
        
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def handle_imbalance(self, X_train, y_train):
        """Handle imbalanced data using SMOTE"""
        print("\n⚖️  Handling Imbalanced Data with SMOTE...")
        
        print(f"   Original training set distribution:")
        print(f"   {y_train.value_counts().to_dict()}")
        
        # Apply SMOTE
        sm = SMOTE(random_state=42, k_neighbors=5)
        X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)
        
        print(f"\n   After SMOTE resampling:")
        print(f"   {pd.Series(y_train_resampled).value_counts().to_dict()}")
        
        return X_train_resampled, y_train_resampled
    
    def clean_data(self, df):
        """Remove rows with invalid/zero Declared_Value or Declared_Weight"""
        original_len = len(df)
        
        # Drop rows where Declared_Value is 0 (invalid declarations)
        df = df[df['Declared_Value'] > 0].copy()
        dropped_val = original_len - len(df)
        
        # Drop rows where Declared_Weight is 0 (invalid declarations)
        before = len(df)
        df = df[df['Declared_Weight'] > 0].copy()
        dropped_wt = before - len(df)
        
        df = df.reset_index(drop=True)
        
        print(f"\n   Data Cleaning:")
        print(f"   Dropped {dropped_val} rows with Declared_Value = 0")
        print(f"   Dropped {dropped_wt} rows with Declared_Weight = 0")
        print(f"   Remaining: {len(df)} rows (from {original_len})")
        
        return df
    
    def train(self, df, test_size=0.2, random_state=42):
        """Train the model with Random Forest and Isolation Forest"""
        
        print("\n" + "="*80)
        print("SMARTCONTAINER RISK ENGINE - MODEL TRAINING")
        print("="*80)
        
        # Clean data: drop invalid rows
        df = self.clean_data(df)
        
        # Prepare data
        X, y = self.prepare_data(df)
        
        print(f"\nDataset prepared:")
        print(f"   Features: {X.shape[1]}")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Classes: {y.unique()}")
        print(f"   Class distribution:\n{y.value_counts()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Handle imbalance with SMOTE
        X_train_resampled, y_train_resampled = self.handle_imbalance(X_train, y_train)
        
        # Train Random Forest
        print("\n🌲 Training Random Forest Classifier...")
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=30,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.rf_model.fit(X_train_resampled, y_train_resampled)
        
        # Evaluate Random Forest
        rf_pred = self.rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        
        print(f"\n✓ Random Forest trained successfully!")
        print(f"   Accuracy: {rf_accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, rf_pred, 
                                   target_names=['Clear', 'Low Risk', 'Critical']))
        
        # Train Isolation Forest for anomaly detection
        print("\n🔍 Training Isolation Forest for Anomaly Detection...")
        self.if_model = IsolationForest(
            contamination=0.03,
            random_state=random_state,
            n_estimators=200,
            max_features=1.0
        )
        
        self.if_model.fit(X_train_resampled)
        
        # Cross-validation
        print("\n📊 Cross-Validation Results:")
        cv_scores = cross_val_score(self.rf_model, X_train_resampled, y_train_resampled, 
                                   cv=5, scoring='f1_weighted')
        print(f"   CV F1 Scores: {cv_scores}")
        print(f"   Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Feature importance
        print("\n🎯 Top 10 Most Important Features:")
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(feature_importance.head(10).to_string(index=False))
        
        # Detailed test metrics
        print("\n" + "="*80)
        print("DETAILED TEST SET METRICS")
        print("="*80)
        print(f"Accuracy: {rf_accuracy:.4f}")
        print(f"Precision (weighted): {precision_score(y_test, rf_pred, average='weighted'):.4f}")
        print(f"Recall (weighted): {recall_score(y_test, rf_pred, average='weighted'):.4f}")
        print(f"F1-Score (weighted): {f1_score(y_test, rf_pred, average='weighted'):.4f}")
        
        cm = confusion_matrix(y_test, rf_pred)
        print(f"\nConfusion Matrix:\n{cm}")
        
        self.is_trained = True
        
        return {
            'model': self.rf_model,
            'anomaly_model': self.if_model,
            'accuracy': rf_accuracy,
            'feature_importance': feature_importance,
            'X_test': X_test,
            'y_test': y_test,
            'rf_pred': rf_pred
        }
    
    def save_model(self, model_path='models'):
        """Save trained models"""
        os.makedirs(model_path, exist_ok=True)
        
        pickle.dump(self.rf_model, open(f'{model_path}/random_forest_model.pkl', 'wb'))
        pickle.dump(self.if_model, open(f'{model_path}/isolation_forest_model.pkl', 'wb'))
        pickle.dump(self.feature_engineer, open(f'{model_path}/feature_engineer.pkl', 'wb'))
        
        print(f"\n✓ Models saved to {model_path}/")
        
    def load_model(self, model_path='models'):
        """Load pre-trained models"""
        self.rf_model = pickle.load(open(f'{model_path}/random_forest_model.pkl', 'rb'))
        self.if_model = pickle.load(open(f'{model_path}/isolation_forest_model.pkl', 'rb'))
        self.feature_engineer = pickle.load(open(f'{model_path}/feature_engineer.pkl', 'rb'))
        
        self.is_trained = True
        print(f"✓ Models loaded from {model_path}/")

def train_model_pipeline():
    """Main training pipeline"""
    
    # Load historical data
    print("📥 Loading historical data...")
    df_hist = load_data('data/historical_data.csv')
    
    # Initialize and train model
    model = SmartContainerRiskModel()
    results = model.train(df_hist)
    
    # Save model
    model.save_model()
    
    # Validate predictions vs actual labels on full historical data
    validate_predictions(df_hist, model)
    
    return model, results

def validate_predictions(df_hist, model):
    """Validate model predictions against actual Clearance_Status in historical data"""
    from predict import RiskPredictor
    
    print("\n" + "="*80)
    print("VALIDATION: Predictions vs Actual Labels (Historical Data)")
    print("="*80)
    
    # Clean same way as training
    df_clean = df_hist[(df_hist['Declared_Value'] > 0) & (df_hist['Declared_Weight'] > 0)].copy()
    
    predictor = RiskPredictor()
    preds = predictor.predict(df_clean)
    
    actual = df_clean['Clearance_Status'].values
    predicted = preds['Risk_Level'].values
    
    # Compute match
    matches = (actual == predicted).sum()
    total = len(actual)
    accuracy = matches / total * 100
    
    print(f"\n   Total containers validated: {total}")
    print(f"   Correct predictions: {matches}")
    print(f"   Prediction Accuracy: {accuracy:.2f}%")
    
    # Per-class breakdown
    for label in ['Clear', 'Low Risk', 'Critical']:
        mask = actual == label
        label_total = mask.sum()
        label_correct = ((actual == predicted) & mask).sum()
        if label_total > 0:
            print(f"   {label}: {label_correct}/{label_total} correct ({label_correct/label_total*100:.1f}%)")
    
    # Misclassification analysis
    mismatches = actual != predicted
    if mismatches.sum() > 0:
        print(f"\n   Misclassification Summary ({mismatches.sum()} containers):")
        mismatch_df = pd.DataFrame({'Actual': actual[mismatches], 'Predicted': predicted[mismatches]})
        print(mismatch_df.groupby(['Actual', 'Predicted']).size().to_string())

if __name__ == "__main__":
    model, results = train_model_pipeline()
