"""
Prediction Module for SmartContainer Risk Engine
Generates Risk Scores, Risk Levels, and Explanations
"""
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from utils import create_risk_level, get_explanation, load_data
import warnings
warnings.filterwarnings('ignore')

class RiskPredictor:
    """Generate risk predictions with explanations"""
    
    def __init__(self, model_path='models'):
        self.rf_model = None
        self.if_model = None
        self.feature_engineer = None
        self.risk_scaler = MinMaxScaler(feature_range=(0, 100))
        self.model_path = model_path
        self.load_models()
        
    def load_models(self):
        """Load trained models"""
        if os.path.exists(f'{self.model_path}/random_forest_model.pkl'):
            self.rf_model = pickle.load(open(f'{self.model_path}/random_forest_model.pkl', 'rb'))
            self.if_model = pickle.load(open(f'{self.model_path}/isolation_forest_model.pkl', 'rb'))
            self.feature_engineer = pickle.load(open(f'{self.model_path}/feature_engineer.pkl', 'rb'))
            print("[OK] Models loaded successfully")
        else:
            print("[WARNING] Models not found. Please train the model first.")
            
    def predict(self, df):
        """Generate predictions for new data"""
        
        print(f"\nGenerating predictions for {len(df)} containers...")
        
        # Keep original data for explanations
        df_original = df.copy()
        
        # Transform features
        X = self.feature_engineer.transform(df)
        
        # Drop non-feature columns
        cols_to_drop = ['Container_ID', 'Declaration_Date (YYYY-MM-DD)', 
                       'Declaration_Time', 'Importer_ID', 'Exporter_ID', 'Clearance_Status']
        X = X.drop(columns=cols_to_drop, errors='ignore')
        
        # Ensure all columns are numeric
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0)
        
        # Get class probabilities from Random Forest
        class_probs = self.rf_model.predict_proba(X)
        rf_predictions = self.rf_model.predict(X)
        
        # Map RF predictions to risk levels directly (highest accuracy)
        rf_label_map = {0: 'Clear', 1: 'Low Risk', 2: 'Critical'}
        risk_levels = [rf_label_map[p] for p in rf_predictions]
        
        # Calculate risk score for display (based on class probabilities)
        risk_scores_raw = (class_probs[:, 2] * 100 + class_probs[:, 1] * 50) / 1.5
        risk_scores = np.clip(risk_scores_raw, 0, 100)
        
        # Get anomaly scores from Isolation Forest
        anomaly_scores = -self.if_model.score_samples(X)
        anomaly_scores_norm = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-6) * 30
        
        # Combine scores: 70% RF prediction + 30% anomaly detection (for display score only)
        final_risk_scores = (risk_scores * 0.7 + anomaly_scores_norm * 0.3)
        final_risk_scores = np.clip(final_risk_scores, 0, 100)
        
        # Ensure risk score aligns with RF risk level
        # Critical: use max of combined score or probability-based score (50-100 range)
        # Low Risk: use max of combined score or probability-based score (20-50 range)
        for i, level in enumerate(risk_levels):
            if level == 'Critical':
                prob_score = 50 + (class_probs[i, 2] * 50)
                final_risk_scores[i] = max(final_risk_scores[i], prob_score)
            elif level == 'Low Risk':
                prob_score = 20 + (class_probs[i, 1] * 30)
                final_risk_scores[i] = max(final_risk_scores[i], prob_score)
        
        # Generate explanations using ORIGINAL raw data (not scaled)
        explanations = []
        for idx in range(len(df_original)):
            score = final_risk_scores[idx]
            raw_row = df_original.iloc[idx]
            
            explanation = self._generate_explanation(raw_row, score)
            explanations.append(explanation)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Container_ID': df_original['Container_ID'],
            'Risk_Score': np.round(final_risk_scores, 2),
            'Risk_Level': risk_levels,
            'Explanation': explanations,
            'Confidence': np.round(np.max(class_probs, axis=1) * 100, 2)
        })
        
        print(f"[OK] Predictions generated successfully!")
        print(f"\n[STAT] Risk Distribution:")
        print(results_df['Risk_Level'].value_counts())
        
        return results_df
    
    def _generate_explanation(self, row, risk_score):
        """Generate rule-based explanation from RAW (unscaled) row data"""
        explanations = []
        
        # --- Weight discrepancy (computed from raw declared vs measured) ---
        declared_w = row.get('Declared_Weight', 0)
        measured_w = row.get('Measured_Weight', 0)
        if declared_w and declared_w != 0:
            weight_diff_pct = abs(measured_w - declared_w) / declared_w * 100
        else:
            weight_diff_pct = 0.0

        if weight_diff_pct > 5:
            explanations.append(f"wt discrepancy {weight_diff_pct:.1f}%")
        
        # --- Dwell time analysis ---
        dwell = row.get('Dwell_Time_Hours', 0)
        if dwell > 120:
            explanations.append(f"extreme dwell {dwell:.1f}h")
        elif dwell > 72:
            explanations.append(f"high dwell {dwell:.1f}h")
        
        # --- Value-weight anomaly ---
        if measured_w and measured_w > 0:
            declared_val = row.get('Declared_Value', 0)
            vwr = declared_val / measured_w
            if vwr > 10000:
                explanations.append(f"high value/wt ratio ${vwr:,.0f}/kg")
            elif vwr > 5000:
                explanations.append(f"unusual value/wt ${vwr:,.0f}/kg")
        
        # --- Weight ratio anomaly (only if no weight discrepancy already) ---
        if weight_diff_pct <= 5 and declared_w and declared_w != 0:
            weight_ratio = measured_w / declared_w
            if weight_ratio > 1.2 or weight_ratio < 0.8:
                explanations.append(f"abnormal wt ratio {weight_ratio:.2f}")
        
        # --- Fallback context from risk score ---
        if risk_score >= 50 and len(explanations) == 0:
            explanations.append("combined risk indicators elevated")
        elif risk_score >= 20 and len(explanations) == 0:
            explanations.append("moderate risk detected")
        elif len(explanations) == 0:
            explanations.append("normal shipping pattern")
        
        return "; ".join(explanations[:2]) if explanations else "normal shipping pattern"
    
    def generate_report(self, predictions_df, output_path='output'):
        """Generate comprehensive report"""
        
        os.makedirs(output_path, exist_ok=True)
        
        # Save predictions to CSV
        output_file = f'{output_path}/risk_predictions.csv'
        try:
            predictions_df.to_csv(output_file, index=False)
        except PermissionError:
            output_file = f'{output_path}/risk_predictions_new.csv'
            predictions_df.to_csv(output_file, index=False)
        print(f"\n[SAVED] Predictions saved to {output_file}")
        
        # Generate summary report
        print("\n" + "="*80)
        print("RISK ASSESSMENT SUMMARY REPORT")
        print("="*80)
        
        print(f"\nTotal Containers Analyzed: {len(predictions_df)}")
        
        critical_count = len(predictions_df[predictions_df['Risk_Level'] == 'Critical'])
        low_risk_count = len(predictions_df[predictions_df['Risk_Level'] == 'Low Risk'])
        clear_count = len(predictions_df[predictions_df['Risk_Level'] == 'Clear'])
        
        print(f"\nRisk Distribution:")
        print(f"  [CRITICAL] Critical:  {critical_count:>5} ({critical_count/len(predictions_df)*100:>6.2f}%)")
        print(f"  [LOW_RISK] Low Risk:  {low_risk_count:>5} ({low_risk_count/len(predictions_df)*100:>6.2f}%)")
        print(f"  [CLEAR] Clear:     {clear_count:>5} ({clear_count/len(predictions_df)*100:>6.2f}%)")
        
        print(f"\nRisk Score Statistics:")
        print(f"  Mean:   {predictions_df['Risk_Score'].mean():.2f}")
        print(f"  Median: {predictions_df['Risk_Score'].median():.2f}")
        print(f"  Min:    {predictions_df['Risk_Score'].min():.2f}")
        print(f"  Max:    {predictions_df['Risk_Score'].max():.2f}")
        print(f"  Std:    {predictions_df['Risk_Score'].std():.2f}")
        
        print(f"\nTop 10 Highest Risk Containers:")
        top_risk = predictions_df.nlargest(10, 'Risk_Score')[['Container_ID', 'Risk_Score', 'Risk_Level', 'Explanation']]
        for idx, row in top_risk.iterrows():
            print(f"  {row['Container_ID']}: {row['Risk_Score']:.2f} ({row['Risk_Level']}) - {row['Explanation']}")
        
        print("\n" + "="*80)
        
        return predictions_df

def predict_realtime_data():
    """Predict on realtime data"""
    
    print("\n" + "="*80)
    print("PREDICTING ON REALTIME DATA")
    print("="*80)
    
    # Load realtime data
    print("\nLoading realtime data...")
    df_realtime = load_data('data/realtime_data.csv')
    
    # Initialize predictor
    predictor = RiskPredictor()
    
    # Make predictions
    predictions = predictor.predict(df_realtime)
    
    # Generate report
    predictor.generate_report(predictions)
    
    return predictions

def predict_historical_data():
    """Predict on historical data for validation"""
    
    print("\n" + "="*80)
    print("PREDICTING ON HISTORICAL (TEST) DATA")
    print("="*80)
    
    # Load historical data
    print("\nLoading historical data...")
    df_hist = load_data('data/historical_data.csv')
    
    # Initialize predictor
    predictor = RiskPredictor()
    
    # Make predictions
    predictions = predictor.predict(df_hist)
    
    # Generate report
    predictor.generate_report(predictions)
    
    return predictions

if __name__ == "__main__":
    # Predict on both datasets
    hist_predictions = predict_historical_data()
    realtime_predictions = predict_realtime_data()
