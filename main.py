"""
Main Execution Script for SmartContainer Risk Engine
Full Pipeline: EDA → Feature Engineering → Model Training → Prediction
"""

import sys
import os

# Ensure project directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eda_analysis import exploratory_data_analysis
from model_training import train_model_pipeline
from predict import predict_realtime_data, predict_historical_data
import argparse

def main():
    """Main execution pipeline"""
    
    parser = argparse.ArgumentParser(description='SmartContainer Risk Engine')
    parser.add_argument('--eda', action='store_true', help='Run EDA analysis')
    parser.add_argument('--train', action='store_true', help='Train ML models')
    parser.add_argument('--predict', action='store_true', help='Generate predictions')
    parser.add_argument('--dashboard', action='store_true', help='Launch Streamlit dashboard')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    
    args = parser.parse_args()
    
    # Default to all if no args
    if not any([args.eda, args.train, args.predict, args.dashboard, args.all]):
        args.all = True
    
    print("\n" + "="*80)
    print("SMARTCONTAINER RISK ENGINE - FULL PIPELINE EXECUTION")
    print("="*80)
    
    # Step 1: EDA
    if args.eda or args.all:
        print("\n" + "→"*40)
        print("STEP 1: EXPLORATORY DATA ANALYSIS")
        print("→"*40)
        try:
            hist_data, hist_encoded = exploratory_data_analysis('data/historical_data.csv')
        except Exception as e:
            print(f"❌ EDA failed: {str(e)}")
    
    # Step 2: Model Training
    if args.train or args.all:
        print("\n" + "→"*40)
        print("STEP 2: MODEL TRAINING WITH SMOTE")
        print("→"*40)
        try:
            model, results = train_model_pipeline()
            print("\n✓ Model training completed successfully!")
        except Exception as e:
            print(f"❌ Training failed: {str(e)}")
    
    # Step 3: Predictions
    if args.predict or args.all:
        print("\n" + "→"*40)
        print("STEP 3: GENERATING PREDICTIONS")
        print("→"*40)
        try:
            print("\n1️⃣  Predicting on Historical Data (Test Set)...")
            hist_pred = predict_historical_data()
            
            print("\n2️⃣  Predicting on Real-time Data...")
            realtime_pred = predict_realtime_data()
            
            print("\n✓ Predictions completed successfully!")
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
    
    # Step 4: Dashboard
    if args.dashboard:
        print("\n" + "→"*40)
        print("STEP 4: LAUNCHING STREAMLIT DASHBOARD")
        print("→"*40)
        print("Starting Streamlit server...")
        print("Open your browser and navigate to: http://localhost:8501")
        
        os.system("streamlit run dashboard.py")
    
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETED")
    print("="*80)
    
    print("\n📁 Output Files Generated:")
    print("   ├── models/")
    print("   │   ├── random_forest_model.pkl")
    print("   │   ├── isolation_forest_model.pkl")
    print("   │   └── feature_engineer.pkl")
    print("   └── output/")
    print("       ├── risk_predictions.csv")
    print("       └── [Additional reports]")
    
    print("\n🚀 Next Steps:")
    print("   1. Review predictions in output/risk_predictions.csv")
    print("   2. Launch dashboard: python main.py --dashboard")
    print("   3. Make predictions on new data: python predict.py")

if __name__ == "__main__":
    main()
