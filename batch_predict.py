"""
Batch Prediction Script
Make predictions on new CSV files
"""

import sys
import os
import pandas as pd
from predict import RiskPredictor

def predict_csv_file(input_file, output_file=None):
    """
    Make predictions on a CSV file
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save predictions (optional)
    
    Returns:
        DataFrame with predictions
    """
    
    print(f"\n[INFO] Loading data from {input_file}...")
    
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {input_file}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load file: {str(e)}")
        return None
    
    print(f"[INFO] Loaded {len(df)} containers")
    
    # Initialize predictor
    print("[INFO] Loading trained models...")
    predictor = RiskPredictor()
    
    # Make predictions
    print("[INFO] Generating predictions...")
    predictions = predictor.predict(df)
    
    # Save if output file specified
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        predictions.to_csv(output_file, index=False)
        print(f"[SUCCESS] Predictions saved to {output_file}")
    
    # Generate report
    predictor.generate_report(predictions)
    
    return predictions

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python batch_predict.py <input_csv> [output_csv]")
        print("\nExample:")
        print("  python batch_predict.py data/new_containers.csv output/predictions.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    predictions = predict_csv_file(input_file, output_file)
    
    if predictions is not None:
        print("\n[OK] Batch prediction completed successfully")
        sys.exit(0)
    else:
        print("\n[ERROR] Batch prediction failed")
        sys.exit(1)
