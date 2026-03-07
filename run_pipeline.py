"""
Quick-Start Execution Script for SmartContainer Risk Engine
Run the complete pipeline in one command
"""

import sys
import os
import subprocess

def run_command(cmd, description):
    """Run a command and report status"""
    print("\n" + "="*80)
    print(f"[STEP] {description}")
    print("="*80)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
        if result.returncode == 0:
            print(f"[SUCCESS] {description} completed successfully")
            return True
        else:
            print(f"[ERROR] {description} failed with code {result.returncode}")
            return False
    except Exception as e:
        print(f"[ERROR] {description} failed: {str(e)}")
        return False

def main():
    """Main execution"""
    
    print("\n" + "#"*80)
    print("# SMARTCONTAINER RISK ENGINE - COMPLETE PIPELINE")
    print("# HackaMINEd-2026 Hackathon Solution")
    print("#"*80)
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    # Python executable
    python_exe = ".\\venv\\Scripts\\python.exe" if os.name == 'nt' else "python"
    
    steps = [
        (f"{python_exe} model_training.py", "Train ML Models with SMOTE"),
        (f"{python_exe} predict.py", "Generate Risk Predictions"),
    ]
    
    completed = 0
    for cmd, desc in steps:
        if run_command(cmd, desc):
            completed += 1
        else:
            print(f"\n[ABORT] Pipeline stopped due to error")
            return False
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"\nCompleted Steps: {completed}/{len(steps)}")
    
    if completed == len(steps):
        print("\n[SUCCESS] All pipeline steps completed successfully!")
        print("\nGenerated Files:")
        print("  - models/random_forest_model.pkl")
        print("  - models/isolation_forest_model.pkl")
        print("  - models/feature_engineer.pkl")
        print("  - output/risk_predictions.csv")
        
        print("\nNext Steps:")
        print("  1. Review predictions: output/risk_predictions.csv")
        print("  2. Launch dashboard: streamlit run dashboard.py")
        print("  3. Make new predictions: python predict.py")
        
        return True
    else:
        print("\n[FAILED] Pipeline execution failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
