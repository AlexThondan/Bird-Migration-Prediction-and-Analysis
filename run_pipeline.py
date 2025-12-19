"""
Bird Migration Project Pipeline Runner
=====================================

This script runs the complete ML pipeline for bird migration analysis including
EDA, preprocessing, training, and MLflow tracking.

Author: Bird Migration Research Team
Date: November 2025
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n🚀 Running: {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
        else:
            print(f"❌ {description} failed!")
            print("STDERR:", result.stderr[-500:])  # Last 500 chars
            return False
            
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False
    
    return True

def setup_environment():
    """Setup required directories and environment"""
    print("🔧 Setting up environment...")
    
    directories = ['results', 'models', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✓ Directory: {directory}")
    
    return True

def run_complete_pipeline():
    """Run the complete ML pipeline"""
    print("🐦 Bird Migration ML Pipeline Runner")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now()}")
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed!")
        return False
    
    # Pipeline steps
    pipeline_steps = [
        {
            'script': 'scripts/01_eda_analysis.py',
            'description': 'Exploratory Data Analysis (EDA)',
            'required': True
        },
        {
            'script': 'scripts/02_data_preprocessing.py', 
            'description': 'Data Preprocessing Pipeline',
            'required': True
        },
        {
            'script': 'scripts/03_model_training.py',
            'description': 'ML Model Training Pipeline',
            'required': True
        },
        {
            'script': 'scripts/04_mlflow_tracker.py',
            'description': 'MLflow Experiment Analysis',
            'required': False
        }
    ]
    
    success_count = 0
    failed_steps = []
    
    # Run each step
    for step in pipeline_steps:
        script_path = step['script']
        
        if not os.path.exists(script_path):
            print(f"⚠️  Script not found: {script_path}")
            if step['required']:
                failed_steps.append(step['description'])
            continue
        
        success = run_script(script_path, step['description'])
        
        if success:
            success_count += 1
        elif step['required']:
            failed_steps.append(step['description'])
            print(f"❌ Required step failed: {step['description']}")
            # Don't break - continue with other steps
    
    # Pipeline summary
    print("\n" + "="*80)
    print("🎯 PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"✅ Successfully completed: {success_count}/{len(pipeline_steps)} steps")
    print(f"📅 Finished at: {datetime.now()}")
    
    if failed_steps:
        print(f"\n❌ Failed steps:")
        for step in failed_steps:
            print(f"   • {step}")
    else:
        print("\n🎉 All pipeline steps completed successfully!")
    
    # Next steps information
    print("\n📋 Next Steps:")
    print("   1. 🌐 Start MLflow UI: mlflow ui --backend-store-uri sqlite:///mlruns.db")
    print("   2. 📊 View experiment results in the MLflow dashboard")
    print("   3. 📁 Check 'results/' directory for analysis outputs")
    print("   4. 🤖 Check 'models/' directory for trained models")
    
    return len(failed_steps) == 0

def run_individual_step(step_name):
    """Run an individual pipeline step"""
    scripts_map = {
        'eda': 'scripts/01_eda_analysis.py',
        'preprocess': 'scripts/02_data_preprocessing.py',
        'train': 'scripts/03_model_training.py',
        'track': 'scripts/04_mlflow_tracker.py'
    }
    
    if step_name not in scripts_map:
        print(f"❌ Unknown step: {step_name}")
        print(f"Available steps: {list(scripts_map.keys())}")
        return False
    
    script_path = scripts_map[step_name]
    description = {
        'eda': 'Exploratory Data Analysis',
        'preprocess': 'Data Preprocessing', 
        'train': 'Model Training',
        'track': 'MLflow Tracking Analysis'
    }[step_name]
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False
    
    return run_script(script_path, description)

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Bird Migration ML Pipeline Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Run complete pipeline
  python run_pipeline.py --step eda         # Run only EDA step
  python run_pipeline.py --step preprocess  # Run only preprocessing step
  python run_pipeline.py --step train       # Run only training step
  python run_pipeline.py --step track       # Run only MLflow analysis step
        """
    )
    
    parser.add_argument(
        '--step', 
        choices=['eda', 'preprocess', 'train', 'track'],
        help='Run a specific pipeline step instead of the complete pipeline'
    )
    
    args = parser.parse_args()
    
    if args.step:
        # Run individual step
        success = run_individual_step(args.step)
        if success:
            print(f"\n🎉 Step '{args.step}' completed successfully!")
        else:
            print(f"\n❌ Step '{args.step}' failed!")
            sys.exit(1)
    else:
        # Run complete pipeline
        success = run_complete_pipeline()
        if not success:
            sys.exit(1)

if __name__ == "__main__":
    main()