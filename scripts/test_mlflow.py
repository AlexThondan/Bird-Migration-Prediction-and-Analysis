"""
MLflow Test Script - Verify MLflow functionality
===============================================

This script tests basic MLflow functionality to ensure the tracking system is working.
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"

def test_mlflow_basic_functionality():
    """Test basic MLflow functionality"""
    print("🧪 Testing MLflow Basic Functionality")
    print("=" * 50)
    
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"✓ MLflow tracking URI set: {MLFLOW_TRACKING_URI}")
    
    # Create test experiment
    experiment_name = "MLflow_Test_Experiment"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"✓ Created experiment: {experiment_name} (ID: {experiment_id})")
    except Exception as e:
        if "already exists" in str(e):
            print(f"✓ Experiment already exists: {experiment_name}")
        else:
            print(f"❌ Error creating experiment: {e}")
            return False
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Test run creation
    with mlflow.start_run(run_name=f"test_run_{datetime.now().strftime('%H%M%S')}"):
        print("✓ Started MLflow run")
        
        # Log parameters
        mlflow.log_param("test_param", "test_value")
        mlflow.log_param("timestamp", datetime.now().isoformat())
        print("✓ Logged parameters")
        
        # Log metrics
        mlflow.log_metric("test_metric", 0.95)
        mlflow.log_metric("accuracy", 0.87)
        print("✓ Logged metrics")
        
        # Create and train a simple model
        X = np.random.rand(100, 4)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log model
        mlflow.sklearn.log_model(model, "test_model")
        mlflow.log_metric("model_accuracy", accuracy)
        print("✓ Logged model and accuracy")
        
        # Create test artifact
        os.makedirs("temp_artifacts", exist_ok=True)
        test_data = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        test_data.to_csv("temp_artifacts/test_data.csv", index=False)
        mlflow.log_artifacts("temp_artifacts")
        print("✓ Logged artifacts")
        
        # Clean up
        import shutil
        if os.path.exists("temp_artifacts"):
            shutil.rmtree("temp_artifacts")
    
    print("\n🎉 MLflow test completed successfully!")
    return True

def test_mlflow_client():
    """Test MLflow client functionality"""
    print("\n🔍 Testing MLflow Client Functionality")
    print("=" * 50)
    
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        
        # List experiments
        experiments = client.search_experiments()
        print(f"✓ Found {len(experiments)} experiments")
        
        for exp in experiments:
            print(f"  - {exp.name} (ID: {exp.experiment_id})")
        
        # Get runs from test experiment
        test_exp = next((exp for exp in experiments if "Test" in exp.name), None)
        if test_exp:
            runs = client.search_runs(experiment_ids=[test_exp.experiment_id])
            print(f"✓ Found {len(runs)} runs in test experiment")
        
        return True
        
    except Exception as e:
        print(f"❌ MLflow client test failed: {e}")
        return False

def start_mlflow_ui():
    """Instructions to start MLflow UI"""
    print(f"\n🌐 To view MLflow UI, run:")
    print(f"mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI} --port 5001")
    print("Then visit: http://localhost:5001")

def main():
    """Main test function"""
    print("🐦 Bird Migration MLflow Setup Test")
    print("=" * 60)
    
    # Test basic functionality
    basic_test = test_mlflow_basic_functionality()
    
    # Test client
    client_test = test_mlflow_client()
    
    # Results
    print("\n📊 Test Results:")
    print(f"  Basic MLflow functionality: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"  MLflow client functionality: {'✅ PASS' if client_test else '❌ FAIL'}")
    
    if basic_test and client_test:
        print("\n🎉 All MLflow tests passed! The system is ready.")
        start_mlflow_ui()
    else:
        print("\n❌ Some tests failed. Please check your MLflow installation.")
    
    return basic_test and client_test

if __name__ == "__main__":
    main()