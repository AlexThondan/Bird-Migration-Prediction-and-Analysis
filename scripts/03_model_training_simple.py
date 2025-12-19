"""
Bird Migration Model Training with MLflow Tracking
=================================================

This script trains multiple ML algorithms on bird migration data and tracks
all experiments with MLflow for comparison and analysis.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Try to import XGBoost, skip if not available
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available, will skip XGBoost model")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "Bird_Migration_ML_Training"

class SimpleMigrationTrainer:
    """
    Simple ML training pipeline for bird migration prediction
    """
    
    def __init__(self, data_path="data"):
        """Initialize the trainer"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoder = None
        self.scaler = None
        self.training_results = {}
        
        # Setup directories
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Setup MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        try:
            mlflow.create_experiment(EXPERIMENT_NAME)
        except:
            pass
        mlflow.set_experiment(EXPERIMENT_NAME)
    
    def load_data(self):
        """Load and combine migration datasets"""
        print("📊 Loading bird migration datasets...")
        
        datasets = []
        dataset_files = ['Bird_Migration_Dataset_A.csv', 'Bird_Migration_Dataset_B.csv', 
                        'Bird_Migration_Dataset_C.csv', 'Bird_Migration_Dataset_D.csv']
        
        for file in dataset_files:
            file_path = os.path.join(self.data_path, file)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    datasets.append(df)
                    print(f"   ✓ Loaded {file}: {df.shape[0]} records")
                except Exception as e:
                    print(f"   ⚠️  Error loading {file}: {e}")
        
        if datasets:
            self.df = pd.concat(datasets, ignore_index=True)
            print(f"\n🎯 Combined dataset: {self.df.shape[0]} records, {self.df.shape[1]} features")
        else:
            raise FileNotFoundError("No dataset files found")
    
    def prepare_data(self, target_column='conservation_status', sample_size=5000):
        """Prepare data for training"""
        print(f"\n🔧 Preparing data for training...")
        print(f"Target column: {target_column}")
        
        # Sample data for faster training
        if len(self.df) > sample_size:
            self.df = self.df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} records for faster training")
        
        # Select features - numeric columns only for simplicity
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and ID columns
        feature_columns = [col for col in numeric_columns if col != target_column and 'id' not in col.lower()]
        
        print(f"Selected {len(feature_columns)} numeric features")
        
        # Prepare X and y
        X = self.df[feature_columns].copy()
        y = self.df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Encode target variable if it's categorical
        if y.dtype == 'object':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y.astype(str))
            
            # Save label encoder
            joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
            print(f"Encoded target variable: {len(np.unique(y))} classes")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Save scaler
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Features: {self.X_train.shape[1]}")
        
        return feature_columns
    
    def get_models(self):
        """Get ML models to train"""
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=50, max_depth=10, random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42, max_iter=1000
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=50, random_state=42
            ),
            'SVM': SVC(
                random_state=42, probability=True, gamma='scale'
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5
            ),
            'DecisionTree': DecisionTreeClassifier(
                random_state=42, max_depth=10
            ),
            'NaiveBayes': GaussianNB(),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=50, random_state=42, eval_metric='logloss'
            )
        }
        
        return models
    
    def train_model(self, model_name, model):
        """Train a single model and log to MLflow"""
        
        run_name = f"{model_name}_{datetime.now().strftime('%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            print(f"\n🔄 Training {model_name}...")
            
            try:
                # Log model parameters
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("train_samples", len(self.X_train))
                mlflow.log_param("test_samples", len(self.X_test))
                mlflow.log_param("features", self.X_train.shape[1])
                
                # Train model
                start_time = datetime.now()
                model.fit(self.X_train, self.y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Make predictions
                y_train_pred = model.predict(self.X_train)
                y_test_pred = model.predict(self.X_test)
                
                # Calculate metrics
                train_accuracy = accuracy_score(self.y_train, y_train_pred)
                test_accuracy = accuracy_score(self.y_test, y_test_pred)
                
                # Detailed metrics
                test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
                    self.y_test, y_test_pred, average='weighted'
                )
                
                # Cross-validation score
                try:
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=3)
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                except:
                    cv_mean = cv_std = 0
                
                # Log metrics
                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("test_accuracy", test_accuracy)
                mlflow.log_metric("test_precision", test_precision)
                mlflow.log_metric("test_recall", test_recall)
                mlflow.log_metric("test_f1", test_f1)
                mlflow.log_metric("cv_mean", cv_mean)
                mlflow.log_metric("cv_std", cv_std)
                mlflow.log_metric("training_time_seconds", training_time)
                
                # ROC AUC for binary classification
                try:
                    if hasattr(model, 'predict_proba') and len(np.unique(self.y_test)) == 2:
                        y_proba = model.predict_proba(self.X_test)[:, 1]
                        roc_auc = roc_auc_score(self.y_test, y_proba)
                        mlflow.log_metric("test_roc_auc", roc_auc)
                except:
                    roc_auc = None
                
                # Save model
                model_path = f"models/{model_name}_model.pkl"
                joblib.dump(model, model_path)
                
                # Log model to MLflow
                if 'XGB' in model_name:
                    mlflow.xgboost.log_model(model, model_name)
                else:
                    mlflow.sklearn.log_model(model, model_name)
                
                # Create confusion matrix
                self.create_confusion_matrix(model_name, self.y_test, y_test_pred)
                
                # Store results
                results = {
                    'model_name': model_name,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'test_precision': test_precision,
                    'test_recall': test_recall,
                    'test_f1': test_f1,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'training_time': training_time,
                    'model': model
                }
                
                self.training_results[model_name] = results
                
                print(f"   ✅ {model_name} - Test Accuracy: {test_accuracy:.4f}")
                print(f"   📊 Cross-val: {cv_mean:.4f} (±{cv_std:.4f})")
                print(f"   ⏱️  Training time: {training_time:.2f} seconds")
                
                return results
                
            except Exception as e:
                print(f"   ❌ Error training {model_name}: {e}")
                mlflow.log_param("error", str(e))
                return None
    
    def create_confusion_matrix(self, model_name, y_true, y_pred):
        """Create and save confusion matrix"""
        try:
            plt.figure(figsize=(8, 6))
            
            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_name} - Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            # Save plot
            plot_path = f'results/{model_name}_confusion_matrix.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log as artifact
            mlflow.log_artifact(plot_path)
            
        except Exception as e:
            print(f"   ⚠️  Could not create confusion matrix for {model_name}: {e}")
    
    def create_model_comparison(self):
        """Create model comparison visualization"""
        print("\n📊 Creating model comparison...")
        
        if not self.training_results:
            print("No training results to compare")
            return
        
        # Prepare data for comparison
        models = list(self.training_results.keys())
        test_accuracies = [self.training_results[m]['test_accuracy'] for m in models]
        cv_means = [self.training_results[m]['cv_mean'] for m in models]
        training_times = [self.training_results[m]['training_time'] for m in models]
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Test accuracy comparison
        axes[0, 0].bar(models, test_accuracies, color='skyblue')
        axes[0, 0].set_title('Test Accuracy by Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(test_accuracies):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Cross-validation scores
        axes[0, 1].bar(models, cv_means, color='lightgreen')
        axes[0, 1].set_title('Cross-Validation Mean Score')
        axes[0, 1].set_ylabel('CV Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Training time comparison
        axes[1, 0].bar(models, training_times, color='orange')
        axes[1, 0].set_title('Training Time by Model')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Accuracy vs Training time scatter
        axes[1, 1].scatter(training_times, test_accuracies, s=100, alpha=0.7)
        for i, model in enumerate(models):
            axes[1, 1].annotate(model, (training_times[i], test_accuracies[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_xlabel('Training Time (seconds)')
        axes[1, 1].set_ylabel('Test Accuracy')
        axes[1, 1].set_title('Accuracy vs Training Time')
        
        plt.tight_layout()
        plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✓ Model comparison saved")
    
    def save_training_summary(self):
        """Save training summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(self.training_results),
            "best_model": max(self.training_results.keys(), 
                            key=lambda x: self.training_results[x]['test_accuracy']) if self.training_results else None,
            "model_results": {
                name: {k: v for k, v in results.items() if k != 'model'}
                for name, results in self.training_results.items()
            }
        }
        
        # Save summary
        with open('results/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        print("   ✓ Training summary saved")
        return summary
    
    def run_training_pipeline(self, target_column='conservation_status'):
        """Run complete training pipeline"""
        print("🚀 Starting Bird Migration ML Training Pipeline")
        print("=" * 60)
        
        try:
            # Load data
            self.load_data()
            
            # Prepare data
            feature_columns = self.prepare_data(target_column)
            
            # Get models
            models = self.get_models()
            
            print(f"\n🤖 Training {len(models)} ML models...")
            
            # Train each model
            for model_name, model in models.items():
                self.train_model(model_name, model)
            
            # Create comparison
            self.create_model_comparison()
            
            # Save summary
            summary = self.save_training_summary()
            
            # Find best model
            if self.training_results:
                best_model_name = max(self.training_results.keys(), 
                                    key=lambda x: self.training_results[x]['test_accuracy'])
                best_accuracy = self.training_results[best_model_name]['test_accuracy']
                
                print(f"\n🏆 Best Model: {best_model_name}")
                print(f"🎯 Best Test Accuracy: {best_accuracy:.4f}")
            
            print(f"\n✅ Training pipeline completed!")
            print(f"📊 {len(self.training_results)} models trained successfully")
            print(f"📈 View results in MLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI} --port 5001")
            
            return self.training_results
            
        except Exception as e:
            print(f"❌ Training pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main execution function"""
    print("🐦 Bird Migration ML Training with MLflow")
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = SimpleMigrationTrainer(data_path="data")
        
        # Run training pipeline
        results = trainer.run_training_pipeline(target_column='conservation_status')
        
        print("\n🎉 ML Training Pipeline Completed Successfully!")
        print("🌐 Start MLflow UI to view results:")
        print(f"   mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI} --port 5001")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()