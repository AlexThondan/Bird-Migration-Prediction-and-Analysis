"""
Bird Migration Machine Learning Training Pipeline with MLflow
============================================================

This script implements comprehensive machine learning model training for bird migration
prediction with MLflow experiment tracking, model versioning, and hyperparameter optimization.

Author: Bird Migration Research Team
Date: November 2025
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           mean_squared_error, mean_absolute_error, r2_score,
                           precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "Bird_Migration_ML_Training"

class MigrationMLTrainer:
    """
    Comprehensive ML training pipeline for bird migration prediction
    """
    
    def __init__(self, data_path="results", models_path="models"):
        """
        Initialize the ML trainer
        
        Args:
            data_path (str): Path to processed data
            models_path (str): Path to save models
        """
        self.data_path = data_path
        self.models_path = models_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_encoder = None
        self.training_metrics = {}
        self.best_models = {}
        
        # Setup MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        try:
            mlflow.create_experiment(EXPERIMENT_NAME)
        except:
            pass
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        # Ensure directories exist
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs("results", exist_ok=True)
    
    def load_processed_data(self, data_file="processed_bird_migration_data.csv"):
        """Load preprocessed data"""
        print("📊 Loading preprocessed data...")
        
        data_file_path = os.path.join(self.data_path, data_file)
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Processed data file not found: {data_file_path}")
        
        self.df = pd.read_csv(data_file_path)
        print(f"   ✓ Loaded dataset: {self.df.shape[0]} records, {self.df.shape[1]} features")
        
        self.training_metrics['dataset_shape'] = self.df.shape
        self.training_metrics['total_records'] = len(self.df)
        self.training_metrics['total_features'] = len(self.df.columns)
    
    def prepare_data_for_training(self, target_column='conservation_status', 
                                task_type='classification', test_size=0.2):
        """
        Prepare data for machine learning training
        
        Args:
            target_column (str): Target variable column
            task_type (str): 'classification' or 'regression'
            test_size (float): Test set size ratio
        """
        print(f"\n🎯 Preparing data for {task_type} task...")
        
        if target_column not in self.df.columns:
            available_targets = ['conservation_status', 'migration_distance_km', 'avg_speed_kmh', 'species']
            available_targets = [col for col in available_targets if col in self.df.columns]
            if available_targets:
                target_column = available_targets[0]
                print(f"   ⚠️ Target column not found. Using '{target_column}' instead.")
            else:
                raise ValueError(f"Target column '{target_column}' not found and no suitable alternatives.")
        
        # Prepare features and target
        exclude_cols = [target_column, 'record_id'] + [col for col in self.df.columns 
                       if self.df[col].dtype == 'object' and col != target_column]
        
        feature_columns = [col for col in self.df.columns if col not in exclude_cols]
        
        X = self.df[feature_columns].copy()
        y = self.df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Handle target variable
        if task_type == 'classification':
            if y.dtype == 'object':
                self.target_encoder = LabelEncoder()
                y = self.target_encoder.fit_transform(y.astype(str))
                # Save encoder
                joblib.dump(self.target_encoder, os.path.join(self.models_path, 'target_encoder.pkl'))
            
            # Store class information
            unique_classes = len(np.unique(y))
            self.training_metrics['num_classes'] = unique_classes
            self.training_metrics['class_distribution'] = dict(zip(*np.unique(y, return_counts=True)))
            
        elif task_type == 'regression':
            if y.dtype == 'object':
                # Try to convert to numeric
                y = pd.to_numeric(y, errors='coerce')
                y = y.dropna()
                X = X.loc[y.index]
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, 
            stratify=y if task_type == 'classification' and len(np.unique(y)) > 1 else None
        )
        
        self.training_metrics['train_size'] = len(self.X_train)
        self.training_metrics['test_size'] = len(self.X_test)
        self.training_metrics['feature_count'] = X.shape[1]
        self.training_metrics['target_column'] = target_column
        self.training_metrics['task_type'] = task_type
        
        print(f"   ✓ Training set: {len(self.X_train)} samples")
        print(f"   ✓ Test set: {len(self.X_test)} samples")
        print(f"   ✓ Features: {X.shape[1]}")
        print(f"   ✓ Target: {target_column}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_classification_models(self):
        """Get classification models with their hyperparameter grids"""
        models = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7]
                }
            },
            'XGBoost': {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            },
            'SVM': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            }
        }
        return models
    
    def get_regression_models(self):
        """Get regression models with their hyperparameter grids"""
        models = {
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'XGBoost': {
                'model': xgb.XGBRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'Ridge': {
                'model': Ridge(random_state=42),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['auto', 'svd', 'cholesky']
                }
            },
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {}
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            },
            'KNN': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            }
        }
        return models
    
    def train_model_with_cv(self, model_name, model_config, task_type='classification', 
                           search_type='grid', cv_folds=5, n_iter=20):
        """
        Train a model with cross-validation and hyperparameter optimization
        
        Args:
            model_name (str): Name of the model
            model_config (dict): Model configuration with 'model' and 'params'
            task_type (str): 'classification' or 'regression'
            search_type (str): 'grid' or 'random'
            cv_folds (int): Number of CV folds
            n_iter (int): Number of iterations for random search
        """
        
        with mlflow.start_run(run_name=f"{model_name}_{task_type}_{datetime.now().strftime('%H%M%S')}"):
            print(f"\n🔄 Training {model_name} for {task_type}...")
            
            # Log basic parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("task_type", task_type)
            mlflow.log_param("search_type", search_type)
            mlflow.log_param("cv_folds", cv_folds)
            
            model = model_config['model']
            param_grid = model_config['params']
            
            try:
                if param_grid:  # Hyperparameter optimization
                    if search_type == 'grid':
                        search = GridSearchCV(
                            model, param_grid, cv=cv_folds, 
                            scoring='accuracy' if task_type == 'classification' else 'r2',
                            n_jobs=-1, verbose=0
                        )
                    else:  # random search
                        search = RandomizedSearchCV(
                            model, param_grid, cv=cv_folds, n_iter=n_iter,
                            scoring='accuracy' if task_type == 'classification' else 'r2',
                            n_jobs=-1, verbose=0, random_state=42
                        )
                    
                    # Fit the search
                    search.fit(self.X_train, self.y_train)
                    best_model = search.best_estimator_
                    
                    # Log best parameters
                    for param, value in search.best_params_.items():
                        mlflow.log_param(f"best_{param}", value)
                    
                    mlflow.log_metric("best_cv_score", search.best_score_)
                    
                else:  # No hyperparameter optimization
                    best_model = model
                    best_model.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_train_pred = best_model.predict(self.X_train)
                y_test_pred = best_model.predict(self.X_test)
                
                # Calculate metrics
                if task_type == 'classification':
                    metrics = self.calculate_classification_metrics(
                        self.y_train, y_train_pred, self.y_test, y_test_pred, best_model
                    )
                else:
                    metrics = self.calculate_regression_metrics(
                        self.y_train, y_train_pred, self.y_test, y_test_pred
                    )
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                
                # Save and log model
                model_path = os.path.join(self.models_path, f"{model_name}_{task_type}_model.pkl")
                joblib.dump(best_model, model_path)
                
                # Log model to MLflow
                if 'XGB' in model_name:
                    mlflow.xgboost.log_model(best_model, f"{model_name}_{task_type}")
                else:
                    mlflow.sklearn.log_model(
                        best_model, 
                        f"{model_name}_{task_type}",
                        registered_model_name=f"BirdMigration_{model_name}_{task_type}"
                    )
                
                # Create and log visualizations
                self.create_model_visualizations(
                    best_model, model_name, task_type, 
                    self.y_test, y_test_pred
                )
                
                # Store best model
                self.best_models[f"{model_name}_{task_type}"] = {
                    'model': best_model,
                    'metrics': metrics,
                    'predictions': {
                        'y_train_pred': y_train_pred,
                        'y_test_pred': y_test_pred
                    }
                }
                
                print(f"   ✅ {model_name} training completed successfully")
                
                return best_model, metrics
                
            except Exception as e:
                print(f"   ❌ Error training {model_name}: {e}")
                mlflow.log_param("error", str(e))
                return None, {}
    
    def calculate_classification_metrics(self, y_train_true, y_train_pred, 
                                       y_test_true, y_test_pred, model):
        """Calculate classification metrics"""
        metrics = {}
        
        # Accuracy
        metrics['train_accuracy'] = accuracy_score(y_train_true, y_train_pred)
        metrics['test_accuracy'] = accuracy_score(y_test_true, y_test_pred)
        
        # Precision, Recall, F1
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            y_train_true, y_train_pred, average='weighted'
        )
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            y_test_true, y_test_pred, average='weighted'
        )
        
        metrics['train_precision'] = train_precision
        metrics['train_recall'] = train_recall
        metrics['train_f1'] = train_f1
        metrics['test_precision'] = test_precision
        metrics['test_recall'] = test_recall
        metrics['test_f1'] = test_f1
        
        # ROC AUC (for binary classification or with predict_proba)
        try:
            if hasattr(model, 'predict_proba') and len(np.unique(y_test_true)) == 2:
                y_test_proba = model.predict_proba(self.X_test)[:, 1]
                metrics['test_roc_auc'] = roc_auc_score(y_test_true, y_test_proba)
        except:
            pass
        
        return metrics
    
    def calculate_regression_metrics(self, y_train_true, y_train_pred, 
                                   y_test_true, y_test_pred):
        """Calculate regression metrics"""
        metrics = {}
        
        # MSE
        metrics['train_mse'] = mean_squared_error(y_train_true, y_train_pred)
        metrics['test_mse'] = mean_squared_error(y_test_true, y_test_pred)
        
        # RMSE
        metrics['train_rmse'] = np.sqrt(metrics['train_mse'])
        metrics['test_rmse'] = np.sqrt(metrics['test_mse'])
        
        # MAE
        metrics['train_mae'] = mean_absolute_error(y_train_true, y_train_pred)
        metrics['test_mae'] = mean_absolute_error(y_test_true, y_test_pred)
        
        # R²
        metrics['train_r2'] = r2_score(y_train_true, y_train_pred)
        metrics['test_r2'] = r2_score(y_test_true, y_test_pred)
        
        return metrics
    
    def create_model_visualizations(self, model, model_name, task_type, y_true, y_pred):
        """Create and save model visualizations"""
        plt.style.use('seaborn-v0_8')
        
        if task_type == 'classification':
            # Confusion Matrix
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
            axes[0].set_title(f'{model_name} - Confusion Matrix')
            axes[0].set_xlabel('Predicted')
            axes[0].set_ylabel('Actual')
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
                
                # Top 10 features
                top_indices = np.argsort(feature_importance)[-10:]
                top_importance = feature_importance[top_indices]
                top_names = [feature_names[i] for i in top_indices]
                
                axes[1].barh(top_names, top_importance)
                axes[1].set_title(f'{model_name} - Top 10 Feature Importance')
                axes[1].set_xlabel('Importance')
        
        else:  # regression
            # Prediction vs Actual plot
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            axes[0].scatter(y_true, y_pred, alpha=0.6)
            axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            axes[0].set_xlabel('Actual Values')
            axes[0].set_ylabel('Predicted Values')
            axes[0].set_title(f'{model_name} - Predictions vs Actual')
            
            # Residuals plot
            residuals = y_true - y_pred
            axes[1].scatter(y_pred, residuals, alpha=0.6)
            axes[1].axhline(y=0, color='r', linestyle='--')
            axes[1].set_xlabel('Predicted Values')
            axes[1].set_ylabel('Residuals')
            axes[1].set_title(f'{model_name} - Residuals Plot')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"results/{model_name}_{task_type}_evaluation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Log plot as artifact
        mlflow.log_artifact(plot_path)
    
    def run_training_pipeline(self, target_column='conservation_status', 
                            task_type='classification', models_to_train=None,
                            hyperparameter_optimization=True):
        """
        Run the complete training pipeline
        
        Args:
            target_column (str): Target variable column
            task_type (str): 'classification' or 'regression'
            models_to_train (list): List of model names to train
            hyperparameter_optimization (bool): Whether to perform hyperparameter optimization
        """
        
        print("🚀 Starting Bird Migration ML Training Pipeline")
        print("=" * 70)
        
        # Load data
        self.load_processed_data()
        
        # Prepare data
        self.prepare_data_for_training(target_column, task_type)
        
        # Get models
        if task_type == 'classification':
            models = self.get_classification_models()
        else:
            models = self.get_regression_models()
        
        # Filter models if specified
        if models_to_train:
            models = {k: v for k, v in models.items() if k in models_to_train}
        
        print(f"\n🤖 Training {len(models)} models for {task_type}...")
        
        # Train models
        results = {}
        for model_name, model_config in models.items():
            if not hyperparameter_optimization:
                model_config['params'] = {}  # Skip hyperparameter optimization
            
            model, metrics = self.train_model_with_cv(
                model_name, model_config, task_type
            )
            
            if model is not None:
                results[model_name] = {
                    'model': model,
                    'metrics': metrics
                }
        
        # Find best model
        best_model_name = self.find_best_model(results, task_type)
        
        # Create comparison report
        comparison_report = self.create_model_comparison_report(results, task_type)
        
        # Save final artifacts
        self.save_training_artifacts(results, comparison_report)
        
        print(f"\n🏆 Best model: {best_model_name}")
        print("✅ Training pipeline completed successfully!")
        
        return results
    
    def find_best_model(self, results, task_type):
        """Find the best performing model"""
        if not results:
            return None
        
        best_model_name = None
        best_score = -np.inf if task_type == 'classification' else np.inf
        
        metric_key = 'test_accuracy' if task_type == 'classification' else 'test_r2'
        
        for model_name, result in results.items():
            if metric_key in result['metrics']:
                score = result['metrics'][metric_key]
                
                if task_type == 'classification' and score > best_score:
                    best_score = score
                    best_model_name = model_name
                elif task_type == 'regression' and score > best_score:  # Higher R² is better
                    best_score = score
                    best_model_name = model_name
        
        return best_model_name
    
    def create_model_comparison_report(self, results, task_type):
        """Create a comprehensive model comparison report"""
        comparison_data = []
        
        for model_name, result in results.items():
            metrics = result['metrics']
            
            if task_type == 'classification':
                comparison_data.append({
                    'Model': model_name,
                    'Test_Accuracy': metrics.get('test_accuracy', 0),
                    'Test_Precision': metrics.get('test_precision', 0),
                    'Test_Recall': metrics.get('test_recall', 0),
                    'Test_F1': metrics.get('test_f1', 0)
                })
            else:
                comparison_data.append({
                    'Model': model_name,
                    'Test_R2': metrics.get('test_r2', 0),
                    'Test_RMSE': metrics.get('test_rmse', 0),
                    'Test_MAE': metrics.get('test_mae', 0)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison report
        comparison_df.to_csv(f'results/model_comparison_{task_type}.csv', index=False)
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        if task_type == 'classification':
            metrics_to_plot = ['Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1']
        else:
            metrics_to_plot = ['Test_R2', 'Test_RMSE', 'Test_MAE']
        
        for i, metric in enumerate(metrics_to_plot):
            plt.subplot(1, len(metrics_to_plot), i+1)
            plt.bar(comparison_df['Model'], comparison_df[metric])
            plt.title(metric)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'results/model_comparison_{task_type}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df
    
    def save_training_artifacts(self, results, comparison_report):
        """Save training artifacts and reports"""
        print("\n💾 Saving training artifacts...")
        
        # Save training summary
        training_summary = {
            "timestamp": datetime.now().isoformat(),
            "training_metrics": self.training_metrics,
            "models_trained": list(results.keys()),
            "best_models": {name: info['metrics'] for name, info in results.items()},
            "artifacts_location": {
                "models": self.models_path,
                "results": "results",
                "comparison_report": f"results/model_comparison_{self.training_metrics.get('task_type', 'unknown')}.csv"
            }
        }
        
        with open('results/training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=4, default=str)
        
        print("   ✓ Training summary saved")
        print(f"   ✓ Model comparison report saved")
        print(f"   ✓ {len(results)} models saved in '{self.models_path}'")


def main():
    """Main execution function"""
    print("🤖 Bird Migration ML Training Pipeline with MLflow")
    print("=" * 70)
    
    try:
        # Initialize trainer
        trainer = MigrationMLTrainer(data_path="results", models_path="models")
        
        # Run training pipeline for classification
        print("\n🎯 Running Classification Pipeline...")
        classification_results = trainer.run_training_pipeline(
            target_column='conservation_status',
            task_type='classification',
            models_to_train=['RandomForest', 'XGBoost', 'LogisticRegression', 'GradientBoosting'],
            hyperparameter_optimization=True
        )
        
        # Run training pipeline for regression (migration distance prediction)
        print("\n📏 Running Regression Pipeline...")
        trainer_reg = MigrationMLTrainer(data_path="results", models_path="models")
        regression_results = trainer_reg.run_training_pipeline(
            target_column='migration_distance_km',
            task_type='regression',
            models_to_train=['RandomForest', 'XGBoost', 'Ridge'],
            hyperparameter_optimization=True
        )
        
        print("\n🎉 Complete ML Training Pipeline Finished Successfully!")
        print(f"📊 Classification models: {len(classification_results)}")
        print(f"📊 Regression models: {len(regression_results)}")
        print(f"📈 View results in MLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
        
    except Exception as e:
        print(f"❌ Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()