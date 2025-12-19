# Bird Migration ML Pipeline with MLflow Tracking

## Overview

This project implements a comprehensive machine learning pipeline for bird migration analysis with MLflow experiment tracking. The pipeline includes exploratory data analysis (EDA), data preprocessing, model training, and experiment management.

## 🚀 Quick Start

### Run Complete Pipeline
```bash
python run_pipeline.py
```

### Run Individual Steps
```bash
# Exploratory Data Analysis
python run_pipeline.py --step eda

# Data Preprocessing
python run_pipeline.py --step preprocess

# Model Training
python run_pipeline.py --step train

# MLflow Analysis
python run_pipeline.py --step track
```

### Start MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5001
```

## 📁 Project Structure

```
bird_migration_project/
├── scripts/
│   ├── 01_eda_analysis.py          # Exploratory Data Analysis with MLflow
│   ├── 02_data_preprocessing.py    # Data preprocessing pipeline
│   ├── 03_model_training.py        # ML model training with hyperparameter optimization
│   └── 04_mlflow_tracker.py        # MLflow experiment analysis and visualization
├── data/                           # Raw datasets
│   ├── Bird_Migration_Dataset_A.csv
│   ├── Bird_Migration_Dataset_B.csv
│   ├── Bird_Migration_Dataset_C.csv
│   └── Bird_Migration_Dataset_D.csv
├── results/                        # Analysis results and visualizations
├── models/                         # Trained models and preprocessors
├── mlruns.db                       # MLflow tracking database
├── run_pipeline.py                 # Main pipeline runner
└── README_ML_Pipeline.md           # This documentation
```

## 🔬 Pipeline Components

### 1. Exploratory Data Analysis (01_eda_analysis.py)
- **Purpose**: Comprehensive data exploration and quality assessment
- **Features**:
  - Data loading and combination from multiple sources
  - Missing value analysis
  - Species distribution analysis
  - Migration pattern insights
  - Statistical summaries
  - Comprehensive visualizations
- **MLflow Tracking**: Logs data quality metrics, insights, and visualization artifacts

### 2. Data Preprocessing (02_data_preprocessing.py)
- **Purpose**: Clean and prepare data for machine learning
- **Features**:
  - Advanced missing value imputation (KNN for numerics, mode for categoricals)
  - Categorical encoding (label encoding, one-hot encoding)
  - Feature engineering (distance categories, speed ratios, efficiency metrics)
  - Feature scaling (StandardScaler, RobustScaler)
  - Feature selection (mutual information, f-classif)
  - Train-test splitting with stratification
- **MLflow Tracking**: Logs preprocessing parameters, data transformations, and model artifacts

### 3. Model Training (03_model_training.py)
- **Purpose**: Train and evaluate multiple ML models with hyperparameter optimization
- **Supported Models**:
  - **Classification**: RandomForest, XGBoost, LogisticRegression, GradientBoosting, SVM, KNN
  - **Regression**: RandomForest, XGBoost, Ridge, LinearRegression, SVR, KNN
- **Features**:
  - Cross-validation with hyperparameter optimization
  - Grid search and random search
  - Comprehensive model evaluation metrics
  - Model comparison and visualization
  - Automatic model registration
- **MLflow Tracking**: Logs all hyperparameters, metrics, models, and evaluation plots

### 4. MLflow Experiment Tracker (04_mlflow_tracker.py)
- **Purpose**: Analyze and visualize experiment results
- **Features**:
  - Experiment summary and statistics
  - Model performance comparison
  - Experiment timeline visualization
  - Registered model management
  - Data export capabilities
  - Comprehensive reporting

## 📊 MLflow Experiments

The pipeline creates three main experiments:

1. **Bird_Migration_EDA_Analysis**: Tracks data exploration metrics and insights
2. **Bird_Migration_Data_Preprocessing**: Tracks data transformation steps and quality metrics
3. **Bird_Migration_ML_Training**: Tracks model training experiments with hyperparameters and performance metrics

## 🎯 Available Tasks

### Classification Tasks
- **Conservation Status Prediction**: Predict species conservation status (Least Concern, Near Threatened, Endangered, etc.)
- **Species Classification**: Classify bird species based on migration characteristics

### Regression Tasks
- **Migration Distance Prediction**: Predict migration distance based on species and environmental factors
- **Migration Speed Prediction**: Predict average migration speed

## 📈 Model Performance Tracking

All model experiments are automatically tracked with:

### Classification Metrics
- Accuracy
- Precision, Recall, F1-score
- ROC AUC (for binary classification)
- Confusion matrices

### Regression Metrics
- R² Score
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Residual analysis plots

## 🔧 Configuration

### MLflow Configuration
- **Tracking URI**: `sqlite:///mlruns.db`
- **Default Port**: 5001
- **Artifact Location**: Local filesystem

### Model Hyperparameters
Each model includes optimized hyperparameter grids for:
- Tree-based models: n_estimators, max_depth, min_samples_split
- Ensemble methods: learning_rate, subsample ratios
- Linear models: regularization parameters
- Distance-based models: neighbor counts, distance metrics

## 📋 Requirements

```bash
# Core ML libraries
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
xgboost>=1.5.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# MLflow tracking
mlflow>=1.20.0

# Utilities
joblib>=1.1.0
```

## 🚀 Usage Examples

### Basic Usage
```python
# Run complete pipeline
from run_pipeline import run_complete_pipeline
success = run_complete_pipeline()
```

### Individual Components
```python
# Run EDA
from scripts.eda_analysis import BirdMigrationEDA
eda = BirdMigrationEDA("data")
eda.run_complete_eda()

# Run preprocessing
from scripts.data_preprocessing import MigrationDataPreprocessor
preprocessor = MigrationDataPreprocessor("data")
results = preprocessor.run_preprocessing_pipeline()

# Run training
from scripts.model_training import MigrationMLTrainer
trainer = MigrationMLTrainer()
models = trainer.run_training_pipeline()
```

## 📊 Viewing Results

1. **MLflow Dashboard**: Access the web UI at `http://localhost:5001`
2. **Results Directory**: Check `results/` for CSV files and plots
3. **Models Directory**: Check `models/` for trained model artifacts

## 🐛 Troubleshooting

### Common Issues

1. **Missing Data Files**: Ensure all CSV files are in the `data/` directory
2. **MLflow Database Lock**: Stop any running MLflow instances before starting new ones
3. **Memory Issues**: For large datasets, consider reducing the number of hyperparameter combinations

### Error Handling
- All scripts include comprehensive error handling and logging
- Failed experiments are logged to MLflow for debugging
- Pipeline continues execution even if individual steps fail

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure MLflow tracking is properly implemented
5. Submit a pull request

## 📝 License

This project is created for educational and research purposes in bird migration analysis.

---

**🐦 Happy Migration Modeling!** 🚀