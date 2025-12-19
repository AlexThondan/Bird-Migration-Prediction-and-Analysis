
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
import joblib
import os
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "Bird_Migration_Data_Preprocessing"

class MigrationDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for bird migration data
    """
    
    def __init__(self, data_path="data"):
        """
        Initialize the preprocessor
        
        Args:
            data_path (str): Path to data directory
        """
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.preprocessing_metrics = {}
        self.encoders = {}
        self.scalers = {}
        self.feature_selectors = {}
        
        # Setup MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        try:
            mlflow.create_experiment(EXPERIMENT_NAME)
        except:
            pass
        mlflow.set_experiment(EXPERIMENT_NAME)
    
    def load_data(self):
        """Load and combine all migration datasets"""
        print("📊 Loading bird migration datasets...")
        
        datasets = []
        dataset_files = ['Bird_Migration_Dataset_A.csv', 'Bird_Migration_Dataset_B.csv', 
                        'Bird_Migration_Dataset_C.csv', 'Bird_Migration_Dataset_D.csv']
        
        for file in dataset_files:
            file_path = os.path.join(self.data_path, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['dataset_source'] = file.replace('.csv', '')
                datasets.append(df)
                print(f"   ✓ Loaded {file}: {df.shape[0]} records")
        
        if datasets:
            self.df = pd.concat(datasets, ignore_index=True)
            print(f"\n🎯 Combined dataset: {self.df.shape[0]} records, {self.df.shape[1]} features")
            
            # Store initial metrics
            self.preprocessing_metrics['original_records'] = len(self.df)
            self.preprocessing_metrics['original_features'] = len(self.df.columns)
        else:
            raise FileNotFoundError("No dataset files found in the specified path")
    
    def initial_data_assessment(self):
        """Perform initial data quality assessment"""
        print("\n🔍 Performing initial data assessment...")
        
        # Missing values analysis
        missing_counts = self.df.isnull().sum()
        missing_percentage = (missing_counts / len(self.df)) * 100
        
        # Data types analysis
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        # Store metrics
        self.preprocessing_metrics['missing_values_total'] = int(missing_counts.sum())
        self.preprocessing_metrics['missing_percentage_mean'] = float(missing_percentage.mean())
        self.preprocessing_metrics['numeric_features_count'] = len(numeric_cols)
        self.preprocessing_metrics['categorical_features_count'] = len(categorical_cols)
        
        print(f"   • Total missing values: {missing_counts.sum()}")
        print(f"   • Average missing percentage: {missing_percentage.mean():.2f}%")
        print(f"   • Numeric features: {len(numeric_cols)}")
        print(f"   • Categorical features: {len(categorical_cols)}")
        
        # Identify high missing value columns
        high_missing_cols = missing_percentage[missing_percentage > 50].index.tolist()
        if high_missing_cols:
            print(f"   ⚠️  High missing value columns (>50%): {high_missing_cols}")
            self.preprocessing_metrics['high_missing_columns'] = high_missing_cols
        
        return numeric_cols, categorical_cols, missing_counts
    
    def handle_missing_values(self, numeric_cols, categorical_cols, strategy='advanced'):
        """Handle missing values using various strategies"""
        print(f"\n🔧 Handling missing values using '{strategy}' strategy...")
        
        self.processed_df = self.df.copy()
        
        if strategy == 'basic':
            # Simple imputation
            # Numeric: median, Categorical: mode
            for col in numeric_cols:
                if self.processed_df[col].isnull().any():
                    median_val = self.processed_df[col].median()
                    self.processed_df[col].fillna(median_val, inplace=True)
            
            for col in categorical_cols:
                if self.processed_df[col].isnull().any():
                    mode_val = self.processed_df[col].mode().iloc[0] if not self.processed_df[col].mode().empty else 'Unknown'
                    self.processed_df[col].fillna(mode_val, inplace=True)
                    
        elif strategy == 'advanced':
            # Advanced imputation with KNN for numerics and mode for categoricals
            if numeric_cols:
                # KNN imputation for numeric features
                numeric_imputer = KNNImputer(n_neighbors=5, weights='uniform')
                self.processed_df[numeric_cols] = numeric_imputer.fit_transform(self.processed_df[numeric_cols])
                self.encoders['numeric_imputer'] = numeric_imputer
            
            # Mode imputation for categorical features
            for col in categorical_cols:
                if self.processed_df[col].isnull().any():
                    mode_val = self.processed_df[col].mode().iloc[0] if not self.processed_df[col].mode().empty else 'Unknown'
                    self.processed_df[col].fillna(mode_val, inplace=True)
        
        # Calculate missing values after imputation
        remaining_missing = self.processed_df.isnull().sum().sum()
        self.preprocessing_metrics['remaining_missing_after_imputation'] = int(remaining_missing)
        
        print(f"   ✓ Missing values handled. Remaining: {remaining_missing}")
    
    def encode_categorical_features(self, categorical_cols):
        """Encode categorical features"""
        print("\n🔤 Encoding categorical features...")
        
        encoded_features = []
        
        for col in categorical_cols:
            if col in self.processed_df.columns:
                unique_values = self.processed_df[col].nunique()
                
                if unique_values <= 2:
                    # Binary encoding for binary categories
                    le = LabelEncoder()
                    self.processed_df[f'{col}_encoded'] = le.fit_transform(self.processed_df[col].astype(str))
                    self.encoders[f'{col}_label_encoder'] = le
                    encoded_features.append(f'{col}_encoded')
                    
                elif unique_values <= 10:
                    # One-hot encoding for low cardinality
                    dummies = pd.get_dummies(self.processed_df[col], prefix=col)
                    self.processed_df = pd.concat([self.processed_df, dummies], axis=1)
                    encoded_features.extend(dummies.columns.tolist())
                    
                else:
                    # Label encoding for high cardinality
                    le = LabelEncoder()
                    self.processed_df[f'{col}_encoded'] = le.fit_transform(self.processed_df[col].astype(str))
                    self.encoders[f'{col}_label_encoder'] = le
                    encoded_features.append(f'{col}_encoded')
                
                print(f"   ✓ Encoded {col} (unique values: {unique_values})")
        
        self.preprocessing_metrics['encoded_features_count'] = len(encoded_features)
        self.preprocessing_metrics['categorical_encoding_completed'] = True
        
        print(f"   ✓ Total encoded features created: {len(encoded_features)}")
        
        return encoded_features
    
    def feature_engineering(self):
        """Create new features through feature engineering"""
        print("\n⚙️ Performing feature engineering...")
        
        engineered_features = []
        
        # Distance-based features
        if 'migration_distance_km' in self.processed_df.columns:
            # Distance categories
            self.processed_df['distance_category'] = pd.cut(
                self.processed_df['migration_distance_km'],
                bins=[0, 1000, 5000, 10000, float('inf')],
                labels=['Short', 'Medium', 'Long', 'Ultra-long']
            )
            engineered_features.append('distance_category')
            
            # Log distance for skewed data
            self.processed_df['log_distance'] = np.log1p(self.processed_df['migration_distance_km'])
            engineered_features.append('log_distance')
        
        # Speed-based features
        if 'avg_speed_kmh' in self.processed_df.columns:
            # Speed categories
            self.processed_df['speed_category'] = pd.cut(
                self.processed_df['avg_speed_kmh'],
                bins=[0, 30, 60, 90, float('inf')],
                labels=['Slow', 'Moderate', 'Fast', 'Very_fast']
            )
            engineered_features.append('speed_category')
        
        # Time-based features
        if 'year' in self.processed_df.columns:
            # Decade feature
            self.processed_df['decade'] = (self.processed_df['year'] // 10) * 10
            engineered_features.append('decade')
        
        # Ratio features
        if 'avg_wingspan_cm' in self.processed_df.columns and 'avg_mass_g' in self.processed_df.columns:
            self.processed_df['wingspan_mass_ratio'] = (
                self.processed_df['avg_wingspan_cm'] / self.processed_df['avg_mass_g']
            )
            engineered_features.append('wingspan_mass_ratio')
        
        # Migration efficiency
        if 'migration_distance_km' in self.processed_df.columns and 'estimated_duration_days' in self.processed_df.columns:
            self.processed_df['migration_efficiency'] = (
                self.processed_df['migration_distance_km'] / self.processed_df['estimated_duration_days']
            ).replace([np.inf, -np.inf], 0)
            engineered_features.append('migration_efficiency')
        
        # Encode new categorical features
        for feature in ['distance_category', 'speed_category']:
            if feature in self.processed_df.columns:
                le = LabelEncoder()
                self.processed_df[f'{feature}_encoded'] = le.fit_transform(self.processed_df[feature].astype(str))
                self.encoders[f'{feature}_label_encoder'] = le
                engineered_features.append(f'{feature}_encoded')
        
        self.preprocessing_metrics['engineered_features_count'] = len(engineered_features)
        print(f"   ✓ Created {len(engineered_features)} engineered features")
        
        return engineered_features
    
    def scale_features(self, numeric_features, scaling_method='robust'):
        """Scale numeric features"""
        print(f"\n📏 Scaling numeric features using '{scaling_method}' method...")
        
        if scaling_method == 'standard':
            scaler = StandardScaler()
        elif scaling_method == 'robust':
            scaler = RobustScaler()
        else:
            print("   ⚠️ Unknown scaling method. Using RobustScaler.")
            scaler = RobustScaler()
        
        # Select only existing numeric features
        available_numeric_features = [col for col in numeric_features if col in self.processed_df.columns]
        
        if available_numeric_features:
            self.processed_df[available_numeric_features] = scaler.fit_transform(
                self.processed_df[available_numeric_features]
            )
            self.scalers['feature_scaler'] = scaler
            
            self.preprocessing_metrics['scaled_features_count'] = len(available_numeric_features)
            self.preprocessing_metrics['scaling_method'] = scaling_method
            
            print(f"   ✓ Scaled {len(available_numeric_features)} numeric features")
        else:
            print("   ⚠️ No numeric features found for scaling")
    
    def feature_selection(self, target_column=None, method='mutual_info', k=50):
        """Perform feature selection"""
        print(f"\n🎯 Performing feature selection using '{method}' method...")
        
        if target_column is None or target_column not in self.processed_df.columns:
            print("   ⚠️ Target column not specified or not found. Skipping feature selection.")
            return []
        
        # Prepare features (exclude target and non-predictive columns)
        exclude_cols = [target_column, 'record_id', 'dataset_source'] + [col for col in self.processed_df.columns if '_encoded' not in col and self.processed_df[col].dtype == 'object']
        feature_columns = [col for col in self.processed_df.columns if col not in exclude_cols]
        
        X = self.processed_df[feature_columns]
        y = self.processed_df[target_column]
        
        # Handle missing values in features
        X = X.fillna(X.median())
        
        # Encode target if categorical
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
            self.encoders['target_encoder'] = le_target
        
        # Select features
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, len(feature_columns)))
        else:  # f_classif
            selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_columns)))
        
        try:
            X_selected = selector.fit_transform(X, y)
            selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
            
            self.feature_selectors['feature_selector'] = selector
            self.preprocessing_metrics['selected_features_count'] = len(selected_features)
            self.preprocessing_metrics['feature_selection_method'] = method
            
            print(f"   ✓ Selected {len(selected_features)} features out of {len(feature_columns)}")
            return selected_features
            
        except Exception as e:
            print(f"   ⚠️ Feature selection failed: {e}")
            return feature_columns
    
    def create_train_test_split(self, target_column=None, test_size=0.2):
        """Create train-test split"""
        print(f"\n📊 Creating train-test split (test size: {test_size})...")
        
        if target_column is None or target_column not in self.processed_df.columns:
            print("   ⚠️ Target column not specified. Creating random split.")
            train_df, test_df = train_test_split(self.processed_df, test_size=test_size, random_state=42)
        else:
            # Stratified split based on target
            y = self.processed_df[target_column]
            if y.dtype == 'object' and y.nunique() <= 100:  # Categorical with reasonable unique values
                train_df, test_df = train_test_split(
                    self.processed_df, test_size=test_size, stratify=y, random_state=42
                )
            else:
                train_df, test_df = train_test_split(
                    self.processed_df, test_size=test_size, random_state=42
                )
        
        self.preprocessing_metrics['train_size'] = len(train_df)
        self.preprocessing_metrics['test_size'] = len(test_df)
        self.preprocessing_metrics['test_split_ratio'] = test_size
        
        print(f"   ✓ Training set: {len(train_df)} records")
        print(f"   ✓ Test set: {len(test_df)} records")
        
        return train_df, test_df
    
    def save_preprocessing_artifacts(self):
        """Save preprocessing artifacts"""
        print("\n💾 Saving preprocessing artifacts...")
        
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Save processed data
        self.processed_df.to_csv('results/processed_bird_migration_data.csv', index=False)
        print("   ✓ Processed data saved")
        
        # Save encoders and scalers
        if self.encoders:
            joblib.dump(self.encoders, 'models/encoders.pkl')
            print("   ✓ Encoders saved")
        
        if self.scalers:
            joblib.dump(self.scalers, 'models/scalers.pkl')
            print("   ✓ Scalers saved")
        
        if self.feature_selectors:
            joblib.dump(self.feature_selectors, 'models/feature_selectors.pkl')
            print("   ✓ Feature selectors saved")
        
        # Save preprocessing report
        preprocessing_report = {
            "timestamp": datetime.now().isoformat(),
            "preprocessing_metrics": self.preprocessing_metrics,
            "data_shape": {
                "original": [self.preprocessing_metrics['original_records'], self.preprocessing_metrics['original_features']],
                "processed": list(self.processed_df.shape) if self.processed_df is not None else [0, 0]
            },
            "artifacts_saved": {
                "processed_data": "results/processed_bird_migration_data.csv",
                "encoders": "models/encoders.pkl" if self.encoders else None,
                "scalers": "models/scalers.pkl" if self.scalers else None,
                "feature_selectors": "models/feature_selectors.pkl" if self.feature_selectors else None
            }
        }
        
        with open('results/preprocessing_report.json', 'w') as f:
            json.dump(preprocessing_report, f, indent=4)
        
        print("   ✓ Preprocessing report saved")
        
        return preprocessing_report
    
    def run_preprocessing_pipeline(self, target_column='conservation_status', 
                                 missing_strategy='advanced', scaling_method='robust',
                                 feature_selection_method='mutual_info', k_features=50):
        """Run the complete preprocessing pipeline"""
        
        run_name = f"Preprocessing_Pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            try:
                print("🚀 Starting Bird Migration Data Preprocessing Pipeline")
                print("=" * 70)
                
                # Log parameters
                mlflow.log_param("target_column", target_column)
                mlflow.log_param("missing_strategy", missing_strategy)
                mlflow.log_param("scaling_method", scaling_method)
                mlflow.log_param("feature_selection_method", feature_selection_method)
                mlflow.log_param("k_features", k_features)
                mlflow.log_param("pipeline_timestamp", datetime.now().isoformat())
                
                # Step 1: Load data
                self.load_data()
                
                # Step 2: Initial assessment
                numeric_cols, categorical_cols, missing_counts = self.initial_data_assessment()
                
                # Step 3: Handle missing values
                self.handle_missing_values(numeric_cols, categorical_cols, strategy=missing_strategy)
                
                # Step 4: Encode categorical features
                encoded_features = self.encode_categorical_features(categorical_cols)
                
                # Step 5: Feature engineering
                engineered_features = self.feature_engineering()
                
                # Update numeric columns list
                numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns.tolist()
                
                # Step 6: Scale features
                self.scale_features(numeric_cols, scaling_method=scaling_method)
                
                # Step 7: Feature selection
                selected_features = self.feature_selection(target_column, method=feature_selection_method, k=k_features)
                
                # Step 8: Create train-test split
                train_df, test_df = self.create_train_test_split(target_column)
                
                # Step 9: Save artifacts
                preprocessing_report = self.save_preprocessing_artifacts()
                
                # Log metrics to MLflow
                for metric_name, metric_value in self.preprocessing_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                    else:
                        mlflow.log_param(f"info_{metric_name}", str(metric_value)[:500])
                
                # Log artifacts
                if os.path.exists('results'):
                    mlflow.log_artifacts('results')
                if os.path.exists('models'):
                    mlflow.log_artifacts('models')
                
                # Log models
                if self.scalers.get('feature_scaler'):
                    mlflow.sklearn.log_model(
                        self.scalers['feature_scaler'], 
                        "feature_scaler",
                        registered_model_name="BirdMigration_FeatureScaler"
                    )
                
                print("\n✅ Preprocessing pipeline completed successfully!")
                print(f"📊 Final dataset shape: {self.processed_df.shape}")
                print(f"📈 View results in MLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
                
                return {
                    'processed_data': self.processed_df,
                    'train_data': train_df,
                    'test_data': test_df,
                    'encoders': self.encoders,
                    'scalers': self.scalers,
                    'selected_features': selected_features,
                    'preprocessing_report': preprocessing_report
                }
                
            except Exception as e:
                print(f"❌ Preprocessing pipeline failed: {e}")
                mlflow.log_param("error", str(e))
                raise


def main():
    """Main execution function"""
    print("🔄 Bird Migration Data Preprocessing Pipeline with MLflow")
    print("=" * 70)
    
    try:
        # Initialize preprocessor
        preprocessor = MigrationDataPreprocessor(data_path="data")
        
        # Run preprocessing pipeline
        results = preprocessor.run_preprocessing_pipeline(
            target_column='conservation_status',  # Change as needed
            missing_strategy='advanced',
            scaling_method='robust',
            feature_selection_method='mutual_info',
            k_features=50
        )
        
        print("\n🎉 Preprocessing Pipeline Completed Successfully!")
        print("📁 Processed files saved in 'results/' and 'models/' directories")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()