"""
Simplified Bird Migration EDA Analysis with MLflow Tracking
==========================================================

This script performs essential exploratory data analysis on bird migration datasets
and tracks the analysis process using MLflow for experiment management.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.tracking
import warnings
import os
from datetime import datetime
import json

# Configuration
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "Bird_Migration_EDA_Analysis"

class BirdMigrationEDA:
    """
    Simplified EDA class for bird migration data analysis
    """
    
    def __init__(self, data_path="data"):
        """Initialize EDA analysis"""
        self.data_path = data_path
        self.df = None
        self.eda_metrics = {}
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Setup MLflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        try:
            mlflow.create_experiment(EXPERIMENT_NAME)
        except:
            pass  # Experiment already exists
        
        mlflow.set_experiment(EXPERIMENT_NAME)
    
    def load_and_combine_data(self):
        """Load and combine all migration datasets"""
        print("📊 Loading bird migration datasets...")
        
        datasets = []
        dataset_files = ['Bird_Migration_Dataset_A.csv', 'Bird_Migration_Dataset_B.csv', 
                        'Bird_Migration_Dataset_C.csv', 'Bird_Migration_Dataset_D.csv']
        
        for file in dataset_files:
            file_path = os.path.join(self.data_path, file)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df['dataset_source'] = file.replace('.csv', '')
                    datasets.append(df)
                    print(f"   ✓ Loaded {file}: {df.shape[0]} records")
                except Exception as e:
                    print(f"   ⚠️  Error loading {file}: {e}")
        
        if datasets:
            self.df = pd.concat(datasets, ignore_index=True)
            print(f"\n🎯 Combined dataset: {self.df.shape[0]} records, {self.df.shape[1]} features")
            
            # Basic data info
            self.eda_metrics['total_records'] = len(self.df)
            self.eda_metrics['total_features'] = len(self.df.columns)
            self.eda_metrics['datasets_combined'] = len(datasets)
        else:
            raise FileNotFoundError("No dataset files found in the specified path")
    
    def basic_data_exploration(self):
        """Perform basic data exploration and quality assessment"""
        print("\n🔍 Performing basic data exploration...")
        
        # Dataset info
        print(f"Dataset shape: {self.df.shape}")
        memory_usage_mb = self.df.memory_usage(deep=True).sum() / 1024**2
        print(f"Memory usage: {memory_usage_mb:.2f} MB")
        
        # Missing values analysis
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        self.eda_metrics['missing_values_total'] = int(missing_data.sum())
        self.eda_metrics['missing_percentage_avg'] = float(missing_percentage.mean())
        
        print(f"\n📋 Data Quality Summary:")
        print(f"   • Total missing values: {missing_data.sum()}")
        print(f"   • Average missing percentage: {missing_percentage.mean():.2f}%")
        
        # Unique species count
        if 'species' in self.df.columns:
            unique_species = self.df['species'].nunique()
            self.eda_metrics['unique_species'] = unique_species
            print(f"   • Unique species: {unique_species}")
        
        # Numerical features summary
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"   • Numerical features: {len(numerical_cols)}")
        
        # Categorical features summary
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        print(f"   • Categorical features: {len(categorical_cols)}")
        
        return missing_data, missing_percentage
    
    def species_analysis(self):
        """Analyze species distribution and characteristics"""
        print("\n🦅 Analyzing species distribution...")
        
        if 'species' not in self.df.columns:
            print("❌ Species column not found")
            return
        
        # Species frequency
        species_counts = self.df['species'].value_counts()
        print(f"   • Most common species: {species_counts.index[0]} ({species_counts.iloc[0]} records)")
        print(f"   • Least common species: {species_counts.index[-1]} ({species_counts.iloc[-1]} records)")
        
        # Species diversity metrics
        self.eda_metrics['most_common_species'] = species_counts.index[0]
        self.eda_metrics['max_species_records'] = int(species_counts.iloc[0])
        self.eda_metrics['min_species_records'] = int(species_counts.iloc[-1])
        
        # Conservation status analysis
        if 'conservation_status' in self.df.columns:
            conservation_counts = self.df['conservation_status'].value_counts()
            print(f"\n🛡️  Conservation Status Distribution:")
            for status, count in conservation_counts.head().items():
                print(f"   • {status}: {count} species")
            
            self.eda_metrics['conservation_status_distribution'] = conservation_counts.to_dict()
    
    def migration_patterns_analysis(self):
        """Analyze migration patterns and characteristics"""
        print("\n🛤️  Analyzing migration patterns...")
        
        # Distance analysis
        if 'migration_distance_km' in self.df.columns:
            distances = self.df['migration_distance_km'].dropna()
            if len(distances) > 0:
                print(f"   • Average migration distance: {distances.mean():.2f} km")
                print(f"   • Maximum distance: {distances.max():.2f} km")
                print(f"   • Minimum distance: {distances.min():.2f} km")
                
                self.eda_metrics['avg_migration_distance'] = float(distances.mean())
                self.eda_metrics['max_migration_distance'] = float(distances.max())
                self.eda_metrics['min_migration_distance'] = float(distances.min())
        
        # Speed analysis
        if 'avg_speed_kmh' in self.df.columns:
            speeds = self.df['avg_speed_kmh'].dropna()
            if len(speeds) > 0:
                print(f"   • Average migration speed: {speeds.mean():.2f} km/h")
                print(f"   • Maximum speed: {speeds.max():.2f} km/h")
                print(f"   • Minimum speed: {speeds.min():.2f} km/h")
                
                self.eda_metrics['avg_migration_speed'] = float(speeds.mean())
                self.eda_metrics['max_migration_speed'] = float(speeds.max())
                self.eda_metrics['min_migration_speed'] = float(speeds.min())
        
        # Seasonal patterns
        if 'migration_season' in self.df.columns:
            seasonal_counts = self.df['migration_season'].value_counts()
            print(f"\n📅 Seasonal Migration Patterns:")
            for season, count in seasonal_counts.items():
                print(f"   • {season}: {count} migrations")
            
            self.eda_metrics['seasonal_distribution'] = seasonal_counts.to_dict()
        
        # Flyway analysis
        if 'flyway' in self.df.columns:
            flyway_counts = self.df['flyway'].value_counts()
            print(f"\n🛣️  Flyway Usage:")
            for flyway, count in flyway_counts.items():
                print(f"   • {flyway}: {count} migrations")
            
            self.eda_metrics['flyway_distribution'] = flyway_counts.to_dict()
    
    def create_basic_visualizations(self):
        """Create basic visualizations"""
        print("\n📈 Creating basic visualizations...")
        
        try:
            # Create a figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Bird Migration EDA Overview', fontsize=16)
            
            # 1. Species distribution (top species)
            if 'species' in self.df.columns:
                species_counts = self.df['species'].value_counts().head(10)
                axes[0, 0].barh(range(len(species_counts)), species_counts.values)
                axes[0, 0].set_yticks(range(len(species_counts)))
                axes[0, 0].set_yticklabels(species_counts.index, fontsize=8)
                axes[0, 0].set_title('Top 10 Species by Records')
                axes[0, 0].set_xlabel('Number of Records')
            
            # 2. Migration distance distribution
            if 'migration_distance_km' in self.df.columns:
                distances = self.df['migration_distance_km'].dropna()
                if len(distances) > 0:
                    axes[0, 1].hist(distances, bins=30, alpha=0.7, color='skyblue')
                    axes[0, 1].set_title('Migration Distance Distribution')
                    axes[0, 1].set_xlabel('Distance (km)')
                    axes[0, 1].set_ylabel('Frequency')
            
            # 3. Conservation status
            if 'conservation_status' in self.df.columns:
                status_counts = self.df['conservation_status'].value_counts()
                axes[1, 0].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
                axes[1, 0].set_title('Conservation Status Distribution')
            
            # 4. Seasonal migration patterns
            if 'migration_season' in self.df.columns:
                seasonal_counts = self.df['migration_season'].value_counts()
                axes[1, 1].bar(seasonal_counts.index, seasonal_counts.values, color='lightgreen')
                axes[1, 1].set_title('Migration by Season')
                axes[1, 1].set_ylabel('Number of Migrations')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig('results/eda_overview.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("   ✓ Basic visualizations saved")
            
        except Exception as e:
            print(f"   ⚠️  Error creating visualizations: {e}")
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        print("\n📄 Creating summary report...")
        
        summary = {
            "analysis_timestamp": datetime.now().isoformat(),
            "dataset_summary": {
                "total_records": self.eda_metrics.get('total_records', 0),
                "total_features": self.eda_metrics.get('total_features', 0),
                "unique_species": self.eda_metrics.get('unique_species', 0),
                "datasets_combined": self.eda_metrics.get('datasets_combined', 0)
            },
            "data_quality": {
                "missing_values_total": self.eda_metrics.get('missing_values_total', 0),
                "missing_percentage_avg": self.eda_metrics.get('missing_percentage_avg', 0)
            },
            "migration_insights": {
                "avg_migration_distance": self.eda_metrics.get('avg_migration_distance', 0),
                "avg_migration_speed": self.eda_metrics.get('avg_migration_speed', 0),
                "most_common_species": self.eda_metrics.get('most_common_species', 'Unknown')
            },
            "patterns": {
                "seasonal_distribution": self.eda_metrics.get('seasonal_distribution', {}),
                "flyway_distribution": self.eda_metrics.get('flyway_distribution', {}),
                "conservation_status_distribution": self.eda_metrics.get('conservation_status_distribution', {})
            }
        }
        
        # Save summary as JSON
        with open('results/eda_summary_report.json', 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        print("   ✓ Summary report saved as 'eda_summary_report.json'")
        return summary
    
    def run_complete_eda(self):
        """Run the complete EDA pipeline with MLflow tracking"""
        
        run_name = f"EDA_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            try:
                print("🚀 Starting Bird Migration EDA Analysis")
                print("=" * 60)
                
                # Log parameters
                mlflow.log_param("data_path", self.data_path)
                mlflow.log_param("analysis_date", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                
                # Step 1: Load data
                self.load_and_combine_data()
                
                # Step 2: Basic exploration
                missing_data, missing_percentage = self.basic_data_exploration()
                
                # Step 3: Species analysis
                self.species_analysis()
                
                # Step 4: Migration patterns
                self.migration_patterns_analysis()
                
                # Step 5: Create visualizations
                self.create_basic_visualizations()
                
                # Step 6: Create summary report
                summary = self.create_summary_report()
                
                # Log all metrics to MLflow
                for metric_name, metric_value in self.eda_metrics.items():
                    try:
                        if isinstance(metric_value, (int, float)) and not np.isnan(metric_value):
                            mlflow.log_metric(metric_name, metric_value)
                        else:
                            # For complex metrics, log as parameters
                            mlflow.log_param(metric_name, str(metric_value)[:500])  # Limit length
                    except Exception as e:
                        print(f"   ⚠️  Could not log metric {metric_name}: {e}")
                
                # Log artifacts
                if os.path.exists('results'):
                    try:
                        mlflow.log_artifacts('results')
                    except Exception as e:
                        print(f"   ⚠️  Could not log artifacts: {e}")
                
                print("\n✅ EDA Analysis completed successfully!")
                print(f"📊 Results saved to MLflow with tracking URI: {MLFLOW_TRACKING_URI}")
                
            except Exception as e:
                print(f"❌ Error during EDA analysis: {e}")
                mlflow.log_param("error", str(e))
                raise

def main():
    """Main execution function"""
    print("🐦 Bird Migration EDA Analysis with MLflow Tracking")
    print("=" * 60)
    
    try:
        # Initialize EDA analyzer
        eda_analyzer = BirdMigrationEDA(data_path="data")
        
        # Run complete EDA
        eda_analyzer.run_complete_eda()
        
        print("\n🎉 EDA Analysis Pipeline Completed Successfully!")
        print(f"📈 View results in MLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI} --port 5001")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()