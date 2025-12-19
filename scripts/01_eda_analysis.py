import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
import mlflow.tracking
import warnings
import os
from datetime import datetime
import json

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('default')  # Use default style instead of seaborn-v0_8
sns.set_palette("husl")
# Set matplotlib backend for headless environments
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"
EXPERIMENT_NAME = "Bird_Migration_EDA_Analysis"

class BirdMigrationEDA:
    """
    Comprehensive EDA class for bird migration data analysis
    """
    
    def __init__(self, data_path="data"):
        """
        Initialize EDA analysis
        
        Args:
            data_path (str): Path to data directory
        """
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
                df = pd.read_csv(file_path)
                df['dataset_source'] = file.replace('.csv', '')
                datasets.append(df)
                print(f"   ✓ Loaded {file}: {df.shape[0]} records")
        
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
        print(f"Memory usage: {self.df.memory_usage().sum() / 1024**2:.2f} MB")
        
        # Missing values analysis
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        self.eda_metrics['missing_values_total'] = missing_data.sum()
        self.eda_metrics['missing_percentage_avg'] = missing_percentage.mean()
        
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
        
        # Create species distribution plot
        plt.figure(figsize=(15, 8))
        top_species = species_counts.head(20)
        plt.subplot(2, 2, 1)
        top_species.plot(kind='barh')
        plt.title('Top 20 Species by Record Count')
        plt.xlabel('Number of Records')
        
        # Conservation status pie chart
        if 'conservation_status' in self.df.columns:
            plt.subplot(2, 2, 2)
            conservation_counts.plot(kind='pie', autopct='%1.1f%%')
            plt.title('Conservation Status Distribution')
            plt.ylabel('')
        
        plt.tight_layout()
        plt.savefig('results/species_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show
    
    def migration_patterns_analysis(self):
        """Analyze migration patterns and characteristics"""
        print("\n🛤️  Analyzing migration patterns...")
        
        # Distance analysis
        if 'migration_distance_km' in self.df.columns:
            distances = self.df['migration_distance_km'].dropna()
            print(f"   • Average migration distance: {distances.mean():.2f} km")
            print(f"   • Maximum distance: {distances.max():.2f} km")
            print(f"   • Minimum distance: {distances.min():.2f} km")
            
            self.eda_metrics['avg_migration_distance'] = distances.mean()
            self.eda_metrics['max_migration_distance'] = distances.max()
            self.eda_metrics['min_migration_distance'] = distances.min()
        
        # Speed analysis
        if 'avg_speed_kmh' in self.df.columns:
            speeds = self.df['avg_speed_kmh'].dropna()
            print(f"   • Average migration speed: {speeds.mean():.2f} km/h")
            print(f"   • Maximum speed: {speeds.max():.2f} km/h")
            print(f"   • Minimum speed: {speeds.min():.2f} km/h")
            
            self.eda_metrics['avg_migration_speed'] = speeds.mean()
            self.eda_metrics['max_migration_speed'] = speeds.max()
            self.eda_metrics['min_migration_speed'] = speeds.min()
        
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
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📈 Creating comprehensive visualizations...")
        
        # Set up the plotting area
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Distance vs Speed scatter plot
        if 'migration_distance_km' in self.df.columns and 'avg_speed_kmh' in self.df.columns:
            plt.subplot(3, 3, 1)
            scatter_data = self.df[['migration_distance_km', 'avg_speed_kmh']].dropna()
            plt.scatter(scatter_data['migration_distance_km'], scatter_data['avg_speed_kmh'], alpha=0.6)
            plt.xlabel('Migration Distance (km)')
            plt.ylabel('Average Speed (km/h)')
            plt.title('Distance vs Speed Relationship')
        
        # 2. Seasonal migration distribution
        if 'migration_season' in self.df.columns:
            plt.subplot(3, 3, 2)
            seasonal_counts = self.df['migration_season'].value_counts()
            plt.pie(seasonal_counts.values, labels=seasonal_counts.index, autopct='%1.1f%%')
            plt.title('Seasonal Migration Distribution')
        
        # 3. Flyway usage
        if 'flyway' in self.df.columns:
            plt.subplot(3, 3, 3)
            flyway_counts = self.df['flyway'].value_counts()
            plt.barh(flyway_counts.index, flyway_counts.values)
            plt.title('Migration Flyway Usage')
            plt.xlabel('Number of Migrations')
        
        # 4. Distance distribution histogram
        if 'migration_distance_km' in self.df.columns:
            plt.subplot(3, 3, 4)
            self.df['migration_distance_km'].hist(bins=50, alpha=0.7)
            plt.xlabel('Migration Distance (km)')
            plt.ylabel('Frequency')
            plt.title('Migration Distance Distribution')
        
        # 5. Speed distribution histogram
        if 'avg_speed_kmh' in self.df.columns:
            plt.subplot(3, 3, 5)
            self.df['avg_speed_kmh'].hist(bins=50, alpha=0.7, color='orange')
            plt.xlabel('Average Speed (km/h)')
            plt.ylabel('Frequency')
            plt.title('Migration Speed Distribution')
        
        # 6. Species conservation status
        if 'conservation_status' in self.df.columns:
            plt.subplot(3, 3, 6)
            conservation_counts = self.df['conservation_status'].value_counts()
            plt.bar(conservation_counts.index, conservation_counts.values)
            plt.xticks(rotation=45)
            plt.title('Conservation Status Distribution')
            plt.ylabel('Number of Species')
        
        # 7. Stopover analysis
        if 'stopover_count' in self.df.columns:
            plt.subplot(3, 3, 7)
            stopover_data = self.df['stopover_count'].dropna()
            plt.hist(stopover_data, bins=20, alpha=0.7, color='green')
            plt.xlabel('Number of Stopovers')
            plt.ylabel('Frequency')
            plt.title('Stopover Count Distribution')
        
        # 8. Migration duration
        if 'estimated_duration_days' in self.df.columns:
            plt.subplot(3, 3, 8)
            duration_data = self.df['estimated_duration_days'].dropna()
            plt.hist(duration_data, bins=30, alpha=0.7, color='purple')
            plt.xlabel('Duration (days)')
            plt.ylabel('Frequency')
            plt.title('Migration Duration Distribution')
        
        # 9. Year-wise migration trends
        if 'year' in self.df.columns:
            plt.subplot(3, 3, 9)
            yearly_counts = self.df['year'].value_counts()
            plt.plot(yearly_counts.index, yearly_counts.values, marker='o')
            plt.xlabel('Year')
            plt.ylabel('Number of Migrations')
            plt.title('Year-wise Migration Trends')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/comprehensive_eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show
    
    def correlation_analysis(self):
        """Perform correlation analysis on numerical features"""
        print("\n🔗 Performing correlation analysis...")
        
        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) > 2:
            # Calculate correlation matrix
            correlation_matrix = self.df[numerical_cols].corr()
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:  # Strong correlation threshold
                        strong_correlations.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            print(f"   • Found {len(strong_correlations)} strong correlations (|r| > 0.5)")
            for corr in strong_correlations[:5]:  # Show top 5
                print(f"     - {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f}")
            
            # Create correlation heatmap
            plt.figure(figsize=(15, 12))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, fmt='.2f', linewidths=0.5)
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig('results/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close instead of show
            
            self.eda_metrics['strong_correlations_count'] = len(strong_correlations)
    
    def run_complete_eda(self):
        """Run the complete EDA pipeline with MLflow tracking"""
        
        with mlflow.start_run(run_name=f"EDA_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
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
                self.create_visualizations()
                
                # Step 6: Correlation analysis
                self.correlation_analysis()
                
                # Log all metrics to MLflow
                for metric_name, metric_value in self.eda_metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                    else:
                        # For complex metrics, log as parameters
                        mlflow.log_param(metric_name, str(metric_value)[:500])  # Limit length
                
                # Create summary report
                self.create_summary_report()
                
                # Log artifacts
                if os.path.exists('results'):
                    mlflow.log_artifacts('results')
                
                print("\n✅ EDA Analysis completed successfully!")
                print(f"📊 Results saved to MLflow with tracking URI: {MLFLOW_TRACKING_URI}")
                
            except Exception as e:
                print(f"❌ Error during EDA analysis: {e}")
                mlflow.log_param("error", str(e))
                raise
    
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
        os.makedirs('results', exist_ok=True)
        with open('results/eda_summary_report.json', 'w') as f:
            json.dump(summary, f, indent=4, default=str)
        
        print("   ✓ Summary report saved as 'eda_summary_report.json'")
        
        # Log summary as artifact
        mlflow.log_dict(summary, "eda_summary_report.json")


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
        print(f"📈 View results in MLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()