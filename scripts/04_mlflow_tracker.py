
import mlflow
import mlflow.tracking
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlruns.db"

class MLflowTracker:
    """
    MLflow experiment tracking and model management utilities
    """
    
    def __init__(self):
        """Initialize MLflow tracker"""
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.client = mlflow.tracking.MlflowClient()
        
        print("🔗 MLflow Tracker initialized")
        print(f"📍 Tracking URI: {MLFLOW_TRACKING_URI}")
    
    def list_experiments(self):
        """List all MLflow experiments"""
        print("\n📋 Available MLflow Experiments:")
        print("=" * 50)
        
        experiments = self.client.search_experiments()
        
        for exp in experiments:
            print(f"🧪 Experiment: {exp.name}")
            print(f"   📁 ID: {exp.experiment_id}")
            print(f"   📊 Lifecycle Stage: {exp.lifecycle_stage}")
            if exp.tags:
                print(f"   🏷️  Tags: {exp.tags}")
            print()
        
        return experiments
    
    def get_experiment_runs(self, experiment_name):
        """Get all runs for a specific experiment"""
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                print(f"❌ Experiment '{experiment_name}' not found")
                return pd.DataFrame()
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                output_format="pandas"
            )
            
            print(f"\n📊 Found {len(runs)} runs in experiment '{experiment_name}'")
            return runs
            
        except Exception as e:
            print(f"❌ Error getting runs: {e}")
            return pd.DataFrame()
    
    def display_experiment_summary(self, experiment_name="Bird_Migration_ML_Training"):
        """Display detailed summary of an experiment"""
        print(f"\n🎯 Experiment Summary: {experiment_name}")
        print("=" * 60)
        
        runs_df = self.get_experiment_runs(experiment_name)
        
        if runs_df.empty:
            print("No runs found in this experiment.")
            return
        
        # Basic statistics
        print(f"📈 Total Runs: {len(runs_df)}")
        print(f"📅 Date Range: {runs_df['start_time'].min()} to {runs_df['start_time'].max()}")
        
        # Status summary
        status_counts = runs_df['status'].value_counts()
        print(f"\n📋 Run Status Summary:")
        for status, count in status_counts.items():
            print(f"   {status}: {count}")
        
        # Model performance summary
        metric_columns = [col for col in runs_df.columns if col.startswith('metrics.')]
        
        if metric_columns:
            print(f"\n🏆 Model Performance Summary:")
            
            # Classification metrics
            classification_metrics = ['metrics.test_accuracy', 'metrics.test_f1', 'metrics.test_precision']
            classification_metrics = [m for m in classification_metrics if m in metric_columns]
            
            if classification_metrics:
                print("\n🎯 Classification Models:")
                class_df = runs_df[runs_df['params.task_type'] == 'classification']
                if not class_df.empty:
                    for metric in classification_metrics:
                        if metric in class_df.columns:
                            best_value = class_df[metric].max()
                            best_run = class_df[class_df[metric] == best_value]['tags.mlflow.runName'].iloc[0]
                            print(f"   📊 Best {metric.replace('metrics.', '')}: {best_value:.4f} ({best_run})")
            
            # Regression metrics
            regression_metrics = ['metrics.test_r2', 'metrics.test_rmse', 'metrics.test_mae']
            regression_metrics = [m for m in regression_metrics if m in metric_columns]
            
            if regression_metrics:
                print("\n📏 Regression Models:")
                reg_df = runs_df[runs_df['params.task_type'] == 'regression']
                if not reg_df.empty:
                    for metric in regression_metrics:
                        if metric in reg_df.columns:
                            if 'r2' in metric:
                                best_value = reg_df[metric].max()
                            else:  # Lower is better for RMSE, MAE
                                best_value = reg_df[metric].min()
                            best_run_idx = reg_df[metric].idxmax() if 'r2' in metric else reg_df[metric].idxmin()
                            best_run = reg_df.loc[best_run_idx, 'tags.mlflow.runName']
                            print(f"   📊 Best {metric.replace('metrics.', '')}: {best_value:.4f} ({best_run})")
        
        return runs_df
    
    def compare_models(self, experiment_name="Bird_Migration_ML_Training", task_type="classification"):
        """Compare models within an experiment"""
        print(f"\n⚖️  Model Comparison - {task_type.title()}")
        print("=" * 50)
        
        runs_df = self.get_experiment_runs(experiment_name)
        
        if runs_df.empty:
            print("No runs found to compare.")
            return
        
        # Filter by task type
        task_runs = runs_df[runs_df['params.task_type'] == task_type].copy()
        
        if task_runs.empty:
            print(f"No {task_type} runs found.")
            return
        
        # Prepare data for visualization
        model_names = task_runs['params.model_name'].fillna('Unknown')
        
        if task_type == "classification":
            metrics = ['metrics.test_accuracy', 'metrics.test_f1', 'metrics.test_precision']
            metric_names = ['Accuracy', 'F1-Score', 'Precision']
        else:
            metrics = ['metrics.test_r2', 'metrics.test_rmse', 'metrics.test_mae']
            metric_names = ['R²', 'RMSE', 'MAE']
        
        # Create comparison plots
        available_metrics = [m for m in metrics if m in task_runs.columns]
        available_metric_names = [metric_names[i] for i, m in enumerate(metrics) if m in available_metrics]
        
        if available_metrics:
            fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 6))
            if len(available_metrics) == 1:
                axes = [axes]
            
            for i, (metric, metric_name) in enumerate(zip(available_metrics, available_metric_names)):
                metric_data = task_runs.groupby('params.model_name')[metric].max()
                
                axes[i].bar(metric_data.index, metric_data.values)
                axes[i].set_title(f'{metric_name} by Model')
                axes[i].set_ylabel(metric_name)
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for j, (model, value) in enumerate(metric_data.items()):
                    if not np.isnan(value):
                        axes[i].text(j, value, f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'results/model_comparison_{task_type}_mlflow.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Print detailed comparison table
        comparison_data = []
        for _, run in task_runs.iterrows():
            model_info = {
                'Model': run['params.model_name'],
                'Run_Name': run['tags.mlflow.runName'],
                'Duration': run['end_time'] - run['start_time'] if pd.notna(run['end_time']) else 'Running'
            }
            
            # Add available metrics
            for metric in available_metrics:
                if metric in run and pd.notna(run[metric]):
                    model_info[metric.replace('metrics.', '').title()] = f"{run[metric]:.4f}"
                else:
                    model_info[metric.replace('metrics.', '').title()] = 'N/A'
            
            comparison_data.append(model_info)
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n📊 Detailed Model Comparison:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def show_experiment_timeline(self, experiment_name="Bird_Migration_ML_Training"):
        """Show experiment timeline"""
        print(f"\n📅 Experiment Timeline: {experiment_name}")
        print("=" * 50)
        
        runs_df = self.get_experiment_runs(experiment_name)
        
        if runs_df.empty:
            return
        
        # Convert timestamps
        runs_df['start_time'] = pd.to_datetime(runs_df['start_time'])
        runs_df['end_time'] = pd.to_datetime(runs_df['end_time'])
        
        # Create timeline plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, (_, run) in enumerate(runs_df.iterrows()):
            start_time = run['start_time']
            end_time = run['end_time'] if pd.notna(run['end_time']) else datetime.now()
            duration = end_time - start_time
            
            model_name = run['params.model_name'] if 'params.model_name' in run else 'Unknown'
            task_type = run['params.task_type'] if 'params.task_type' in run else 'Unknown'
            
            color = 'blue' if task_type == 'classification' else 'green'
            
            ax.barh(i, duration.total_seconds()/60, left=start_time, 
                   color=color, alpha=0.7, height=0.6)
            
            ax.text(start_time + duration/2, i, f'{model_name}\n({task_type})', 
                   ha='center', va='center', fontsize=8)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Runs')
        ax.set_title(f'Experiment Timeline: {experiment_name}')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/experiment_timeline.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def list_registered_models(self):
        """List all registered models"""
        print("\n🏷️  Registered Models")
        print("=" * 40)
        
        try:
            registered_models = self.client.search_registered_models()
            
            if not registered_models:
                print("No registered models found.")
                return
            
            for model in registered_models:
                print(f"\n📦 Model: {model.name}")
                print(f"   📝 Description: {model.description or 'No description'}")
                print(f"   🏷️  Tags: {model.tags}")
                
                # Get model versions
                versions = self.client.get_latest_versions(model.name)
                for version in versions:
                    print(f"   🔖 Version {version.version}: {version.current_stage}")
            
            return registered_models
            
        except Exception as e:
            print(f"❌ Error listing registered models: {e}")
            return []
    
    def export_experiment_data(self, experiment_name, output_file=None):
        """Export experiment data to CSV"""
        if output_file is None:
            output_file = f"results/mlflow_experiment_{experiment_name.replace(' ', '_')}.csv"
        
        print(f"\n💾 Exporting experiment data to: {output_file}")
        
        runs_df = self.get_experiment_runs(experiment_name)
        
        if runs_df.empty:
            print("No data to export.")
            return
        
        # Clean and export
        runs_df.to_csv(output_file, index=False)
        print(f"✅ Exported {len(runs_df)} runs to {output_file}")
        
        return output_file
    
    def generate_experiment_report(self, experiment_name="Bird_Migration_ML_Training"):
        """Generate comprehensive experiment report"""
        print(f"\n📑 Generating Comprehensive Report for: {experiment_name}")
        print("=" * 70)
        
        # Get experiment data
        runs_df = self.display_experiment_summary(experiment_name)
        
        if runs_df.empty:
            print("❌ No data found for report generation.")
            return
        
        # Compare models
        print("\n" + "="*70)
        self.compare_models(experiment_name, "classification")
        
        print("\n" + "="*70)
        self.compare_models(experiment_name, "regression")
        
        # Show timeline
        print("\n" + "="*70)
        self.show_experiment_timeline(experiment_name)
        
        # Export data
        print("\n" + "="*70)
        csv_file = self.export_experiment_data(experiment_name)
        
        # Create summary statistics
        self.create_summary_statistics(runs_df, experiment_name)
        
        print(f"\n✅ Comprehensive report generated for '{experiment_name}'")
        print("📂 Check 'results/' directory for exported files and visualizations")
        
        return runs_df
    
    def create_summary_statistics(self, runs_df, experiment_name):
        """Create summary statistics visualization"""
        print("\n📊 Creating Summary Statistics...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'MLflow Experiment Statistics: {experiment_name}', fontsize=16)
        
        # 1. Runs by model type
        if 'params.model_name' in runs_df.columns:
            model_counts = runs_df['params.model_name'].value_counts()
            axes[0, 0].pie(model_counts.values, labels=model_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Runs by Model Type')
        
        # 2. Runs by task type
        if 'params.task_type' in runs_df.columns:
            task_counts = runs_df['params.task_type'].value_counts()
            axes[0, 1].bar(task_counts.index, task_counts.values)
            axes[0, 1].set_title('Runs by Task Type')
            axes[0, 1].set_ylabel('Number of Runs')
        
        # 3. Run duration distribution
        runs_df['duration'] = pd.to_datetime(runs_df['end_time']) - pd.to_datetime(runs_df['start_time'])
        runs_df['duration_minutes'] = runs_df['duration'].dt.total_seconds() / 60
        
        valid_durations = runs_df['duration_minutes'].dropna()
        if not valid_durations.empty:
            axes[1, 0].hist(valid_durations, bins=15, alpha=0.7, color='skyblue')
            axes[1, 0].set_title('Run Duration Distribution')
            axes[1, 0].set_xlabel('Duration (minutes)')
            axes[1, 0].set_ylabel('Frequency')
        
        # 4. Success rate over time
        runs_df['date'] = pd.to_datetime(runs_df['start_time']).dt.date
        success_by_date = runs_df.groupby('date')['status'].apply(lambda x: (x == 'FINISHED').mean())
        
        if len(success_by_date) > 1:
            axes[1, 1].plot(success_by_date.index, success_by_date.values, marker='o')
            axes[1, 1].set_title('Success Rate Over Time')
            axes[1, 1].set_ylabel('Success Rate')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/mlflow_experiment_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function to demonstrate MLflow tracking capabilities"""
    print("🐦 Bird Migration MLflow Experiment Tracker")
    print("=" * 60)
    
    # Initialize tracker
    tracker = MLflowTracker()
    
    # List all experiments
    experiments = tracker.list_experiments()
    
    # Generate comprehensive reports for key experiments
    experiment_names = [
        "Bird_Migration_EDA_Analysis",
        "Bird_Migration_Data_Preprocessing", 
        "Bird_Migration_ML_Training"
    ]
    
    for exp_name in experiment_names:
        try:
            print(f"\n{'='*80}")
            print(f"📊 ANALYZING EXPERIMENT: {exp_name}")
            print('='*80)
            
            tracker.generate_experiment_report(exp_name)
            
        except Exception as e:
            print(f"⚠️  Could not analyze experiment '{exp_name}': {e}")
    
    # List registered models
    tracker.list_registered_models()
    
    print(f"\n🎉 MLflow Analysis Complete!")
    print(f"🌐 Start MLflow UI with: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print("📁 Check 'results/' directory for exported analysis files")


if __name__ == "__main__":
    main()