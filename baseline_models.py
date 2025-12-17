"""
Baseline Models for Software Defect Prediction
===============================================
This script trains and evaluates traditional ML models:
- Random Forest (RF)
- Support Vector Machine (SVM)
- Multilayer Perceptron (MLP)

Author: Your Name
Date: December 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

class DefectPredictor:
    """Class to train and evaluate defect prediction models"""
    
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.model = None
        self.results = {}
        
    def create_model(self):
        """Create model based on type"""
        if self.model_type == 'rf':
            print("📊 Creating Random Forest model...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',  # Handle imbalance
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            print("📊 Creating SVM model...")
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',  # Handle imbalance
                probability=True,  # For ROC-AUC
                random_state=42
            )
        elif self.model_type == 'mlp':
            print("📊 Creating MLP (Neural Network) model...")
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"✅ {self.model_type.upper()} model created!")
        return self.model
    
    def train(self, X_train, y_train):
        """Train the model"""
        print(f"\n{'='*80}")
        print(f"🚀 TRAINING {self.model_type.upper()} MODEL")
        print(f"{'='*80}")
        print(f"Training samples: {len(X_train)}")
        print(f"Defective samples: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
        
        if self.model is None:
            self.create_model()
        
        print("\n⏳ Training in progress...")
        self.model.fit(X_train, y_train)
        print("✅ Training complete!")
        
    def evaluate(self, X_test, y_test, dataset_name='Test'):
        """Evaluate the model"""
        print(f"\n{'='*80}")
        print(f"📈 EVALUATING {self.model_type.upper()} MODEL")
        print(f"{'='*80}")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            auc_roc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_roc = 0.0
        
        # Store results
        self.results = {
            'model': self.model_type.upper(),
            'dataset': dataset_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # Print results
        print("\n📊 PERFORMANCE METRICS:")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Value':<15}")
        print(f"{'='*60}")
        print(f"{'Accuracy':<20} {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"{'Precision':<20} {precision:.4f} ({precision*100:.2f}%)")
        print(f"{'Recall':<20} {recall:.4f} ({recall*100:.2f}%)")
        print(f"{'F1-Score':<20} {f1:.4f} ({f1*100:.2f}%)")
        print(f"{'AUC-ROC':<20} {auc_roc:.4f} ({auc_roc*100:.2f}%)")
        print(f"{'='*60}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n📋 CONFUSION MATRIX:")
        print(f"{'='*60}")
        print(f"                Predicted")
        print(f"              Non-Defect  Defect")
        print(f"Actual Non-D    {cm[0,0]:>6}     {cm[0,1]:>6}")
        print(f"       Defect   {cm[1,0]:>6}     {cm[1,1]:>6}")
        print(f"{'='*60}")
        
        # Classification Report
        print("\n📄 CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Non-Defective', 'Defective']))
        
        return self.results

def load_dataset(train_file, test_file):
    """Load train and test datasets"""
    print(f"\n📂 Loading datasets...")
    print(f"   Train: {train_file}")
    print(f"   Test:  {test_file}")
    
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    # Split features and target
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']
    
    print(f"✅ Datasets loaded!")
    print(f"   Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"   Test:  {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def plot_results(all_results, dataset_name, output_dir='results'):
    """Create visualization of results"""
    print(f"\n📊 Creating visualizations...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    models = [r['model'] for r in all_results]
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Baseline Models Performance - {dataset_name}', 
                 fontsize=16, fontweight='bold')
    
    # Plot each metric
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        values = [r[metric] for r in all_results]
        bars = ax.bar(models, values, color=['#2ecc71', '#3498db', '#e74c3c'])
        
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Hide the last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, f'{dataset_name}_baseline_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_file}")
    
    plt.close()
    
    # Create ROC curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    for result, color in zip(all_results, colors):
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
        auc = result['auc_roc']
        ax.plot(fpr, tpr, color=color, lw=2, 
               label=f"{result['model']} (AUC = {auc:.3f})")
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curves - {dataset_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Save ROC curve
    roc_file = os.path.join(output_dir, f'{dataset_name}_roc_curves.png')
    plt.savefig(roc_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {roc_file}")
    
    plt.close()

def save_results_table(all_results, dataset_name, output_dir='results'):
    """Save results to CSV"""
    print(f"\n💾 Saving results table...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame
    results_df = pd.DataFrame([
        {
            'Model': r['model'],
            'Dataset': r['dataset'],
            'Accuracy': f"{r['accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1-Score': f"{r['f1_score']:.4f}",
            'AUC-ROC': f"{r['auc_roc']:.4f}"
        }
        for r in all_results
    ])
    
    # Save to CSV
    output_file = os.path.join(output_dir, f'{dataset_name}_baseline_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"✅ Saved: {output_file}")
    
    # Print table
    print("\n" + "="*80)
    print("RESULTS SUMMARY TABLE")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)

def train_and_evaluate_all(dataset_name='cm1'):
    """Train and evaluate all baseline models on a dataset"""
    print(f"\n{'#'*80}")
    print(f"# BASELINE MODELS TRAINING AND EVALUATION")
    print(f"# Dataset: {dataset_name.upper()}")
    print(f"{'#'*80}")
    
    # Load data
    train_file = f'data/processed/{dataset_name}_train.csv'
    test_file = f'data/processed/{dataset_name}_test.csv'
    
    X_train, X_test, y_train, y_test = load_dataset(train_file, test_file)
    
    # Train all models
    models = ['rf', 'svm', 'mlp']
    all_results = []
    
    for model_type in models:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_type.upper()}")
        print(f"{'='*80}")
        
        predictor = DefectPredictor(model_type=model_type)
        predictor.train(X_train, y_train)
        results = predictor.evaluate(X_test, y_test, dataset_name=dataset_name.upper())
        all_results.append(results)
    
    # Create visualizations
    plot_results(all_results, dataset_name.upper())
    
    # Save results
    save_results_table(all_results, dataset_name.upper())
    
    print(f"\n{'#'*80}")
    print(f"# ✅ COMPLETED: {dataset_name.upper()}")
    print(f"{'#'*80}\n")
    
    return all_results

def main():
    """Main function to run baseline experiments"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           BASELINE MODELS FOR DEFECT PREDICTION              ║
    ║                                                              ║
    ║  Models: Random Forest, SVM, MLP                            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # List of datasets to process
    datasets = ['cm1', 'pc1', 'jm1', 'kc1']
    
    # Process each dataset
    all_dataset_results = {}
    
    for dataset in datasets:
        train_file = f'data/processed/{dataset}_train.csv'
        if os.path.exists(train_file):
            results = train_and_evaluate_all(dataset)
            all_dataset_results[dataset] = results
        else:
            print(f"\n⚠️ Dataset not found: {dataset}")
            print(f"   Skipping...")
    
    # Create summary across all datasets
    print(f"\n{'#'*80}")
    print(f"# FINAL SUMMARY - ALL DATASETS")
    print(f"{'#'*80}\n")
    
    for dataset, results in all_dataset_results.items():
        print(f"\n{dataset.upper()}:")
        print(f"{'='*80}")
        for r in results:
            print(f"  {r['model']:>5} | Acc: {r['accuracy']:.3f} | "
                  f"Prec: {r['precision']:.3f} | Rec: {r['recall']:.3f} | "
                  f"F1: {r['f1_score']:.3f} | AUC: {r['auc_roc']:.3f}")
    
    print(f"\n{'#'*80}")
    print(f"# ✅ ALL BASELINE EXPERIMENTS COMPLETED!")
    print(f"# 📁 Results saved in: results/")
    print(f"# 📊 Check the generated charts and CSV files")
    print(f"{'#'*80}\n")

if __name__ == "__main__":
    main()