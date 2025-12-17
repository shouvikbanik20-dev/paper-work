"""
SHAP Explainability - FIXED VERSION
===================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

def run_shap_analysis(dataset_name='cm1', model_type='rf'):
    """Run SHAP analysis with error handling"""
    
    print(f"\n{'='*80}")
    print(f"🔍 SHAP Analysis: {model_type.upper()} on {dataset_name.upper()}")
    print(f"{'='*80}\n")
    
    # Load data
    print("📂 Loading data...")
    train_df = pd.read_csv(f'data/processed/{dataset_name}_train.csv')
    test_df = pd.read_csv(f'data/processed/{dataset_name}_test.csv')
    
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target']).head(50)  # Use 50 samples
    y_test = test_df['target'].head(50)
    
    feature_names = X_train.columns.tolist()
    
    print(f"✅ Data loaded: {len(X_train)} train, {len(X_test)} test samples")
    
    # Train model
    print(f"\n🤖 Training {model_type.upper()}...")
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    else:
        model = SVC(probability=True, random_state=42, class_weight='balanced')
    
    model.fit(X_train, y_train)
    print(f"✅ Accuracy: {model.score(X_test, y_test):.3f}")
    
    # SHAP explainer
    print(f"\n🔬 Creating SHAP explainer...")
    if model_type == 'rf':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        background = shap.sample(X_train, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test)
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
    
    print(f"✅ SHAP values computed: shape {shap_values.shape}")
    
    # Create output directory
    os.makedirs('results/explainability', exist_ok=True)
    
    # Plot 1: Summary plot
    print(f"\n📊 Creating summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title(f'Feature Importance - {model_type.upper()} on {dataset_name.upper()}', 
             fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'results/explainability/{dataset_name}_{model_type}_summary.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved summary plot")
    
    # Plot 2: Bar chart
    print(f"\n📊 Creating bar chart...")
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_shap
    }).sort_values('Importance', ascending=False)
    
    top15 = importance_df.head(15).sort_values('Importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top15)), top15['Importance'], color='steelblue')
    plt.yticks(range(len(top15)), top15['Feature'])
    plt.xlabel('Mean |SHAP Value|', fontsize=12)
    plt.title(f'Top 15 Features - {model_type.upper()} on {dataset_name.upper()}', 
             fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/explainability/{dataset_name}_{model_type}_bar.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved bar chart")
    
    # Print top features
    print(f"\n📋 Top 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    print(f"\n{'='*80}")
    print(f"✅ Analysis complete for {dataset_name.upper()} - {model_type.upper()}")
    print(f"{'='*80}\n")
    
    return importance_df

# Run analyses
print("""
╔══════════════════════════════════════════════════════════════╗
║        SHAP EXPLAINABILITY ANALYSIS - FIXED                  ║
╚══════════════════════════════════════════════════════════════╝
""")

# Analyze CM1 with RF and SVM
results_rf = run_shap_analysis('cm1', 'rf')
results_svm = run_shap_analysis('cm1', 'svm')

print("\n" + "="*80)
print("✅ ALL ANALYSES COMPLETE!")
print("📁 Check results/explainability/ for plots")
print("="*80)