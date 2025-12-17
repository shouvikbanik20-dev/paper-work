"""
SHAP Explainability for Defect Prediction Models
=================================================
Generate feature importance explanations using SHAP values

Author: Shouvik Banik
Date: December 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

class DefectExplainer:
    """Generate explanations for defect predictions using SHAP"""
    
    def __init__(self, model_type='rf', dataset_name='cm1'):
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.model = None
        self.explainer = None
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        
        print(f"\n{'='*80}")
        print(f"🔍 INITIALIZING EXPLAINER")
        print(f"{'='*80}")
        print(f"Model: {model_type.upper()}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"{'='*80}\n")
    
    def load_data(self):
        """Load preprocessed data"""
        print("📂 Loading data...")
        
        train_file = f'data/processed/{self.dataset_name}_train.csv'
        test_file = f'data/processed/{self.dataset_name}_test.csv'
        
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        # Split features and target
        self.X_train = train_df.drop(columns=['target'])
        self.y_train = train_df['target']
        self.X_test = test_df.drop(columns=['target'])
        self.y_test = test_df['target']
        
        self.feature_names = self.X_train.columns.tolist()
        
        print(f"✅ Data loaded!")
        print(f"   Features: {len(self.feature_names)}")
        print(f"   Train samples: {len(self.X_train)}")
        print(f"   Test samples: {len(self.X_test)}")
    
    def train_model(self):
        """Train the model (or load if exists)"""
        print(f"\n🤖 Training {self.model_type.upper()} model...")
        
        if self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        elif self.model_type == 'mlp':
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=42
            )
        
        self.model.fit(self.X_train, self.y_train)
        
        # Calculate accuracy
        train_acc = self.model.score(self.X_train, self.y_train)
        test_acc = self.model.score(self.X_test, self.y_test)
        
        print(f"✅ Model trained!")
        print(f"   Train accuracy: {train_acc:.4f}")
        print(f"   Test accuracy: {test_acc:.4f}")
    
    def create_explainer(self):
        """Create SHAP explainer"""
        print(f"\n🔬 Creating SHAP explainer...")
        
        if self.model_type == 'rf':
            # Tree explainer for Random Forest
            self.explainer = shap.TreeExplainer(self.model)
            print("✅ TreeExplainer created (fast!)")
        else:
            # Kernel explainer for SVM and MLP
            # Use subset of training data as background
            background = shap.sample(self.X_train, 100)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                background
            )
            print("✅ KernelExplainer created (may be slow)")
    
    def compute_shap_values(self, num_samples=None):
        """Compute SHAP values for test set"""
        print(f"\n📊 Computing SHAP values...")
        
        if num_samples is None:
            X_explain = self.X_test
        else:
            X_explain = self.X_test.head(num_samples)
        
        if self.model_type == 'rf':
            # For tree models, get SHAP values for class 1 (defective)
            shap_values = self.explainer.shap_values(X_explain)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1
        else:
            # For kernel explainer
            shap_values = self.explainer.shap_values(X_explain)
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]  # Class 1
        
        print(f"✅ SHAP values computed for {len(X_explain)} samples")
        
        return shap_values, X_explain
    
    def plot_summary(self, shap_values, X_explain):
        """Create SHAP summary plot"""
        print(f"\n📊 Creating summary plot...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            X_explain,
            feature_names=self.feature_names,
            show=False,
            plot_size=(10, 8)
        )
        plt.title(f'SHAP Feature Importance - {self.model_type.upper()} on {self.dataset_name.upper()}',
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save
        output_dir = 'results/explainability'
        os.makedirs(output_dir, exist_ok=True)
        output_file = f'{output_dir}/{self.dataset_name}_{self.model_type}_shap_summary.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.close()
    
    def plot_bar_chart(self, shap_values, X_explain):
        """Create SHAP bar chart showing mean importance"""
        print(f"\n📊 Creating bar chart...")
        
        # Calculate mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        # Ensure 1D array
        if isinstance(mean_shap, pd.Series):
            mean_shap = mean_shap.values
        if len(mean_shap.shape) > 1:
            mean_shap = mean_shap.flatten()
        
        # Match lengths
        n_features = min(len(mean_shap), len(self.feature_names))
        mean_shap = mean_shap[:n_features]
        feature_names = self.feature_names[:n_features]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_shap
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=True)
        
        # Plot top 15 features
        top_features = importance_df.tail(15)
        
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_features)), top_features['Importance'].values, color='steelblue')
        plt.yticks(range(len(top_features)), top_features['Feature'].values)
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top 15 Feature Importance - {self.model_type.upper()} on {self.dataset_name.upper()}',
                fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Save
        output_dir = 'results/explainability'
        output_file = f'{output_dir}/{self.dataset_name}_{self.model_type}_shap_bar.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.close()
        
        return importance_df
    
    def explain_single_prediction(self, sample_idx):
        """Explain a single prediction in detail"""
        print(f"\n🔍 Explaining prediction for sample {sample_idx}...")
        
        # Get sample
        sample = self.X_test.iloc[sample_idx:sample_idx+1]
        true_label = self.y_test.iloc[sample_idx]
        
        # Make prediction
        pred = self.model.predict(sample)[0]
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(sample)[0]
        else:
            proba = None
        
        # Get SHAP values
        if self.model_type == 'rf':
            shap_values = self.explainer.shap_values(sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            shap_values = self.explainer.shap_values(sample)
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
        
        # shap_values = shap_values[0]
        
        # # ✅ FIXED: Use bar plot instead of waterfall (more compatible)
        # plt.figure(figsize=(10, 8))
        
        # # # Get top 10 features by absolute SHAP value
        # # top_indices = np.argsort(np.abs(shap_values))[-10:]
        # # top_features = [self.feature_names[i] for i in top_indices]
        # # top_values = shap_values[top_indices]
        # # Get top 10 features by absolute SHAP value
        
        # # Get top 10 features by absolute SHAP value
        # top_indices = np.argsort(np.abs(shap_values))[-10:]
        
        # # Convert to list and ensure integer indices
        # feature_list = list(self.feature_names)
        # top_features = [feature_list[int(i)] for i in top_indices]
        # top_values = shap_values[top_indices]

        # Ensure SHAP values are 1D
        # Ensure SHAP values are 1D
        shap_values = np.array(shap_values).reshape(-1)

        # Align SHAP values with feature names
        n_features = min(len(shap_values), len(self.feature_names))
        shap_values = shap_values[:n_features]
        feature_list = list(self.feature_names)[:n_features]

        # Get top 10 features by absolute SHAP value
        top_indices = np.argsort(np.abs(shap_values))[-10:]

        top_features = [feature_list[i] for i in top_indices]
        top_values = shap_values[top_indices]


        # Create horizontal bar plot
        colors = ['red' if x < 0 else 'green' for x in top_values]
        plt.barh(range(len(top_values)), top_values, color=colors, alpha=0.7)
        plt.yticks(range(len(top_values)), top_features)
        plt.xlabel('SHAP Value (impact on prediction)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, val in enumerate(top_values):
            plt.text(val + (0.01 if val > 0 else -0.01), i, f'{val:.3f}',
                    ha='left' if val > 0 else 'right', va='center', fontsize=9)
        
        plt.title(f'Feature Contributions - Sample {sample_idx}\n' +
                f'Prediction: {"Defective" if pred == 1 else "Non-Defective"} ' +
                f'| True: {"Defective" if true_label == 1 else "Non-Defective"}' +
                (f' | Confidence: {proba[pred]*100:.1f}%' if proba is not None else ''),
                fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save
        output_dir = 'results/explainability'
        output_file = f'{output_dir}/{self.dataset_name}_{self.model_type}_sample_{sample_idx}_explanation.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {output_file}")
        plt.close()
        
        # Generate natural language explanation
        explanation_text = self.generate_nl_explanation(shap_values, sample, pred, proba)
        
        return explanation_text
    
    def generate_nl_explanation(self, shap_values, sample, pred, proba):
        """Generate natural language explanation"""
        
        # Get top 5 most important features
        top_indices = np.argsort(np.abs(shap_values))[-5:][::-1]
        
        explanation = f"\n{'='*80}\n"
        explanation += f"📝 EXPLANATION FOR PREDICTION\n"
        explanation += f"{'='*80}\n\n"
        
        if pred == 1:
            explanation += f"⚠️ Prediction: DEFECTIVE\n"
            if proba is not None:
                explanation += f"   Confidence: {proba[1]*100:.1f}%\n"
        else:
            explanation += f"✅ Prediction: NON-DEFECTIVE\n"
            if proba is not None:
                explanation += f"   Confidence: {proba[0]*100:.1f}%\n"
        
        explanation += f"\n🔍 Top 5 Contributing Factors:\n\n"
        
        for rank, idx in enumerate(top_indices, 1):
            # Convert feature_names to list if needed
            if isinstance(self.feature_names, pd.Index):
                feature_list = self.feature_names.tolist()
            else:
                feature_list = list(self.feature_names)
            
            feature = feature_list[int(idx)]
            value = sample.iloc[0, idx]
            shap_val = shap_values[idx]
            
            direction = "INCREASES" if shap_val > 0 else "DECREASES"
            magnitude = "strongly" if abs(shap_val) > 0.1 else "moderately"
            
            explanation += f"{rank}. {feature} = {value:.4f}\n"
            explanation += f"   This {magnitude} {direction} defect likelihood\n"
            explanation += f"   (SHAP value: {shap_val:+.4f})\n\n"
        
        explanation += f"{'='*80}\n"
        
        return explanation
    
    def run_full_analysis(self, num_samples=100):
        """Run complete explainability analysis"""
        print(f"\n{'='*80}")
        print(f"🚀 RUNNING FULL EXPLAINABILITY ANALYSIS")
        print(f"{'='*80}\n")
        
        # Load data and train model
        self.load_data()
        self.train_model()
        
        # Create explainer
        self.create_explainer()
        
        # Compute SHAP values
        shap_values, X_explain = self.compute_shap_values(num_samples)
        
        # Create visualizations
        self.plot_summary(shap_values, X_explain)
        importance_df = self.plot_bar_chart(shap_values, X_explain)
        
        # Explain specific predictions
        print(f"\n{'='*80}")
        print(f"🔍 GENERATING EXAMPLE EXPLANATIONS")
        print(f"{'='*80}")
        
        # Find interesting samples to explain
        predictions = self.model.predict(self.X_test)
        
        # True positive (correctly predicted defect)
        tp_indices = np.where((predictions == 1) & (self.y_test == 1))[0]
        if len(tp_indices) > 0:
            tp_idx = tp_indices[0]
            print(f"\n📌 Example 1: TRUE POSITIVE (sample {tp_idx})")
            tp_explanation = self.explain_single_prediction(tp_idx)
            print(tp_explanation)
        
        # False positive (incorrectly predicted defect)
        fp_indices = np.where((predictions == 1) & (self.y_test == 0))[0]
        if len(fp_indices) > 0:
            fp_idx = fp_indices[0]
            print(f"\n📌 Example 2: FALSE POSITIVE (sample {fp_idx})")
            fp_explanation = self.explain_single_prediction(fp_idx)
            print(fp_explanation)
        
        # False negative (missed defect)
        fn_indices = np.where((predictions == 0) & (self.y_test == 1))[0]
        if len(fn_indices) > 0:
            fn_idx = fn_indices[0]
            print(f"\n📌 Example 3: FALSE NEGATIVE (sample {fn_idx})")
            fn_explanation = self.explain_single_prediction(fn_idx)
            print(fn_explanation)
        
        print(f"\n{'='*80}")
        print(f"✅ EXPLAINABILITY ANALYSIS COMPLETE!")
        print(f"{'='*80}")
        print(f"📁 Check results/explainability/ for visualizations")
        print(f"{'='*80}\n")
        
        return importance_df

def main():
    """Run explainability analysis on all models and datasets"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        SHAP EXPLAINABILITY ANALYSIS                          ║
    ║                                                              ║
    ║  Generating interpretable explanations for predictions      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Datasets and models to analyze
    datasets = ['cm1']  # Start with CM1, add others if time permits
    models = ['rf', 'svm']  # RF and SVM (MLP is slower)
    
    all_importance = {}
    
    for dataset in datasets:
        for model in models:
            print(f"\n{'#'*80}")
            print(f"# ANALYZING: {model.upper()} on {dataset.upper()}")
            print(f"{'#'*80}")
            
            explainer = DefectExplainer(model_type=model, dataset_name=dataset)
            importance_df = explainer.run_full_analysis(num_samples=100)
            
            all_importance[f"{dataset}_{model}"] = importance_df
    
    print(f"\n{'#'*80}")
    print(f"# ✅ ALL EXPLAINABILITY ANALYSES COMPLETE!")
    print(f"# 📁 Results saved in: results/explainability/")
    print(f"{'#'*80}\n")

if __name__ == "__main__":
    main()