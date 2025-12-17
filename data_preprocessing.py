"""
Data Preprocessing Script for Defect Prediction
================================================
This script processes ARFF files from PROMISE dataset and prepares them for ML models.

Author: Your Name
Date: December 2025
"""

import os
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_arff_file(filepath):
    """Load ARFF file and convert to pandas DataFrame"""
    print(f"\n{'='*80}")
    print(f"Loading: {filepath}")
    print(f"{'='*80}")
    
    try:
        data, meta = arff.loadarff(filepath)
        df = pd.DataFrame(data)
        print(f"✅ Loaded successfully!")
        print(f"   Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None

def convert_byte_strings(df):
    """Convert byte strings to regular strings"""
    print("\n🔄 Converting byte strings to regular strings...")
    
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace("b'", "").str.replace("'", "")
    
    print("✅ Conversion complete!")
    return df

def handle_target_column(df):
    """Identify and convert target column to binary (0/1)"""
    print("\n🎯 Processing target column...")
    
    # Assume last column is target
    target_col = df.columns[-1]
    print(f"   Target column: {target_col}")
    print(f"   Unique values: {df[target_col].unique()}")
    
    # Convert to binary
    # Handle different label formats: true/false, Y/N, yes/no, 1/0
    df[target_col] = df[target_col].astype(str).str.lower()
    
    # Map to 1 (defective) and 0 (non-defective)
    defect_values = ['true', 'yes', 'y', '1', '1.0']
    df[target_col] = df[target_col].apply(lambda x: 1 if x in defect_values else 0)
    
    print(f"✅ Target converted to binary (0/1)")
    print(f"   Class distribution:")
    print(f"   - Non-defective (0): {(df[target_col] == 0).sum()} samples")
    print(f"   - Defective (1): {(df[target_col] == 1).sum()} samples")
    
    return df, target_col

def handle_missing_values(df):
    """Handle missing values if any"""
    print("\n🔍 Checking for missing values...")
    
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    
    if len(missing_cols) == 0:
        print("✅ No missing values found!")
        return df
    
    print(f"⚠️ Found missing values in {len(missing_cols)} columns:")
    for col, count in missing_cols.items():
        print(f"   - {col}: {count} missing ({count/len(df)*100:.2f}%)")
    
    print("\n🔧 Handling missing values:")
    # For numeric columns, fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"   - Filled {col} with median: {median_val}")
    
    # For categorical columns, fill with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"   - Filled {col} with mode: {mode_val}")
    
    print("✅ Missing values handled!")
    return df

def remove_duplicates(df):
    """Remove duplicate rows"""
    print("\n🔍 Checking for duplicates...")
    
    initial_count = len(df)
    df = df.drop_duplicates()
    final_count = len(df)
    removed = initial_count - final_count
    
    if removed > 0:
        print(f"⚠️ Removed {removed} duplicate rows")
    else:
        print("✅ No duplicates found!")
    
    return df

def split_features_target(df, target_col):
    """Split features and target"""
    print("\n✂️ Splitting features and target...")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Convert all features to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill any NaN created during conversion
    X = X.fillna(X.median())
    
    print(f"✅ Split complete!")
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    
    return X, y

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """Create train/test split"""
    print(f"\n📊 Creating train/test split (test_size={test_size})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print("✅ Split created!")
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    print(f"   Train defects: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
    print(f"   Test defects: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

def normalize_features(X_train, X_test):
    """Normalize features using StandardScaler"""
    print("\n📏 Normalizing features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("✅ Normalization complete!")
    print(f"   Mean of first feature: {X_train_scaled.iloc[:, 0].mean():.6f} (should be ~0)")
    print(f"   Std of first feature: {X_train_scaled.iloc[:, 0].std():.6f} (should be ~1)")
    
    return X_train_scaled, X_test_scaled, scaler

def save_processed_data(X_train, X_test, y_train, y_test, dataset_name, output_dir):
    """Save processed data to CSV files"""
    print(f"\n💾 Saving processed data...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine features and target for saving
    train_data = X_train.copy()
    train_data['target'] = y_train.values
    
    test_data = X_test.copy()
    test_data['target'] = y_test.values
    
    # Save files
    train_file = os.path.join(output_dir, f"{dataset_name}_train.csv")
    test_file = os.path.join(output_dir, f"{dataset_name}_test.csv")
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    print(f"✅ Saved:")
    print(f"   - {train_file}")
    print(f"   - {test_file}")
    
    return train_file, test_file

def process_dataset(input_file, output_dir='data/processed'):
    """Main processing pipeline"""
    print(f"\n{'#'*80}")
    print(f"# PROCESSING DATASET")
    print(f"{'#'*80}")
    
    # Extract dataset name
    dataset_name = os.path.basename(input_file).replace('.arff', '').replace('.csv', '')
    
    # Load data
    df = load_arff_file(input_file)
    if df is None:
        return False
    
    # Convert byte strings
    df = convert_byte_strings(df)
    
    # Handle target column
    df, target_col = handle_target_column(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Split features and target
    X, y = split_features_target(df, target_col)
    
    # Create train/test split
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    
    # Normalize features
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)
    
    # Save processed data
    train_file, test_file = save_processed_data(
        X_train_scaled, X_test_scaled, y_train, y_test, dataset_name, output_dir
    )
    
    print(f"\n{'='*80}")
    print(f"✅ PROCESSING COMPLETE FOR {dataset_name.upper()}!")
    print(f"{'='*80}\n")
    
    return True

def main():
    """Process all PROMISE datasets"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        DATA PREPROCESSING FOR DEFECT PREDICTION              ║
    ║                    PROMISE Dataset                           ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Define input files
    datasets = [
        'data/raw/PROMISE/cm1.arff',
        'data/raw/PROMISE/pc1.arff',
        'data/raw/PROMISE/jm1.arff',
        'data/raw/PROMISE/kc1.arff'
    ]
    
    # Process each dataset
    success_count = 0
    for dataset_file in datasets:
        if os.path.exists(dataset_file):
            success = process_dataset(dataset_file)
            if success:
                success_count += 1
        else:
            print(f"\n⚠️ File not found: {dataset_file}")
            print(f"   Skipping...")
    
    print(f"\n{'#'*80}")
    print(f"# SUMMARY")
    print(f"{'#'*80}")
    print(f"✅ Successfully processed: {success_count}/{len(datasets)} datasets")
    print(f"📁 Output directory: data/processed/")
    print(f"\n{'#'*80}\n")

if __name__ == "__main__":
    main()