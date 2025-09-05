#!/usr/bin/env python3
"""
Prepare real phishing dataset for Phase 3 training.
Uses the Phishing_Legitimate_full.csv dataset with 48 features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

def prepare_phishing_data():
    """Load and prepare the phishing dataset."""
    
    # Load the dataset
    print("Loading phishing dataset...")
    df = pd.read_csv('../data/raw/Phishing_Legitimate_full.csv')
    
    # Separate features and labels
    X = df.drop(['id', 'CLASS_LABEL'], axis=1)
    y = df['CLASS_LABEL']
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    # Split the data (60% train, 20% val, 20% test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the processed data
    output_dir = Path('../data/processed/phishing')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'X_train.npy', X_train_scaled)
    np.save(output_dir / 'X_val.npy', X_val_scaled)
    np.save(output_dir / 'X_test.npy', X_test_scaled)
    np.save(output_dir / 'y_train.npy', y_train.values)
    np.save(output_dir / 'y_val.npy', y_val.values)
    np.save(output_dir / 'y_test.npy', y_test.values)
    
    # Save feature names
    feature_names = X.columns.tolist()
    with open(output_dir / 'feature_names.json', 'w') as f:
        json.dump(feature_names, f, indent=2)
    
    # Save metadata
    metadata = {
        'dataset': 'Phishing_Legitimate_full.csv',
        'total_samples': len(df),
        'num_features': len(feature_names),
        'feature_names': feature_names,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'class_distribution': {
            'legitimate': int((y == 0).sum()),
            'phishing': int((y == 1).sum())
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Data saved to {output_dir}")
    print(f"   - X_train.npy: {X_train_scaled.shape}")
    print(f"   - X_val.npy: {X_val_scaled.shape}")
    print(f"   - X_test.npy: {X_test_scaled.shape}")
    print(f"   - Feature names and metadata saved")
    
    return output_dir

if __name__ == "__main__":
    output_dir = prepare_phishing_data()
    print(f"\n🎯 Ready to run Phase 3 with real data!")
    print(f"   Update run_phase3_pipeline.py to use: {output_dir}")
