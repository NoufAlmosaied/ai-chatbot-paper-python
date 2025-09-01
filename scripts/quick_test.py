#!/usr/bin/env python3
"""Quick test script to create processed data for Phase 3"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Create dummy processed data for testing Phase 3
def create_test_data():
    print("Creating test data for Phase 3...")
    
    # Create output directory
    output_dir = Path('data/processed/email')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dummy data
    n_train = 1000
    n_val = 200
    n_test = 200
    n_features = 100
    
    # Create random features and labels
    X_train = np.random.randn(n_train, n_features)
    X_val = np.random.randn(n_val, n_features)
    X_test = np.random.randn(n_test, n_features)
    
    # Binary labels (0 or 1)
    y_train = np.random.randint(0, 2, n_train)
    y_val = np.random.randint(0, 2, n_val)
    y_test = np.random.randint(0, 2, n_test)
    
    # Save arrays
    np.save(output_dir / 'X_train.npy', X_train)
    np.save(output_dir / 'X_val.npy', X_val)
    np.save(output_dir / 'X_test.npy', X_test)
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'y_val.npy', y_val)
    np.save(output_dir / 'y_test.npy', y_test)
    
    print(f"✓ Created test data:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Saved to: {output_dir}")
    
    return True

if __name__ == "__main__":
    create_test_data()