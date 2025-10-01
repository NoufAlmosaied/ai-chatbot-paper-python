#!/usr/bin/env python3
"""
Retrain Random Forest Model with Corrected Features
This script retrains the phishing detection model using the fixed feature extraction
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import the FIXED feature extractor
from services.feature_extractor import FeatureExtractor


class ModelRetrainer:
    """Retrain the Random Forest model with corrected feature extraction."""

    def __init__(self, dataset_path: str = None):
        """Initialize the retrainer."""
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent / "data" / "raw" / "Phishing_Legitimate_full.csv"

        self.dataset_path = Path(dataset_path)
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.best_params = None

        # Create output directories
        self.models_dir = Path(__file__).parent.parent / "models" / "baseline"
        self.reports_dir = Path(__file__).parent.parent / "reports" / "retraining"
        self.plots_dir = self.reports_dir / "plots"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*70}")
        print(f"MODEL RETRAINING WITH CORRECTED FEATURES")
        print(f"{'='*70}")
        print(f"Dataset: {self.dataset_path}")
        print(f"Output directory: {self.models_dir}")

    def load_dataset(self):
        """Load the phishing dataset."""
        print(f"\n[1/7] Loading dataset...")

        df = pd.read_csv(self.dataset_path)
        print(f"  Dataset shape: {df.shape}")
        print(f"  Features: {df.shape[1] - 2}")  # Excluding 'id' and 'CLASS_LABEL'

        # Check label distribution
        label_counts = df['CLASS_LABEL'].value_counts()
        print(f"  Class distribution:")
        print(f"    Legitimate (0): {label_counts[0]} ({label_counts[0]/len(df)*100:.1f}%)")
        print(f"    Phishing (1): {label_counts[1]} ({label_counts[1]/len(df)*100:.1f}%)")

        # Extract features and labels
        X = df.drop(['id', 'CLASS_LABEL'], axis=1).values
        y = df['CLASS_LABEL'].values

        # Store feature names
        self.feature_names = df.drop(['id', 'CLASS_LABEL'], axis=1).columns.tolist()

        print(f"  ✓ Dataset loaded successfully")
        return X, y

    def split_data(self, X, y):
        """Split data into train, validation, and test sets."""
        print(f"\n[2/7] Splitting data...")

        # First split: 70% train+val, 30% test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        # Second split: 70% train, 15% validation (from remaining 85%)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
        )

        print(f"  Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        print(f"  ✓ Data split completed")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning with GridSearchCV."""
        print(f"\n[3/7] Hyperparameter tuning...")

        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced']
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        print(f"  Parameter grid:")
        for param, values in param_grid.items():
            print(f"    {param}: {values}")

        print(f"  Running GridSearchCV with 5-fold cross-validation...")
        grid_search = GridSearchCV(
            rf, param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        print(f"\n  Best parameters found:")
        for param, value in self.best_params.items():
            print(f"    {param}: {value}")
        print(f"  Best cross-validation F1 score: {grid_search.best_score_:.4f}")
        print(f"  ✓ Hyperparameter tuning completed")

        return grid_search.best_estimator_

    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the Random Forest model."""
        print(f"\n[4/7] Training model with best parameters...")

        self.model = RandomForestClassifier(
            **self.best_params,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)

        # Evaluate on validation set
        val_pred = self.model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)

        print(f"  Training accuracy: {train_acc:.4f}")
        print(f"  Validation accuracy: {val_acc:.4f}")
        print(f"  ✓ Model training completed")

        return self.model

    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model on test set."""
        print(f"\n[5/7] Evaluating model on test set...")

        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        print(f"\n  Test Set Performance:")
        print(f"  {'='*50}")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"  ROC AUC:   {roc_auc:.4f}")
        print(f"  {'='*50}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n  Confusion Matrix:")
        print(f"                Predicted")
        print(f"                Legit  Phishing")
        print(f"  Actual Legit  {cm[0,0]:5d}  {cm[0,1]:8d}")
        print(f"         Phish  {cm[1,0]:5d}  {cm[1,1]:8d}")

        # Classification report
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

        # Store results
        self.results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'test_samples': len(y_test)
        }

        print(f"  ✓ Model evaluation completed")

        return self.results

    def plot_results(self, X_test, y_test):
        """Generate visualization plots."""
        print(f"\n[6/7] Generating plots...")

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # 1. Confusion Matrix Heatmap
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legitimate', 'Phishing'],
                    yticklabels=['Legitimate', 'Phishing'])
        plt.title('Confusion Matrix - Retrained Model')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        print(f"  ✓ Saved confusion_matrix.png")

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {self.results["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Retrained Model')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'roc_curve.png', dpi=300)
        plt.close()
        print(f"  ✓ Saved roc_curve.png")

        # 3. Feature Importance (Top 20)
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-20:][::-1]

        plt.figure(figsize=(10, 8))
        plt.barh(range(20), importances[indices])
        plt.yticks(range(20), [self.feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'feature_importance.png', dpi=300)
        plt.close()
        print(f"  ✓ Saved feature_importance.png")

        print(f"  ✓ All plots generated")

    def save_model(self):
        """Save the retrained model and results."""
        print(f"\n[7/7] Saving model and results...")

        # Save model
        model_file = self.models_dir / "random_forest.pkl"
        joblib.dump(self.model, model_file)
        print(f"  ✓ Saved model to {model_file}")

        # Save feature names
        feature_names_file = Path(__file__).parent.parent / "data" / "processed" / "phishing" / "feature_names.json"
        feature_names_file.parent.mkdir(parents=True, exist_ok=True)
        with open(feature_names_file, 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        print(f"  ✓ Saved feature names to {feature_names_file}")

        # Save results
        results_file = self.reports_dir / "retraining_results.json"
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'dataset': str(self.dataset_path),
            'best_params': self.best_params,
            'metrics': self.results,
            'feature_names': self.feature_names
        }
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"  ✓ Saved results to {results_file}")

        print(f"\n{'='*70}")
        print(f"MODEL RETRAINING COMPLETED SUCCESSFULLY")
        print(f"{'='*70}")
        print(f"\nModel Performance Summary:")
        print(f"  Accuracy:  {self.results['accuracy']*100:.2f}%")
        print(f"  Precision: {self.results['precision']*100:.2f}%")
        print(f"  Recall:    {self.results['recall']*100:.2f}%")
        print(f"  F1 Score:  {self.results['f1']*100:.2f}%")
        print(f"  ROC AUC:   {self.results['roc_auc']:.4f}")
        print(f"\nFiles saved:")
        print(f"  Model: {model_file}")
        print(f"  Results: {results_file}")
        print(f"  Plots: {self.plots_dir}")
        print(f"\n{'='*70}\n")


def main():
    """Main retraining function."""
    # Initialize retrainer
    retrainer = ModelRetrainer()

    # Load dataset
    X, y = retrainer.load_dataset()

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = retrainer.split_data(X, y)

    # Hyperparameter tuning
    best_model = retrainer.hyperparameter_tuning(X_train, y_train)

    # Train model
    model = retrainer.train_model(X_train, y_train, X_val, y_val)

    # Evaluate model
    results = retrainer.evaluate_model(X_test, y_test)

    # Plot results
    retrainer.plot_results(X_test, y_test)

    # Save model
    retrainer.save_model()


if __name__ == "__main__":
    main()
