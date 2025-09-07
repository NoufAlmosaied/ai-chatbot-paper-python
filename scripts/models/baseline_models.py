#!/usr/bin/env python3
"""
Baseline Machine Learning Models for Phishing Detection
Phase 3: Model Development and Initial Testing
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Try to import XGBoost, but make it optional
XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    # XGBoost might fail to load on some systems (e.g., macOS without libomp)
    XGBClassifier = None
    print(f"Warning: XGBoost not available: {str(e)[:100]}")
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, Any, Tuple
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class BaselineModels:
    """Collection of baseline ML models for phishing detection."""
    
    def __init__(self, random_state: int = 42):
        """Initialize baseline models."""
        self.random_state = random_state
        self.models = self._initialize_models()
        self.trained_models = {}
        self.model_scores = {}
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize all baseline models with default parameters."""
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'svm_linear': SVC(
                kernel='linear',
                random_state=self.random_state,
                probability=True,
                class_weight='balanced'
            ),
            'svm_rbf': SVC(
                kernel='rbf',
                random_state=self.random_state,
                probability=True,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                learning_rate=0.1
            ),
            'naive_bayes': MultinomialNB(alpha=1.0),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_state,
                class_weight='balanced'
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        
        return models
    
    def train_model(self, model_name: str, X_train: np.ndarray, 
                   y_train: np.ndarray, X_val: np.ndarray = None, 
                   y_val: np.ndarray = None) -> Dict:
        """Train a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        print(f"\nTraining {model_name}...")
        model = self.models[model_name]
        
        # Special handling for models that need non-negative features
        if model_name == 'naive_bayes':
            # Ensure non-negative features for Naive Bayes
            X_train = np.abs(X_train)
            if X_val is not None:
                X_val = np.abs(X_val)
        
        # Train the model
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        
        # Evaluate on training set
        train_pred = model.predict(X_train)
        train_scores = self._calculate_metrics(y_train, train_pred)
        
        # Evaluate on validation set if provided
        val_scores = {}
        if X_val is not None and y_val is not None:
            val_pred = model.predict(X_val)
            val_scores = self._calculate_metrics(y_val, val_pred)
        
        # Store scores
        self.model_scores[model_name] = {
            'train': train_scores,
            'validation': val_scores
        }
        
        print(f"  Training Accuracy: {train_scores['accuracy']:.4f}")
        if val_scores:
            print(f"  Validation Accuracy: {val_scores['accuracy']:.4f}")
        
        return {
            'model': model,
            'train_scores': train_scores,
            'val_scores': val_scores
        }
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train all baseline models."""
        print("\n" + "="*70)
        print("TRAINING ALL BASELINE MODELS")
        print("="*70)
        
        results = {}
        for model_name in self.models.keys():
            try:
                result = self.train_model(model_name, X_train, y_train, X_val, y_val)
                results[model_name] = result
            except Exception as e:
                print(f"  ❌ Error training {model_name}: {str(e)}")
                results[model_name] = None
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions with a trained model."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Special handling for models that need non-negative features
        if model_name == 'naive_bayes':
            X = np.abs(X)
        
        return model.predict(X)
    
    def predict_proba(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Special handling for models that need non-negative features
        if model_name == 'naive_bayes':
            X = np.abs(X)
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            # For models without predict_proba, return binary predictions
            predictions = model.predict(X)
            proba = np.zeros((len(predictions), 2))
            proba[range(len(predictions)), predictions] = 1
            return proba
    
    def get_feature_importance(self, model_name: str) -> np.ndarray:
        """Get feature importance for tree-based models."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_).flatten()
        else:
            return None
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models."""
        if not self.model_scores:
            return pd.DataFrame()
        
        comparison_data = []
        for model_name, scores in self.model_scores.items():
            if scores.get('validation'):
                comparison_data.append({
                    'Model': model_name,
                    'Train Accuracy': scores['train']['accuracy'],
                    'Val Accuracy': scores['validation']['accuracy'],
                    'Val Precision': scores['validation']['precision'],
                    'Val Recall': scores['validation']['recall'],
                    'Val F1': scores['validation']['f1']
                })
            else:
                comparison_data.append({
                    'Model': model_name,
                    'Train Accuracy': scores['train']['accuracy'],
                    'Val Accuracy': None,
                    'Val Precision': None,
                    'Val Recall': None,
                    'Val F1': None
                })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Val F1', ascending=False, na_position='last')
        return df
    
    def save_models(self, output_dir: str = 'models/baseline'):
        """Save all trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            model_path = output_path / f"{model_name}.pkl"
            joblib.dump(model, model_path)
            print(f"  Saved {model_name} to {model_path}")
        
        # Save scores
        scores_path = output_path / "model_scores.json"
        with open(scores_path, 'w') as f:
            json.dump(self.model_scores, f, indent=2)
        
        print(f"  Saved scores to {scores_path}")
    
    def load_model(self, model_name: str, model_path: str):
        """Load a saved model."""
        model = joblib.load(model_path)
        self.trained_models[model_name] = model
        return model


def main():
    """Test baseline models."""
    print("\n" + "="*70)
    print("BASELINE MODELS TEST")
    print("="*70)
    
    # Generate dummy data for testing
    np.random.seed(42)
    X_train = np.random.randn(1000, 50)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.randn(200, 50)
    y_val = np.random.randint(0, 2, 200)
    
    # Initialize baseline models
    baseline = BaselineModels()
    
    # Train all models
    results = baseline.train_all_models(X_train, y_train, X_val, y_val)
    
    # Compare models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    comparison = baseline.compare_models()
    print(comparison.to_string())
    
    # Get best model
    if not comparison.empty:
        best_model = comparison.iloc[0]['Model']
        print(f"\n✓ Best model: {best_model}")


if __name__ == "__main__":
    main()