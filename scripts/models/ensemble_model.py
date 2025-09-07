#!/usr/bin/env python3
"""
Ensemble Model Architecture for Phishing Detection
Phase 3: Model Development and Initial Testing
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Any, Tuple
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class EnsembleModel:
    """Ensemble model combining multiple classifiers."""
    
    def __init__(self, base_models: Dict[str, Any] = None, 
                 ensemble_method: str = 'voting',
                 random_state: int = 42):
        """
        Initialize ensemble model.
        
        Args:
            base_models: Dictionary of trained base models
            ensemble_method: 'voting', 'stacking', 'weighted_voting', or 'blending'
            random_state: Random seed for reproducibility
        """
        self.base_models = base_models or {}
        self.ensemble_method = ensemble_method
        self.random_state = random_state
        self.ensemble_model = None
        self.model_weights = None
        self.performance_scores = {}
        
    def add_model(self, name: str, model: Any):
        """Add a model to the ensemble."""
        self.base_models[name] = model
        print(f"Added {name} to ensemble")
    
    def remove_model(self, name: str):
        """Remove a model from the ensemble."""
        if name in self.base_models:
            del self.base_models[name]
            print(f"Removed {name} from ensemble")
    
    def create_voting_ensemble(self, voting: str = 'soft') -> VotingClassifier:
        """Create a voting ensemble."""
        if not self.base_models:
            raise ValueError("No base models available for ensemble")
        
        estimators = [(name, model) for name, model in self.base_models.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1
        )
        
        return ensemble
    
    def create_stacking_ensemble(self, meta_learner: Any = None) -> StackingClassifier:
        """Create a stacking ensemble with a meta-learner."""
        if not self.base_models:
            raise ValueError("No base models available for ensemble")
        
        if meta_learner is None:
            meta_learner = LogisticRegression(random_state=self.random_state)
        
        estimators = [(name, model) for name, model in self.base_models.items()]
        
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,  # Use 5-fold cross-validation for creating meta-features
            n_jobs=-1
        )
        
        return ensemble
    
    def create_weighted_voting_ensemble(self, weights: List[float] = None) -> VotingClassifier:
        """Create a weighted voting ensemble."""
        if not self.base_models:
            raise ValueError("No base models available for ensemble")
        
        if weights is None:
            # Use uniform weights if not provided
            weights = [1.0] * len(self.base_models)
        
        if len(weights) != len(self.base_models):
            raise ValueError("Number of weights must match number of models")
        
        self.model_weights = dict(zip(self.base_models.keys(), weights))
        
        estimators = [(name, model) for name, model in self.base_models.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )
        
        return ensemble
    
    def create_blending_ensemble(self, X_blend: np.ndarray, y_blend: np.ndarray,
                                meta_learner: Any = None) -> Any:
        """
        Create a blending ensemble using a holdout validation set.
        
        Args:
            X_blend: Blending set features
            y_blend: Blending set labels
            meta_learner: Meta-learner model
        """
        if not self.base_models:
            raise ValueError("No base models available for ensemble")
        
        if meta_learner is None:
            meta_learner = LogisticRegression(random_state=self.random_state)
        
        # Generate predictions from base models on blending set
        blend_features = []
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X_blend)[:, 1]
            else:
                predictions = model.predict(X_blend)
            blend_features.append(predictions)
        
        # Stack predictions as features for meta-learner
        blend_features = np.column_stack(blend_features)
        
        # Train meta-learner
        meta_learner.fit(blend_features, y_blend)
        
        # Create custom ensemble object
        class BlendingEnsemble:
            def __init__(self, base_models, meta_learner):
                self.base_models = base_models
                self.meta_learner = meta_learner
            
            def predict(self, X):
                predictions = []
                for name, model in self.base_models.items():
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)[:, 1]
                    else:
                        pred = model.predict(X)
                    predictions.append(pred)
                
                meta_features = np.column_stack(predictions)
                return self.meta_learner.predict(meta_features)
            
            def predict_proba(self, X):
                predictions = []
                for name, model in self.base_models.items():
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)[:, 1]
                    else:
                        pred = model.predict(X)
                    predictions.append(pred)
                
                meta_features = np.column_stack(predictions)
                if hasattr(self.meta_learner, 'predict_proba'):
                    return self.meta_learner.predict_proba(meta_features)
                else:
                    pred = self.meta_learner.predict(meta_features)
                    proba = np.zeros((len(pred), 2))
                    proba[range(len(pred)), pred.astype(int)] = 1
                    return proba
        
        return BlendingEnsemble(self.base_models, meta_learner)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray = None, y_val: np.ndarray = None,
             weights: List[float] = None):
        """Train the ensemble model."""
        print(f"\nTraining {self.ensemble_method} ensemble...")
        
        if self.ensemble_method == 'voting':
            self.ensemble_model = self.create_voting_ensemble()
        elif self.ensemble_method == 'stacking':
            self.ensemble_model = self.create_stacking_ensemble()
        elif self.ensemble_method == 'weighted_voting':
            self.ensemble_model = self.create_weighted_voting_ensemble(weights)
        elif self.ensemble_method == 'blending':
            if X_val is None or y_val is None:
                raise ValueError("Blending requires validation set")
            self.ensemble_model = self.create_blending_ensemble(X_val, y_val)
            # For blending, base models should already be trained
            return self._evaluate_ensemble(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        # Train ensemble (for voting and stacking)
        self.ensemble_model.fit(X_train, y_train)
        
        return self._evaluate_ensemble(X_train, y_train, X_val, y_val)
    
    def _evaluate_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Evaluate ensemble performance."""
        # Training performance
        train_pred = self.ensemble_model.predict(X_train)
        train_scores = self._calculate_metrics(y_train, train_pred)
        
        print(f"  Training Accuracy: {train_scores['accuracy']:.4f}")
        print(f"  Training F1: {train_scores['f1']:.4f}")
        
        # Validation performance
        val_scores = {}
        if X_val is not None and y_val is not None:
            val_pred = self.ensemble_model.predict(X_val)
            val_scores = self._calculate_metrics(y_val, val_pred)
            
            print(f"  Validation Accuracy: {val_scores['accuracy']:.4f}")
            print(f"  Validation F1: {val_scores['f1']:.4f}")
        
        self.performance_scores = {
            'train': train_scores,
            'validation': val_scores
        }
        
        return self.performance_scores
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the ensemble."""
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not trained yet")
        
        return self.ensemble_model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities from the ensemble."""
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not trained yet")
        
        if hasattr(self.ensemble_model, 'predict_proba'):
            return self.ensemble_model.predict_proba(X)
        else:
            # Return binary predictions if probabilities not available
            predictions = self.ensemble_model.predict(X)
            proba = np.zeros((len(predictions), 2))
            proba[range(len(predictions)), predictions.astype(int)] = 1
            return proba
    
    def get_model_contributions(self, X: np.ndarray) -> pd.DataFrame:
        """Analyze individual model contributions to ensemble predictions."""
        if not self.base_models:
            return pd.DataFrame()
        
        contributions = {}
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                contributions[name] = model.predict_proba(X)[:, 1]
            else:
                contributions[name] = model.predict(X)
        
        df = pd.DataFrame(contributions)
        df['ensemble'] = self.predict_proba(X)[:, 1] if hasattr(self.ensemble_model, 'predict_proba') else self.predict(X)
        
        return df
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray,
                        n_trials: int = 100) -> List[float]:
        """
        Optimize ensemble weights using validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of random weight combinations to try
        """
        if not self.base_models:
            raise ValueError("No base models available")
        
        best_weights = None
        best_score = 0
        n_models = len(self.base_models)
        
        print(f"\nOptimizing ensemble weights ({n_trials} trials)...")
        
        for trial in range(n_trials):
            # Generate random weights
            weights = np.random.dirichlet(np.ones(n_models))
            
            # Create weighted ensemble
            ensemble = self.create_weighted_voting_ensemble(weights.tolist())
            
            # Train on available data (assuming base models are already trained)
            predictions = np.zeros(len(X_val))
            for i, (name, model) in enumerate(self.base_models.items()):
                if hasattr(model, 'predict_proba'):
                    model_pred = model.predict_proba(X_val)[:, 1]
                else:
                    model_pred = model.predict(X_val)
                predictions += weights[i] * model_pred
            
            # Threshold predictions
            predictions = (predictions > 0.5).astype(int)
            
            # Calculate F1 score
            score = f1_score(y_val, predictions)
            
            if score > best_score:
                best_score = score
                best_weights = weights.tolist()
        
        print(f"  Best F1 score: {best_score:.4f}")
        print(f"  Best weights: {dict(zip(self.base_models.keys(), best_weights))}")
        
        self.model_weights = dict(zip(self.base_models.keys(), best_weights))
        return best_weights
    
    def save_ensemble(self, output_dir: str = 'models/ensemble'):
        """Save the ensemble model."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save ensemble model
        ensemble_path = output_path / f"{self.ensemble_method}_ensemble.pkl"
        joblib.dump(self.ensemble_model, ensemble_path)
        print(f"Saved ensemble to {ensemble_path}")
        
        # Save weights if applicable
        if self.model_weights:
            weights_path = output_path / "ensemble_weights.pkl"
            joblib.dump(self.model_weights, weights_path)
            print(f"Saved weights to {weights_path}")
        
        # Save performance scores
        import json
        scores_path = output_path / "ensemble_scores.json"
        with open(scores_path, 'w') as f:
            json.dump(self.performance_scores, f, indent=2)
        print(f"Saved scores to {scores_path}")


def main():
    """Test ensemble model."""
    print("\n" + "="*70)
    print("ENSEMBLE MODEL TEST")
    print("="*70)
    
    # Create dummy models for testing
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    # Generate dummy data
    np.random.seed(42)
    X_train = np.random.randn(1000, 50)
    y_train = np.random.randint(0, 2, 1000)
    X_val = np.random.randn(200, 50)
    y_val = np.random.randint(0, 2, 200)
    
    # Train base models
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    
    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train, y_train)
    
    # Create ensemble
    base_models = {
        'random_forest': rf,
        'logistic_regression': lr,
        'svm': svm
    }
    
    # Test voting ensemble
    print("\n1. Testing Voting Ensemble")
    ensemble = EnsembleModel(base_models, ensemble_method='voting')
    ensemble.train(X_train, y_train, X_val, y_val)
    
    # Test stacking ensemble
    print("\n2. Testing Stacking Ensemble")
    ensemble_stack = EnsembleModel(base_models, ensemble_method='stacking')
    ensemble_stack.train(X_train, y_train, X_val, y_val)
    
    # Test weighted voting with optimized weights
    print("\n3. Testing Weighted Voting Ensemble")
    ensemble_weighted = EnsembleModel(base_models, ensemble_method='weighted_voting')
    best_weights = ensemble_weighted.optimize_weights(X_val, y_val, n_trials=20)
    ensemble_weighted.train(X_train, y_train, X_val, y_val, weights=best_weights)
    
    print("\n✓ Ensemble model test complete")


if __name__ == "__main__":
    main()