"""
Machine Learning Service
Loads and manages the trained phishing detection models
"""

import numpy as np
import joblib
from pathlib import Path
import json
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class MLService:
    """Service for managing ML model predictions."""
    
    def __init__(self, model_path: str = None):
        """Initialize the ML service."""
        self.model = None
        self.model_name = "random_forest"  # Best performing model
        self.feature_names = None
        self.stats = {
            'total_predictions': 0,
            'phishing_detected': 0,
            'legitimate_detected': 0
        }
        
        # Set default model path
        if model_path is None:
            model_path = Path(__file__).parent.parent / "models" / "baseline"
        else:
            model_path = Path(model_path)

        self.model_path = model_path
        self.load_model()
        self.load_feature_names()
    
    def load_model(self):
        """Load the trained model from disk."""
        try:
            model_file = self.model_path / f"{self.model_name}.pkl"
            if model_file.exists():
                self.model = joblib.load(model_file)
                logger.info(f"Model loaded successfully: {self.model_name}")
                return True
            else:
                logger.error(f"Model file not found: {model_file}")
                # Try to load any available model
                for pkl_file in self.model_path.glob("*.pkl"):
                    self.model = joblib.load(pkl_file)
                    self.model_name = pkl_file.stem
                    logger.info(f"Loaded alternative model: {self.model_name}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def load_feature_names(self):
        """Load feature names from the processed data."""
        try:
            feature_file = Path(__file__).parent.parent / "data" / "processed" / "phishing" / "feature_names.json"
            if feature_file.exists():
                with open(feature_file, 'r') as f:
                    self.feature_names = json.load(f)
                logger.info(f"Loaded {len(self.feature_names)} feature names")
            else:
                # Use default feature names if file not found
                self.feature_names = [f"feature_{i}" for i in range(48)]
                logger.warning("Using default feature names")
        except Exception as e:
            logger.error(f"Failed to load feature names: {str(e)}")
            self.feature_names = [f"feature_{i}" for i in range(48)]
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        Make a prediction using the loaded model.
        
        Args:
            features: Feature vector (1D or 2D array)
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            # Ensure features is 2D
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]
            
            # Get feature importance for Random Forest
            top_features = []
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                top_indices = np.argsort(importances)[-5:][::-1]  # Top 5 features
                
                for idx in top_indices:
                    if idx < len(self.feature_names):
                        top_features.append({
                            'name': self.feature_names[idx],
                            'importance': float(importances[idx])
                        })
            
            # Update statistics
            self.stats['total_predictions'] += 1
            if prediction == 1:
                self.stats['phishing_detected'] += 1
            else:
                self.stats['legitimate_detected'] += 1
            
            return {
                'is_phishing': bool(prediction == 1),
                'probability': float(probability[1]),  # Probability of being phishing
                'confidence': float(max(probability)),
                'model_used': self.model_name,
                'top_features': top_features
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_batch(self, features_list: List[np.ndarray]) -> List[Dict]:
        """
        Make predictions for multiple samples.
        
        Args:
            features_list: List of feature vectors
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        for features in features_list:
            try:
                pred = self.predict(features)
                predictions.append(pred)
            except Exception as e:
                predictions.append({
                    'error': str(e),
                    'is_phishing': None
                })
        return predictions
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if not self.is_loaded():
            return {'error': 'Model not loaded'}
        
        info = {
            'model_name': self.model_name,
            'model_type': type(self.model).__name__,
            'features_expected': len(self.feature_names) if self.feature_names else 'Unknown'
        }
        
        # Add model-specific info
        if hasattr(self.model, 'n_estimators'):
            info['n_estimators'] = self.model.n_estimators
        if hasattr(self.model, 'max_depth'):
            info['max_depth'] = self.model.max_depth
            
        return info
    
    def get_stats(self) -> Dict:
        """Get prediction statistics."""
        stats = self.stats.copy()
        if stats['total_predictions'] > 0:
            stats['phishing_rate'] = stats['phishing_detected'] / stats['total_predictions']
            stats['legitimate_rate'] = stats['legitimate_detected'] / stats['total_predictions']
        else:
            stats['phishing_rate'] = 0
            stats['legitimate_rate'] = 0
        
        stats['model_info'] = self.get_model_info()
        return stats
    
    def reset_stats(self):
        """Reset prediction statistics."""
        self.stats = {
            'total_predictions': 0,
            'phishing_detected': 0,
            'legitimate_detected': 0
        }