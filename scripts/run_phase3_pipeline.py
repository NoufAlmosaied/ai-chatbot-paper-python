#!/usr/bin/env python3
"""
Main Pipeline for Phase 3: Model Development and Initial Testing
This script orchestrates the complete model training and evaluation pipeline.
"""

import sys
sys.path.append('.')

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import Phase 2 modules
from data_preprocessing import DataPreprocessor
from automated_tagging import AutomatedTagger

# Import Phase 3 modules
from models.baseline_models import BaselineModels
from models.cross_validation import CrossValidationFramework

# Optional imports
try:
    from models.deep_learning_models import DeepLearningModels
    DL_AVAILABLE = True
except ImportError as e:
    # Only set to None if truly not available
    DL_AVAILABLE = False
    print(f"Warning: Deep learning models not available: {e}")

try:
    from models.ensemble_model import EnsembleModel
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False
    print("Warning: Ensemble models not available")


class Phase3Pipeline:
    """Orchestrates Phase 3: Model Development and Initial Testing."""
    
    def __init__(self, config_path: str = None):
        """Initialize the pipeline."""
        self.config = self._load_config(config_path)
        self.start_time = None
        self.pipeline_report = {}
        self.models = {}
        self.results = {}
        
    def _load_config(self, config_path: str = None) -> dict:
        """Load pipeline configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'data': {
                    'processed_dir': '../data/processed/phishing',  # Use real phishing data
                    'use_sample': False,  # Use full dataset
                    'sample_size': 10000
                },
                'models': {
                    'train_baseline': True,
                    'train_deep_learning': True,  # Enable DL with TensorFlow
                    'train_ensemble': True,
                    'baseline_models': ['logistic_regression', 'random_forest', 
                                      'xgboost', 'svm_rbf'],
                    'dl_models': ['lstm', 'bidirectional_lstm', 'cnn_lstm'],
                    'ensemble_methods': ['voting', 'stacking']
                },
                'training': {
                    'cv_folds': 5,
                    'cv_strategy': 'stratified',
                    'epochs_dl': 10,
                    'batch_size': 32,
                    'early_stopping': True
                },
                'evaluation': {
                    'metrics': ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
                    'generate_plots': True,
                    'statistical_tests': True
                },
                'output': {
                    'models_dir': 'models',
                    'reports_dir': 'reports',
                    'plots_dir': 'plots'
                }
            }
    
    def load_data(self) -> tuple:
        """Load processed data from Phase 2."""
        print("\n" + "="*70)
        print("STEP 1: LOADING PROCESSED DATA")
        print("="*70)
        
        data_dir = Path(self.config['data']['processed_dir'])
        
        if not data_dir.exists():
            print(f"❌ Processed data not found at {data_dir}")
            print("   Please run Phase 2 pipeline first")
            return None, None, None, None, None, None
        
        try:
            # Load data arrays
            X_train = np.load(data_dir / 'X_train.npy')
            X_val = np.load(data_dir / 'X_val.npy')
            X_test = np.load(data_dir / 'X_test.npy')
            y_train = np.load(data_dir / 'y_train.npy')
            y_val = np.load(data_dir / 'y_val.npy')
            y_test = np.load(data_dir / 'y_test.npy')
            
            print(f"✓ Loaded data successfully")
            print(f"  Training samples: {X_train.shape[0]}")
            print(f"  Validation samples: {X_val.shape[0]}")
            print(f"  Test samples: {X_test.shape[0]}")
            print(f"  Feature dimensions: {X_train.shape[1]}")
            
            # Use sample if configured
            if self.config['data']['use_sample']:
                sample_size = min(self.config['data']['sample_size'], len(X_train))
                print(f"\n  Using sample of {sample_size} training examples")
                indices = np.random.choice(len(X_train), sample_size, replace=False)
                X_train = X_train[indices]
                y_train = y_train[indices]
            
            self.pipeline_report['data_loaded'] = {
                'train_samples': X_train.shape[0],
                'val_samples': X_val.shape[0],
                'test_samples': X_test.shape[0],
                'features': X_train.shape[1]
            }
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None, None, None, None, None, None
    
    def train_baseline_models(self, X_train, y_train, X_val, y_val):
        """Train baseline machine learning models."""
        if not self.config['models']['train_baseline']:
            print("\nSkipping baseline models (disabled in config)")
            return None
        
        print("\n" + "="*70)
        print("STEP 2: TRAINING BASELINE MODELS")
        print("="*70)
        
        baseline = BaselineModels()
        
        # Train specified models
        for model_name in self.config['models']['baseline_models']:
            if model_name in baseline.models:
                try:
                    result = baseline.train_model(model_name, X_train, y_train, X_val, y_val)
                    self.models[model_name] = result['model']
                    self.results[model_name] = result
                except Exception as e:
                    print(f"  ❌ Error training {model_name}: {str(e)}")
        
        # Compare models
        comparison = baseline.compare_models()
        if not comparison.empty:
            print("\n" + "-"*70)
            print("Baseline Models Comparison:")
            print(comparison.to_string())
        
        # Save models
        baseline.save_models(self.config['output']['models_dir'] + '/baseline')
        
        self.pipeline_report['baseline_models'] = {
            'models_trained': list(baseline.trained_models.keys()),
            'best_model': comparison.iloc[0]['Model'] if not comparison.empty else None,
            'comparison': comparison.to_dict() if not comparison.empty else {}
        }
        
        return baseline
    
    def train_deep_learning_models(self, X_train, y_train, X_val, y_val):
        """Train deep learning models."""
        if not self.config['models']['train_deep_learning']:
            print("\nSkipping deep learning models (disabled in config)")
            return None
        
        print("\n" + "="*70)
        print("STEP 3: TRAINING DEEP LEARNING MODELS")
        print("="*70)
        
        # Note: For actual text data, you would need to load the original text
        # Here we'll work with the preprocessed features
        print("  Note: Using preprocessed features for DL models")
        print("  For better performance, use raw text with proper DL preprocessing")
        
        if not DL_AVAILABLE:
            print("  ❌ Deep learning models not available - TensorFlow not installed")
            return None
            
        dl_models = DeepLearningModels()
        
        # For demonstration, we'll train simplified versions
        # In production, you'd use the actual text data
        
        trained_dl_models = {}
        
        # Train LSTM-like model on features
        if 'lstm' in self.config['models']['dl_models']:
            try:
                from tensorflow.keras import models, layers
                
                # Build a simple neural network for the features
                model = models.Sequential([
                    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
                    layers.Dropout(0.5),
                    layers.Dense(128, activation='relu'),
                    layers.Dropout(0.5),
                    layers.Dense(64, activation='relu'),
                    layers.Dropout(0.5),
                    layers.Dense(1, activation='sigmoid')
                ])
                
                model.compile(optimizer='adam', loss='binary_crossentropy', 
                            metrics=['accuracy'])
                
                print("\nTraining Neural Network (LSTM substitute)...")
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=min(self.config['training']['epochs_dl'], 5),
                    batch_size=self.config['training']['batch_size'],
                    verbose=1
                )
                
                # Evaluate
                val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
                print(f"  Validation Accuracy: {val_acc:.4f}")
                
                trained_dl_models['neural_network'] = model
                self.models['neural_network'] = model
                
            except Exception as e:
                print(f"  ❌ Error training neural network: {str(e)}")
        
        self.pipeline_report['deep_learning'] = {
            'models_trained': list(trained_dl_models.keys()),
            'training_epochs': self.config['training']['epochs_dl']
        }
        
        return trained_dl_models
    
    def train_ensemble_models(self, X_train, y_train, X_val, y_val, base_models):
        """Train ensemble models."""
        if not self.config['models']['train_ensemble']:
            print("\nSkipping ensemble models (disabled in config)")
            return None
        
        print("\n" + "="*70)
        print("STEP 4: TRAINING ENSEMBLE MODELS")
        print("="*70)
        
        ensemble_results = {}
        
        for ensemble_method in self.config['models']['ensemble_methods']:
            try:
                print(f"\nTraining {ensemble_method} ensemble...")
                
                # Filter out None models
                valid_base_models = {k: v for k, v in base_models.items() 
                                   if v is not None}
                
                if len(valid_base_models) < 2:
                    print(f"  ⚠ Not enough base models for ensemble (need at least 2)")
                    continue
                
                ensemble = EnsembleModel(valid_base_models, 
                                       ensemble_method=ensemble_method)
                
                # Train ensemble
                if ensemble_method == 'weighted_voting':
                    # Optimize weights first
                    weights = ensemble.optimize_weights(X_val, y_val, n_trials=20)
                    results = ensemble.train(X_train, y_train, X_val, y_val, weights)
                else:
                    results = ensemble.train(X_train, y_train, X_val, y_val)
                
                ensemble_results[ensemble_method] = results
                self.models[f'ensemble_{ensemble_method}'] = ensemble
                
                # Save ensemble
                ensemble.save_ensemble(f"{self.config['output']['models_dir']}/ensemble")
                
            except Exception as e:
                print(f"  ❌ Error training {ensemble_method} ensemble: {str(e)}")
        
        self.pipeline_report['ensemble_models'] = {
            'methods_trained': list(ensemble_results.keys()),
            'base_models_used': list(valid_base_models.keys()) if 'valid_base_models' in locals() else []
        }
        
        return ensemble_results
    
    def perform_cross_validation(self, X_train, y_train):
        """Perform cross-validation on all models."""
        print("\n" + "="*70)
        print("STEP 5: CROSS-VALIDATION EVALUATION")
        print("="*70)
        
        cv_framework = CrossValidationFramework(
            cv_strategy=self.config['training']['cv_strategy'],
            n_folds=self.config['training']['cv_folds']
        )
        
        # Evaluate a subset of models for CV
        cv_models = {}
        
        # Add baseline models for CV
        if 'logistic_regression' in self.models:
            cv_models['LogisticRegression'] = self.models['logistic_regression']
        if 'random_forest' in self.models:
            cv_models['RandomForest'] = self.models['random_forest']
        if 'xgboost' in self.models:
            cv_models['XGBoost'] = self.models['xgboost']
        
        if cv_models:
            comparison = cv_framework.evaluate_multiple_models(
                cv_models, X_train, y_train, 
                scoring=self.config['evaluation']['metrics']
            )
            
            print("\n" + "-"*70)
            print("Cross-Validation Results:")
            print(comparison.to_string())
            
            # Generate CV report
            cv_framework.generate_cv_report(
                f"{self.config['output']['reports_dir']}/cv_results.json"
            )
            
            self.pipeline_report['cross_validation'] = {
                'n_folds': self.config['training']['cv_folds'],
                'models_evaluated': list(cv_models.keys()),
                'results': comparison.to_dict() if not comparison.empty else {}
            }
        else:
            print("  ⚠ No models available for cross-validation")
    
    def evaluate_on_test_set(self, X_test, y_test):
        """Final evaluation on test set."""
        print("\n" + "="*70)
        print("STEP 6: FINAL TEST SET EVALUATION")
        print("="*70)
        
        test_results = {}
        
        for model_name, model in self.models.items():
            if model is not None:
                try:
                    # Make predictions
                    if hasattr(model, 'predict'):
                        y_pred = model.predict(X_test)
                        
                        # Calculate metrics
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                        
                        metrics = {
                            'accuracy': accuracy_score(y_test, y_pred),
                            'precision': precision_score(y_test, y_pred, average='binary'),
                            'recall': recall_score(y_test, y_pred, average='binary'),
                            'f1': f1_score(y_test, y_pred, average='binary')
                        }
                        
                        test_results[model_name] = metrics
                        
                        print(f"\n{model_name}:")
                        for metric, value in metrics.items():
                            print(f"  {metric}: {value:.4f}")
                        
                except Exception as e:
                    print(f"  ❌ Error evaluating {model_name}: {str(e)}")
        
        # Find best model
        if test_results:
            best_model = max(test_results.items(), key=lambda x: x[1]['f1'])
            print(f"\n✨ Best Model: {best_model[0]} (F1: {best_model[1]['f1']:.4f})")
        
        self.pipeline_report['test_evaluation'] = test_results
        
        return test_results
    
    def generate_final_report(self):
        """Generate comprehensive pipeline report."""
        print("\n" + "="*70)
        print("GENERATING FINAL REPORT")
        print("="*70)
        
        # Add timing information
        if self.start_time:
            self.pipeline_report['execution_time'] = time.time() - self.start_time
        
        self.pipeline_report['timestamp'] = datetime.now().isoformat()
        self.pipeline_report['config'] = self.config
        
        # Save report
        report_path = Path(self.config['output']['reports_dir']) / 'phase3_pipeline_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.pipeline_report, f, indent=2, default=str)
        
        print(f"✓ Pipeline report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        
        if 'data_loaded' in self.pipeline_report:
            print(f"\nData:")
            for key, value in self.pipeline_report['data_loaded'].items():
                print(f"  {key}: {value}")
        
        if 'baseline_models' in self.pipeline_report:
            print(f"\nBaseline Models:")
            print(f"  Trained: {len(self.pipeline_report['baseline_models']['models_trained'])}")
            if self.pipeline_report['baseline_models']['best_model']:
                print(f"  Best: {self.pipeline_report['baseline_models']['best_model']}")
        
        if 'ensemble_models' in self.pipeline_report:
            print(f"\nEnsemble Models:")
            print(f"  Methods: {self.pipeline_report['ensemble_models']['methods_trained']}")
        
        if 'test_evaluation' in self.pipeline_report:
            print(f"\nTest Set Performance:")
            best_model = max(self.pipeline_report['test_evaluation'].items(), 
                           key=lambda x: x[1]['f1'])
            print(f"  Best Model: {best_model[0]}")
            print(f"  Best F1: {best_model[1]['f1']:.4f}")
        
        if 'execution_time' in self.pipeline_report:
            print(f"\nExecution Time: {self.pipeline_report['execution_time']:.2f} seconds")
    
    def run(self):
        """Execute the complete Phase 3 pipeline."""
        self.start_time = time.time()
        
        print("\n" + "="*70)
        print("PHASE 3 PIPELINE: MODEL DEVELOPMENT AND INITIAL TESTING")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Load data
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
            
            if X_train is None:
                print("\n❌ Failed to load data. Exiting pipeline.")
                return
            
            # Step 2: Train baseline models
            baseline_models = self.train_baseline_models(X_train, y_train, X_val, y_val)
            
            # Step 3: Train deep learning models
            dl_models = self.train_deep_learning_models(X_train, y_train, X_val, y_val)
            
            # Step 4: Train ensemble models
            # Combine baseline and DL models for ensemble
            all_base_models = {}
            if baseline_models:
                all_base_models.update(baseline_models.trained_models)
            if dl_models:
                all_base_models.update(dl_models)
            
            ensemble_results = self.train_ensemble_models(
                X_train, y_train, X_val, y_val, all_base_models
            )
            
            # Step 5: Cross-validation
            self.perform_cross_validation(X_train, y_train)
            
            # Step 6: Final test evaluation
            test_results = self.evaluate_on_test_set(X_test, y_test)
            
            # Step 7: Generate report
            self.generate_final_report()
            
            print("\n" + "="*70)
            print("✅ PHASE 3 PIPELINE COMPLETED SUCCESSFULLY")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            self.pipeline_report['error'] = str(e)
            self.generate_final_report()


def main():
    """Main entry point for Phase 3 pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 3: Model Development and Initial Testing')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--use-sample', action='store_true', 
                       help='Use sample of data for quick testing')
    parser.add_argument('--skip-dl', action='store_true',
                       help='Skip deep learning models')
    parser.add_argument('--skip-ensemble', action='store_true',
                       help='Skip ensemble models')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = Phase3Pipeline(args.config)
    
    # Override config with command line arguments
    if args.use_sample:
        pipeline.config['data']['use_sample'] = True
        pipeline.config['data']['sample_size'] = 1000
    if args.skip_dl:
        pipeline.config['models']['train_deep_learning'] = False
    if args.skip_ensemble:
        pipeline.config['models']['train_ensemble'] = False
    
    # Run pipeline
    pipeline.run()


if __name__ == "__main__":
    main()