#!/usr/bin/env python3
"""
Cross-Validation Framework for Model Evaluation
Phase 3: Model Development and Initial Testing
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, cross_val_score, cross_validate,
    GridSearchCV, RandomizedSearchCV, cross_val_predict
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, make_scorer, confusion_matrix
)
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


class CrossValidationFramework:
    """Comprehensive cross-validation framework for model evaluation."""
    
    def __init__(self, cv_strategy: str = 'stratified', n_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize cross-validation framework.
        
        Args:
            cv_strategy: 'stratified', 'standard', or 'time_series'
            n_folds: Number of folds for cross-validation
            random_state: Random seed for reproducibility
        """
        self.cv_strategy = cv_strategy
        self.n_folds = n_folds
        self.random_state = random_state
        self.cv_results = {}
        self.best_params = {}
        
    def get_cv_splitter(self):
        """Get the appropriate cross-validation splitter."""
        if self.cv_strategy == 'stratified':
            return StratifiedKFold(n_splits=self.n_folds, 
                                  shuffle=True, 
                                  random_state=self.random_state)
        elif self.cv_strategy == 'standard':
            return KFold(n_splits=self.n_folds, 
                        shuffle=True, 
                        random_state=self.random_state)
        else:
            raise ValueError(f"Unknown CV strategy: {self.cv_strategy}")
    
    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray,
                      scoring: List[str] = None) -> Dict:
        """
        Evaluate a model using cross-validation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            scoring: List of scoring metrics
        """
        if scoring is None:
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        cv_splitter = self.get_cv_splitter()
        
        print(f"\nEvaluating {model.__class__.__name__} with {self.n_folds}-fold CV...")
        
        # Perform cross-validation
        scores = cross_validate(model, X, y, cv=cv_splitter, 
                              scoring=scoring, 
                              return_train_score=True,
                              n_jobs=-1)
        
        # Calculate mean and std for each metric
        results = {}
        for metric in scoring:
            train_key = f'train_{metric}'
            test_key = f'test_{metric}'
            
            results[f'{metric}_train_mean'] = scores[train_key].mean()
            results[f'{metric}_train_std'] = scores[train_key].std()
            results[f'{metric}_test_mean'] = scores[test_key].mean()
            results[f'{metric}_test_std'] = scores[test_key].std()
            
            print(f"  {metric.upper()}:")
            print(f"    Train: {results[f'{metric}_train_mean']:.4f} ± {results[f'{metric}_train_std']:.4f}")
            print(f"    Test:  {results[f'{metric}_test_mean']:.4f} ± {results[f'{metric}_test_std']:.4f}")
        
        results['fold_scores'] = scores
        
        return results
    
    def evaluate_multiple_models(self, models: Dict[str, Any], 
                                X: np.ndarray, y: np.ndarray,
                                scoring: List[str] = None) -> pd.DataFrame:
        """
        Evaluate multiple models and compare their performance.
        
        Args:
            models: Dictionary of models to evaluate
            X: Features
            y: Labels
            scoring: List of scoring metrics
        """
        if scoring is None:
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        print("\n" + "="*70)
        print("CROSS-VALIDATION EVALUATION")
        print("="*70)
        
        all_results = {}
        for model_name, model in models.items():
            try:
                results = self.evaluate_model(model, X, y, scoring)
                all_results[model_name] = results
                self.cv_results[model_name] = results
            except Exception as e:
                print(f"  ❌ Error evaluating {model_name}: {str(e)}")
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, results in all_results.items():
            row = {'Model': model_name}
            for metric in scoring:
                row[f'{metric}_mean'] = results[f'{metric}_test_mean']
                row[f'{metric}_std'] = results[f'{metric}_test_std']
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by F1 score
        if 'f1_mean' in df.columns:
            df = df.sort_values('f1_mean', ascending=False)
        
        return df
    
    def hyperparameter_tuning(self, model: Any, param_grid: Dict,
                            X: np.ndarray, y: np.ndarray,
                            scoring: str = 'f1', search_type: str = 'grid') -> Dict:
        """
        Perform hyperparameter tuning using cross-validation.
        
        Args:
            model: Model to tune
            param_grid: Parameter grid for search
            X: Features
            y: Labels
            scoring: Scoring metric for optimization
            search_type: 'grid' or 'random'
        """
        cv_splitter = self.get_cv_splitter()
        
        print(f"\nPerforming {search_type} search for {model.__class__.__name__}...")
        print(f"  Scoring metric: {scoring}")
        print(f"  Parameter grid: {param_grid}")
        
        if search_type == 'grid':
            search = GridSearchCV(model, param_grid, cv=cv_splitter,
                                 scoring=scoring, n_jobs=-1, verbose=1)
        elif search_type == 'random':
            search = RandomizedSearchCV(model, param_grid, cv=cv_splitter,
                                      scoring=scoring, n_jobs=-1, 
                                      n_iter=20, verbose=1,
                                      random_state=self.random_state)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        # Perform search
        search.fit(X, y)
        
        # Store results
        results = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_model': search.best_estimator_,
            'cv_results': search.cv_results_
        }
        
        print(f"\n  Best parameters: {search.best_params_}")
        print(f"  Best CV score: {search.best_score_:.4f}")
        
        model_name = model.__class__.__name__
        self.best_params[model_name] = search.best_params_
        
        return results
    
    def plot_cv_scores(self, model_name: str = None, metric: str = 'f1'):
        """Plot cross-validation scores across folds."""
        if model_name and model_name in self.cv_results:
            results = self.cv_results[model_name]
        else:
            # Plot all models
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Collect data for all models
            train_scores = []
            test_scores = []
            model_names = []
            
            for name, results in self.cv_results.items():
                if f'fold_scores' in results:
                    fold_scores = results['fold_scores']
                    train_key = f'train_{metric}'
                    test_key = f'test_{metric}'
                    
                    if train_key in fold_scores and test_key in fold_scores:
                        train_scores.append(fold_scores[train_key])
                        test_scores.append(fold_scores[test_key])
                        model_names.append(name)
            
            if train_scores:
                # Plot training scores
                axes[0].boxplot(train_scores, labels=model_names)
                axes[0].set_title(f'Training {metric.upper()} Scores')
                axes[0].set_ylabel('Score')
                axes[0].tick_params(axis='x', rotation=45)
                axes[0].grid(True, alpha=0.3)
                
                # Plot test scores
                axes[1].boxplot(test_scores, labels=model_names)
                axes[1].set_title(f'Test {metric.upper()} Scores')
                axes[1].set_ylabel('Score')
                axes[1].tick_params(axis='x', rotation=45)
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                return fig
        
        return None
    
    def calculate_confidence_intervals(self, scores: np.ndarray, 
                                      confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence intervals for scores."""
        n = len(scores)
        mean = scores.mean()
        std = scores.std()
        
        # Calculate confidence interval
        from scipy import stats
        confidence_interval = stats.t.interval(confidence, n-1, 
                                              loc=mean, 
                                              scale=std/np.sqrt(n))
        
        return confidence_interval
    
    def perform_statistical_tests(self, model1_scores: np.ndarray,
                                 model2_scores: np.ndarray) -> Dict:
        """
        Perform statistical tests to compare two models.
        
        Args:
            model1_scores: CV scores for model 1
            model2_scores: CV scores for model 2
        """
        from scipy import stats
        
        # Paired t-test (since scores are from same folds)
        t_stat, p_value = stats.ttest_rel(model1_scores, model2_scores)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        w_stat, w_p_value = stats.wilcoxon(model1_scores, model2_scores)
        
        results = {
            'paired_t_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'wilcoxon_test': {
                'statistic': w_stat,
                'p_value': w_p_value,
                'significant': w_p_value < 0.05
            }
        }
        
        return results
    
    def generate_cv_report(self, output_path: str = 'reports/cv_results.json'):
        """Generate comprehensive cross-validation report."""
        report = {
            'cv_strategy': self.cv_strategy,
            'n_folds': self.n_folds,
            'model_results': {},
            'best_parameters': self.best_params
        }
        
        # Add model results
        for model_name, results in self.cv_results.items():
            model_report = {}
            for key, value in results.items():
                if key != 'fold_scores':  # Exclude raw fold scores from report
                    model_report[key] = value
            report['model_results'][model_name] = model_report
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n✓ CV report saved to {output_path}")
        
        return report
    
    def plot_learning_curves(self, model: Any, X: np.ndarray, y: np.ndarray,
                           train_sizes: np.ndarray = None):
        """Plot learning curves to diagnose bias/variance."""
        from sklearn.model_selection import learning_curve
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        cv_splitter = self.get_cv_splitter()
        
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv_splitter, train_sizes=train_sizes,
            scoring='f1', n_jobs=-1
        )
        
        # Calculate mean and std
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', color='blue', 
                label='Training score')
        plt.fill_between(train_sizes, train_mean - train_std,
                        train_mean + train_std, alpha=0.1, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red',
                label='Validation score')
        plt.fill_between(train_sizes, val_mean - val_std,
                        val_mean + val_std, alpha=0.1, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('F1 Score')
        plt.title(f'Learning Curves - {model.__class__.__name__}')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()


def main():
    """Test cross-validation framework."""
    print("\n" + "="*70)
    print("CROSS-VALIDATION FRAMEWORK TEST")
    print("="*70)
    
    # Generate dummy data
    np.random.seed(42)
    X = np.random.randn(500, 20)
    y = np.random.randint(0, 2, 500)
    
    # Initialize CV framework
    cv_framework = CrossValidationFramework(cv_strategy='stratified', n_folds=5)
    
    # Test with multiple models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=10, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Evaluate models
    comparison = cv_framework.evaluate_multiple_models(models, X, y)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(comparison.to_string())
    
    # Test hyperparameter tuning
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20]
    }
    
    rf_model = RandomForestClassifier(random_state=42)
    tuning_results = cv_framework.hyperparameter_tuning(
        rf_model, param_grid, X, y, search_type='grid'
    )
    
    # Generate report
    cv_framework.generate_cv_report()
    
    print("\n✓ Cross-validation framework test complete")


if __name__ == "__main__":
    main()