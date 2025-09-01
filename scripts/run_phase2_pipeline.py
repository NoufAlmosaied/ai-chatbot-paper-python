#!/usr/bin/env python3
"""
Main Pipeline Orchestrator for Phase 2: Data Annotation and Preprocessing
This script coordinates the entire preprocessing pipeline.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from automated_tagging import AutomatedTagger
from data_preprocessing import DataPreprocessor
from data_validation import DataValidator
from combine_all_datasets import load_and_standardize_datasets, save_combined_dataset


class Phase2Pipeline:
    """Orchestrates the complete Phase 2 pipeline."""
    
    def __init__(self, config_path: str = None):
        """Initialize pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.start_time = None
        self.pipeline_report = {}
        
    def _load_config(self, config_path: str = None) -> dict:
        """Load pipeline configuration."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default configuration
            return {
                'datasets': {
                    'email_dataset': 'data/processed/combined_email_dataset.csv',  # Use combined dataset
                    'url_dataset': 'data/raw/Phishing_Legitimate_full.csv',
                    'use_individual_datasets': False,  # Set to True to use individual files
                    'max_samples': None  # Use all samples
                },
                'tagging': {
                    'batch_size': 100,
                    'apply_tagging': True
                },
                'preprocessing': {
                    'test_size': 0.15,
                    'val_size': 0.15,
                    'apply_smote': True,
                    'normalize_features': True,
                    'tfidf_max_features': 5000,
                    'tfidf_ngram_range': (1, 3),
                    'tfidf_min_df': 5,
                    'tfidf_max_df': 0.95,
                    'random_state': 42
                },
                'validation': {
                    'generate_report': True,
                    'create_visualizations': True
                },
                'output': {
                    'processed_dir': 'data/processed',
                    'reports_dir': 'data/reports',
                    'save_intermediate': True
                }
            }
    
    def load_datasets(self) -> dict:
        """Load all required datasets."""
        print("\n" + "="*70)
        print("STEP 1: LOADING DATASETS")
        print("="*70)
        
        datasets = {}
        
        # Check if combined dataset exists, if not create it
        email_path = Path(self.config['datasets']['email_dataset'])
        if not email_path.exists():
            print(f"\nCombined dataset not found. Creating it now...")
            combined_df = load_and_standardize_datasets()
            if not combined_df.empty:
                save_combined_dataset(combined_df)
                print(f"  ✓ Combined dataset created successfully")
            else:
                print(f"  ❌ Failed to create combined dataset")
                return datasets
        
        # Load email dataset
        if email_path.exists():
            print(f"\nLoading email dataset from {email_path}...")
            df_email = pd.read_csv(email_path)
            
            if self.config['datasets']['max_samples']:
                df_email = df_email.head(self.config['datasets']['max_samples'])
            
            datasets['email'] = df_email
            print(f"  ✓ Loaded {len(df_email)} email samples")
        else:
            print(f"  ⚠ Email dataset not found: {email_path}")
        
        # Load URL dataset
        url_path = Path(self.config['datasets']['url_dataset'])
        if url_path.exists():
            print(f"\nLoading URL dataset from {url_path}...")
            df_url = pd.read_csv(url_path)
            
            if self.config['datasets']['max_samples']:
                df_url = df_url.head(self.config['datasets']['max_samples'])
            
            datasets['url'] = df_url
            print(f"  ✓ Loaded {len(df_url)} URL samples")
        else:
            print(f"  ⚠ URL dataset not found: {url_path}")
        
        self.pipeline_report['datasets_loaded'] = {
            'email_samples': len(datasets.get('email', [])),
            'url_samples': len(datasets.get('url', []))
        }
        
        return datasets
    
    def apply_automated_tagging(self, datasets: dict) -> dict:
        """Apply automated tagging to datasets."""
        if not self.config['tagging']['apply_tagging']:
            print("\nSkipping automated tagging (disabled in config)")
            return datasets
        
        print("\n" + "="*70)
        print("STEP 2: AUTOMATED TAGGING")
        print("="*70)
        
        tagger = AutomatedTagger()
        
        # Tag email dataset
        if 'email' in datasets:
            print("\nTagging email dataset...")
            df_email = datasets['email']
            
            # Apply tagging
            df_email = tagger.process_dataset(
                df_email, 
                text_column='text_combined',
                batch_size=self.config['tagging']['batch_size']
            )
            
            datasets['email'] = df_email
            
            # Calculate tagging statistics
            high_risk = (df_email['risk_level'] == 'high').sum()
            medium_risk = (df_email['risk_level'] == 'medium').sum()
            low_risk = (df_email['risk_level'] == 'low').sum()
            
            print(f"\n  Risk Distribution:")
            print(f"    High: {high_risk} ({100*high_risk/len(df_email):.1f}%)")
            print(f"    Medium: {medium_risk} ({100*medium_risk/len(df_email):.1f}%)")
            print(f"    Low: {low_risk} ({100*low_risk/len(df_email):.1f}%)")
            
            self.pipeline_report['tagging_stats'] = {
                'high_risk': high_risk,
                'medium_risk': medium_risk,
                'low_risk': low_risk
            }
        
        return datasets
    
    def preprocess_data(self, datasets: dict) -> dict:
        """Apply preprocessing pipeline."""
        print("\n" + "="*70)
        print("STEP 3: DATA PREPROCESSING")
        print("="*70)
        
        preprocessor = DataPreprocessor(self.config['preprocessing'])
        processed_data = {}
        
        # Process email dataset
        if 'email' in datasets:
            print("\nPreprocessing email dataset...")
            df_email = datasets['email']
            
            # Ensure label column exists
            if 'label' in df_email.columns:
                processed = preprocessor.process_dataset(
                    df_email,
                    text_column='text_combined',
                    label_column='label'
                )
                processed_data['email'] = processed
                
                # Save processed data
                if self.config['output']['save_intermediate']:
                    preprocessor.save_processed_data(
                        processed,
                        self.config['output']['processed_dir'] + '/email'
                    )
            else:
                print("  ❌ No label column found in email dataset")
        
        # Process URL dataset (if needed)
        if 'url' in datasets:
            print("\nURL dataset features already extracted")
            # URL features are already in the dataset
            processed_data['url_features'] = datasets['url']
        
        self.pipeline_report['preprocessing'] = {
            'datasets_processed': list(processed_data.keys()),
            'features_created': processed_data.get('email', {}).get('feature_names', [])[:10]
        }
        
        return processed_data
    
    def validate_data(self, datasets: dict, processed_data: dict):
        """Validate data quality."""
        if not self.config['validation']['generate_report']:
            print("\nSkipping data validation (disabled in config)")
            return
        
        print("\n" + "="*70)
        print("STEP 4: DATA VALIDATION")
        print("="*70)
        
        validator = DataValidator(self.config['output']['reports_dir'])
        
        # Validate original datasets
        for name, df in datasets.items():
            if isinstance(df, pd.DataFrame):
                print(f"\nValidating {name} dataset...")
                report = validator.generate_quality_report(df)
                
                self.pipeline_report[f'{name}_quality'] = {
                    'quality_score': report['quality_score'],
                    'total_samples': report['dataset_info']['total_samples'],
                    'missing_values': report['missing_values']['total_missing'],
                    'duplicates': report['duplicates']['total_duplicates']
                }
                
                if self.config['validation']['create_visualizations']:
                    validator.visualize_data_quality(df)
    
    def generate_pipeline_report(self):
        """Generate comprehensive pipeline report."""
        print("\n" + "="*70)
        print("STEP 5: GENERATING PIPELINE REPORT")
        print("="*70)
        
        # Add timing information
        if self.start_time:
            self.pipeline_report['execution_time'] = time.time() - self.start_time
        
        self.pipeline_report['timestamp'] = datetime.now().isoformat()
        self.pipeline_report['config'] = self.config
        
        # Save report
        report_path = Path(self.config['output']['reports_dir']) / 'phase2_pipeline_report.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(self.pipeline_report, f, indent=2, default=str)
        
        print(f"\n✓ Pipeline report saved to {report_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("PIPELINE SUMMARY")
        print("="*70)
        
        if 'datasets_loaded' in self.pipeline_report:
            print(f"\nDatasets Loaded:")
            for key, value in self.pipeline_report['datasets_loaded'].items():
                print(f"  - {key}: {value}")
        
        if 'tagging_stats' in self.pipeline_report:
            print(f"\nTagging Statistics:")
            for key, value in self.pipeline_report['tagging_stats'].items():
                print(f"  - {key}: {value}")
        
        if 'email_quality' in self.pipeline_report:
            print(f"\nData Quality Score: {self.pipeline_report['email_quality']['quality_score']:.1f}/100")
        
        if 'execution_time' in self.pipeline_report:
            print(f"\nTotal Execution Time: {self.pipeline_report['execution_time']:.2f} seconds")
    
    def run(self):
        """Execute the complete pipeline."""
        self.start_time = time.time()
        
        print("\n" + "="*70)
        print("PHASE 2 PIPELINE: DATA ANNOTATION AND PREPROCESSING")
        print("="*70)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Load datasets
            datasets = self.load_datasets()
            
            if not datasets:
                print("\n❌ No datasets loaded. Please check data paths.")
                return
            
            # Step 2: Apply automated tagging
            datasets = self.apply_automated_tagging(datasets)
            
            # Step 3: Preprocess data
            processed_data = self.preprocess_data(datasets)
            
            # Step 4: Validate data
            self.validate_data(datasets, processed_data)
            
            # Step 5: Generate report
            self.generate_pipeline_report()
            
            print("\n" + "="*70)
            print("✅ PHASE 2 PIPELINE COMPLETED SUCCESSFULLY")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            self.pipeline_report['error'] = str(e)
            self.generate_pipeline_report()


def main():
    """Main entry point for the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 2 Pipeline: Data Annotation and Preprocessing')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--max-samples', type=int, help='Maximum samples to process')
    parser.add_argument('--skip-tagging', action='store_true', help='Skip automated tagging')
    parser.add_argument('--skip-validation', action='store_true', help='Skip data validation')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = Phase2Pipeline(args.config)
    
    # Override config with command line arguments
    if args.max_samples:
        pipeline.config['datasets']['max_samples'] = args.max_samples
    if args.skip_tagging:
        pipeline.config['tagging']['apply_tagging'] = False
    if args.skip_validation:
        pipeline.config['validation']['generate_report'] = False
    
    # Run pipeline
    pipeline.run()


if __name__ == "__main__":
    main()