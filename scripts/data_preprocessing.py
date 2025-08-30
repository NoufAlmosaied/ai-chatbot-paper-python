#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for Phishing Detection
Phase 2: Data Annotation and Preprocessing
"""

import pandas as pd
import numpy as np
import re
import html
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle
import json
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Complete data preprocessing pipeline for phishing detection."""
    
    def __init__(self, config: Dict = None):
        """Initialize preprocessor with configuration."""
        self.config = config or self._default_config()
        self.tfidf_vectorizer = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = []
        
    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'test_size': 0.15,
            'val_size': 0.15,
            'random_state': 42,
            'tfidf_max_features': 5000,
            'tfidf_ngram_range': (1, 3),
            'tfidf_min_df': 5,
            'tfidf_max_df': 0.95,
            'apply_smote': True,
            'normalize_features': True
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if not text or pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                     ' URL ', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' EMAIL ', text)
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def extract_email_components(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Extract email components (subject, body, headers) if available."""
        df['cleaned_text'] = df[text_column].apply(self.clean_text)
        
        # Try to extract subject if pattern exists
        df['has_subject'] = df[text_column].str.contains('subject:', case=False, na=False).astype(int)
        
        # Extract first line as potential subject
        df['first_line'] = df['cleaned_text'].str.split('\n').str[0].str[:100]
        
        # Calculate text statistics
        df['text_length'] = df['cleaned_text'].str.len()
        df['word_count'] = df['cleaned_text'].str.split().str.len()
        df['avg_word_length'] = df.apply(
            lambda x: np.mean([len(word) for word in x['cleaned_text'].split()]) 
            if x['word_count'] > 0 else 0, axis=1
        )
        
        return df
    
    def create_tfidf_features(self, texts: pd.Series, fit: bool = True) -> np.ndarray:
        """Create TF-IDF features from text."""
        if fit:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config['tfidf_max_features'],
                ngram_range=self.config['tfidf_ngram_range'],
                min_df=self.config['tfidf_min_df'],
                max_df=self.config['tfidf_max_df'],
                stop_words='english'
            )
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        else:
            if self.tfidf_vectorizer is None:
                raise ValueError("TF-IDF vectorizer not fitted yet")
            tfidf_features = self.tfidf_vectorizer.transform(texts).toarray()
        
        return tfidf_features
    
    def extract_url_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract URL-based features from the dataset."""
        # These features are already in the Phishing_Legitimate_full.csv dataset
        url_features = [
            'NumDots', 'SubdomainLevel', 'PathLevel', 'UrlLength', 'NumDash',
            'NumDashInHostname', 'AtSymbol', 'TildeSymbol', 'NumUnderscore',
            'NumPercent', 'NumQueryComponents', 'NumAmpersand', 'NumHash',
            'NumNumericChars', 'NoHttps', 'RandomString', 'IpAddress',
            'DomainInSubdomains', 'DomainInPaths', 'HttpsInHostname',
            'HostnameLength', 'PathLength', 'QueryLength', 'DoubleSlashInPath'
        ]
        
        # Check which features are available
        available_features = [f for f in url_features if f in df.columns]
        
        if available_features:
            print(f"  Found {len(available_features)} URL features in dataset")
            return df[available_features]
        else:
            print("  No URL features found, creating empty dataframe")
            return pd.DataFrame()
    
    def combine_features(self, text_features: np.ndarray, url_features: pd.DataFrame,
                        metadata_features: pd.DataFrame) -> np.ndarray:
        """Combine all feature types into single feature matrix."""
        features_list = []
        
        # Add text features (TF-IDF)
        if text_features is not None and text_features.size > 0:
            features_list.append(text_features)
            print(f"  Added {text_features.shape[1]} TF-IDF features")
        
        # Add URL features
        if not url_features.empty:
            features_list.append(url_features.values)
            print(f"  Added {url_features.shape[1]} URL features")
        
        # Add metadata features
        if not metadata_features.empty:
            features_list.append(metadata_features.values)
            print(f"  Added {metadata_features.shape[1]} metadata features")
        
        # Combine all features
        if features_list:
            combined = np.hstack(features_list)
            print(f"  Total features: {combined.shape[1]}")
            return combined
        else:
            raise ValueError("No features to combine")
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """Split data into train, validation, and test sets."""
        # First split: train+val and test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        # Second split: train and val
        val_size_adjusted = self.config['val_size'] / (1 - self.config['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config['random_state'],
            stratify=y_temp
        )
        
        print(f"\nData split completed:")
        print(f"  Training: {X_train.shape[0]} samples ({100*len(X_train)/len(X):.1f}%)")
        print(f"  Validation: {X_val.shape[0]} samples ({100*len(X_val)/len(X):.1f}%)")
        print(f"  Test: {X_test.shape[0]} samples ({100*len(X_test)/len(X):.1f}%)")
        
        # Check class distribution
        for name, labels in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            unique, counts = np.unique(labels, return_counts=True)
            print(f"  {name} distribution: {dict(zip(unique, counts))}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def apply_smote(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple:
        """Apply SMOTE for handling class imbalance."""
        print("\nApplying SMOTE for class balancing...")
        
        # Check current distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"  Before SMOTE: {dict(zip(unique, counts))}")
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.config['random_state'])
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        # Check new distribution
        unique, counts = np.unique(y_resampled, return_counts=True)
        print(f"  After SMOTE: {dict(zip(unique, counts))}")
        
        return X_resampled, y_resampled
    
    def normalize_features(self, X_train: np.ndarray, X_val: np.ndarray, 
                          X_test: np.ndarray) -> Tuple:
        """Normalize features using StandardScaler."""
        print("\nNormalizing features...")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("  ✓ Features normalized")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def process_dataset(self, df: pd.DataFrame, text_column: str = 'text_combined',
                       label_column: str = 'label') -> Dict:
        """Process entire dataset through the preprocessing pipeline."""
        print("\n" + "="*70)
        print("DATA PREPROCESSING PIPELINE")
        print("="*70)
        
        # Step 1: Clean and extract text components
        print("\n1. Cleaning text data...")
        df = self.extract_email_components(df, text_column)
        print(f"  ✓ Cleaned {len(df)} samples")
        
        # Step 2: Create TF-IDF features
        print("\n2. Creating TF-IDF features...")
        tfidf_features = self.create_tfidf_features(df['cleaned_text'], fit=True)
        print(f"  ✓ Created {tfidf_features.shape[1]} TF-IDF features")
        
        # Step 3: Extract URL features (if available)
        print("\n3. Extracting URL features...")
        url_features = self.extract_url_features(df)
        
        # Step 4: Create metadata features
        print("\n4. Creating metadata features...")
        metadata_features = df[['text_length', 'word_count', 'avg_word_length', 'has_subject']]
        print(f"  ✓ Created {metadata_features.shape[1]} metadata features")
        
        # Step 5: Combine all features
        print("\n5. Combining all features...")
        X = self.combine_features(tfidf_features, url_features, metadata_features)
        
        # Step 6: Encode labels
        print("\n6. Encoding labels...")
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df[label_column])
        print(f"  Classes: {self.label_encoder.classes_}")
        
        # Step 7: Split data
        print("\n7. Splitting data...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Step 8: Apply SMOTE (if configured)
        if self.config['apply_smote']:
            X_train, y_train = self.apply_smote(X_train, y_train)
        
        # Step 9: Normalize features (if configured)
        if self.config['normalize_features']:
            X_train, X_val, X_test = self.normalize_features(X_train, X_val, X_test)
        
        # Store feature names
        self.feature_names = (
            [f'tfidf_{i}' for i in range(tfidf_features.shape[1])] +
            list(url_features.columns) if not url_features.empty else [] +
            list(metadata_features.columns)
        )
        
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE")
        print("="*70)
        print(f"Final dataset shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val: {X_val.shape}")
        print(f"  X_test: {X_test.shape}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': self.feature_names,
            'label_encoder': self.label_encoder,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'scaler': self.scaler
        }
    
    def save_processed_data(self, data: Dict, output_dir: str = 'data/processed'):
        """Save processed data and preprocessing objects."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nSaving processed data to {output_dir}...")
        
        # Save numpy arrays
        np.save(output_path / 'X_train.npy', data['X_train'])
        np.save(output_path / 'X_val.npy', data['X_val'])
        np.save(output_path / 'X_test.npy', data['X_test'])
        np.save(output_path / 'y_train.npy', data['y_train'])
        np.save(output_path / 'y_val.npy', data['y_val'])
        np.save(output_path / 'y_test.npy', data['y_test'])
        
        # Save preprocessing objects
        with open(output_path / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(data['label_encoder'], f)
        
        with open(output_path / 'tfidf_vectorizer.pkl', 'wb') as f:
            pickle.dump(data['tfidf_vectorizer'], f)
        
        with open(output_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(data['scaler'], f)
        
        # Save feature names
        with open(output_path / 'feature_names.json', 'w') as f:
            json.dump(data['feature_names'], f)
        
        # Save configuration
        with open(output_path / 'preprocessing_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print("  ✓ All files saved successfully")


def main():
    """Test the preprocessing pipeline."""
    print("\n" + "="*70)
    print("DATA PREPROCESSING TEST")
    print("="*70)
    
    # Load sample data
    data_path = Path('data/raw/phishing_email.csv')
    
    if data_path.exists():
        print(f"\nLoading data from {data_path}...")
        df = pd.read_csv(data_path, nrows=1000)  # Use subset for testing
        print(f"  Loaded {len(df)} samples")
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Process dataset
        processed_data = preprocessor.process_dataset(df)
        
        # Save processed data
        preprocessor.save_processed_data(processed_data)
        
    else:
        print(f"❌ Data file not found: {data_path}")
        print("   Please ensure datasets are downloaded first")


if __name__ == "__main__":
    main()