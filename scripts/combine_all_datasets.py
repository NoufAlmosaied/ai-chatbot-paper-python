#!/usr/bin/env python3
"""
Combine all phishing datasets into a unified format for training.
This script merges all available CSV files with proper labeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_and_standardize_datasets():
    """Load all CSV files and standardize their format."""
    
    print("\n" + "="*70)
    print("LOADING AND COMBINING ALL PHISHING DATASETS")
    print("="*70)
    
    data_dir = Path('data/raw')
    combined_data = []
    
    # 1. Load phishing_email.csv (already combined, just text and label)
    print("\n1. Loading phishing_email.csv...")
    df1 = pd.read_csv(data_dir / 'phishing_email.csv')
    print(f"   Shape: {df1.shape}")
    print(f"   Columns: {df1.columns.tolist()}")
    # This already has text_combined and label
    combined_data.append(df1[['text_combined', 'label']])
    
    # 2. Load CEAS_08.csv (has sender, receiver, date, subject, body, label, urls)
    print("\n2. Loading CEAS_08.csv...")
    df2 = pd.read_csv(data_dir / 'CEAS_08.csv')
    print(f"   Shape: {df2.shape}")
    print(f"   Columns: {df2.columns.tolist()}")
    # Combine subject and body into text_combined
    df2['text_combined'] = df2['subject'].fillna('') + ' ' + df2['body'].fillna('')
    combined_data.append(df2[['text_combined', 'label']])
    
    # 3. Load Enron.csv (has subject, body, label)
    print("\n3. Loading Enron.csv...")
    df3 = pd.read_csv(data_dir / 'Enron.csv')
    print(f"   Shape: {df3.shape}")
    print(f"   Columns: {df3.columns.tolist()}")
    # Combine subject and body
    df3['text_combined'] = df3['subject'].fillna('') + ' ' + df3['body'].fillna('')
    combined_data.append(df3[['text_combined', 'label']])
    
    # 4. Load Nigerian_Fraud.csv (has sender, receiver, date, subject, body, urls, label)
    print("\n4. Loading Nigerian_Fraud.csv...")
    df4 = pd.read_csv(data_dir / 'Nigerian_Fraud.csv')
    print(f"   Shape: {df4.shape}")
    print(f"   Columns: {df4.columns.tolist()}")
    # Combine subject and body
    df4['text_combined'] = df4['subject'].fillna('') + ' ' + df4['body'].fillna('')
    combined_data.append(df4[['text_combined', 'label']])
    
    # 5. Load SpamAssasin.csv (need to check structure)
    print("\n5. Loading SpamAssasin.csv...")
    try:
        df5 = pd.read_csv(data_dir / 'SpamAssasin.csv')
        print(f"   Shape: {df5.shape}")
        print(f"   Columns: {df5.columns.tolist()}")
        
        # Check if it has the expected columns
        if 'subject' in df5.columns and 'body' in df5.columns:
            df5['text_combined'] = df5['subject'].fillna('') + ' ' + df5['body'].fillna('')
        elif 'text' in df5.columns:
            df5['text_combined'] = df5['text']
        else:
            # If structure is different, try to combine all text columns
            text_cols = [col for col in df5.columns if col not in ['label', 'urls', 'sender', 'receiver', 'date']]
            df5['text_combined'] = df5[text_cols].fillna('').apply(' '.join, axis=1)
        
        if 'label' in df5.columns:
            combined_data.append(df5[['text_combined', 'label']])
    except Exception as e:
        print(f"   Error loading SpamAssasin.csv: {e}")
    
    # 6. Load Ling.csv
    print("\n6. Loading Ling.csv...")
    try:
        df6 = pd.read_csv(data_dir / 'Ling.csv')
        print(f"   Shape: {df6.shape}")
        print(f"   Columns: {df6.columns.tolist()}")
        
        # Combine available text columns
        if 'subject' in df6.columns and 'body' in df6.columns:
            df6['text_combined'] = df6['subject'].fillna('') + ' ' + df6['body'].fillna('')
        elif 'text' in df6.columns:
            df6['text_combined'] = df6['text']
        else:
            text_cols = [col for col in df6.columns if col not in ['label']]
            df6['text_combined'] = df6[text_cols].fillna('').apply(' '.join, axis=1)
        
        if 'label' in df6.columns:
            combined_data.append(df6[['text_combined', 'label']])
    except Exception as e:
        print(f"   Error loading Ling.csv: {e}")
    
    # 7. Load Nazario.csv
    print("\n7. Loading Nazario.csv...")
    try:
        df7 = pd.read_csv(data_dir / 'Nazario.csv')
        print(f"   Shape: {df7.shape}")
        print(f"   Columns: {df7.columns.tolist()}")
        
        # Combine available text columns
        if 'subject' in df7.columns and 'body' in df7.columns:
            df7['text_combined'] = df7['subject'].fillna('') + ' ' + df7['body'].fillna('')
        elif 'text' in df7.columns:
            df7['text_combined'] = df7['text']
        else:
            text_cols = [col for col in df7.columns if col not in ['label', 'urls', 'sender', 'receiver', 'date']]
            df7['text_combined'] = df7[text_cols].fillna('').apply(' '.join, axis=1)
        
        if 'label' in df7.columns:
            combined_data.append(df7[['text_combined', 'label']])
    except Exception as e:
        print(f"   Error loading Nazario.csv: {e}")
    
    # Combine all datasets
    print("\n" + "="*70)
    print("COMBINING ALL DATASETS")
    print("="*70)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Remove duplicates based on text content
        print(f"\nBefore deduplication: {len(combined_df)} samples")
        combined_df = combined_df.drop_duplicates(subset=['text_combined'], keep='first')
        print(f"After deduplication: {len(combined_df)} samples")
        
        # Check label distribution
        print("\nLabel Distribution:")
        print(combined_df['label'].value_counts())
        print(f"\nPercentage distribution:")
        print(combined_df['label'].value_counts(normalize=True) * 100)
        
        return combined_df
    else:
        print("No data to combine!")
        return pd.DataFrame()


def analyze_combined_dataset(df):
    """Analyze the combined dataset."""
    print("\n" + "="*70)
    print("COMBINED DATASET ANALYSIS")
    print("="*70)
    
    print(f"\nTotal Samples: {len(df)}")
    print(f"Features: {df.columns.tolist()}")
    
    # Text length statistics
    df['text_length'] = df['text_combined'].str.len()
    print(f"\nText Length Statistics:")
    print(f"  Mean: {df['text_length'].mean():.0f} characters")
    print(f"  Median: {df['text_length'].median():.0f} characters")
    print(f"  Min: {df['text_length'].min():.0f} characters")
    print(f"  Max: {df['text_length'].max():.0f} characters")
    
    # Check for empty texts
    empty_texts = df['text_combined'].isna().sum() + (df['text_combined'].str.strip() == '').sum()
    print(f"\nEmpty texts: {empty_texts} ({100*empty_texts/len(df):.2f}%)")
    
    # Missing labels
    missing_labels = df['label'].isna().sum()
    print(f"Missing labels: {missing_labels} ({100*missing_labels/len(df):.2f}%)")
    
    return df


def save_combined_dataset(df):
    """Save the combined dataset."""
    output_path = Path('data/processed/combined_email_dataset.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove any rows with missing labels
    df_clean = df.dropna(subset=['label'])
    
    # Remove empty texts
    df_clean = df_clean[df_clean['text_combined'].str.strip() != '']
    
    print(f"\nSaving {len(df_clean)} samples to {output_path}")
    df_clean.to_csv(output_path, index=False)
    
    # Also save a sample for quick testing
    sample_path = Path('data/processed/combined_email_sample.csv')
    df_sample = df_clean.sample(n=min(1000, len(df_clean)), random_state=42)
    df_sample.to_csv(sample_path, index=False)
    print(f"Sample of 1000 saved to {sample_path}")
    
    return df_clean


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("PHISHING DATASET COMBINATION SCRIPT")
    print("="*70)
    
    # Load and combine all datasets
    combined_df = load_and_standardize_datasets()
    
    if not combined_df.empty:
        # Analyze the combined dataset
        combined_df = analyze_combined_dataset(combined_df)
        
        # Save the combined dataset
        final_df = save_combined_dataset(combined_df)
        
        print("\n" + "="*70)
        print("✅ DATASET COMBINATION COMPLETE")
        print("="*70)
        print(f"\nFinal Statistics:")
        print(f"  Total unique samples: {len(final_df)}")
        print(f"  Phishing samples: {(final_df['label'] == 1).sum()}")
        print(f"  Legitimate samples: {(final_df['label'] == 0).sum()}")
        print(f"  Balance ratio: {(final_df['label'] == 1).sum() / (final_df['label'] == 0).sum():.2f}:1")
    else:
        print("\n❌ Failed to combine datasets")


if __name__ == "__main__":
    main()