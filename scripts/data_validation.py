#!/usr/bin/env python3
"""
Data Validation and Quality Checks for Phishing Detection
Phase 2: Data Annotation and Preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataValidator:
    """Comprehensive data validation and quality checks."""
    
    def __init__(self, output_dir: str = 'data/reports'):
        """Initialize validator with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_report = {}
        
    def check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values in the dataset."""
        print("\n1. Checking Missing Values...")
        print("-" * 50)
        
        missing_stats = {
            'total_missing': df.isnull().sum().sum(),
            'columns_with_missing': {},
            'percentage_missing': {}
        }
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_stats['columns_with_missing'][col] = missing_count
                missing_stats['percentage_missing'][col] = (missing_count / len(df)) * 100
                print(f"  {col}: {missing_count} ({missing_stats['percentage_missing'][col]:.2f}%)")
        
        if missing_stats['total_missing'] == 0:
            print("  ✓ No missing values found")
        else:
            print(f"  ⚠ Total missing values: {missing_stats['total_missing']}")
        
        return missing_stats
    
    def check_duplicates(self, df: pd.DataFrame, subset: List[str] = None) -> Dict:
        """Check for duplicate records."""
        print("\n2. Checking Duplicates...")
        print("-" * 50)
        
        duplicate_stats = {
            'total_duplicates': 0,
            'duplicate_indices': [],
            'duplicate_percentage': 0
        }
        
        # Check for exact duplicates
        duplicates = df.duplicated(subset=subset, keep='first')
        duplicate_stats['total_duplicates'] = duplicates.sum()
        duplicate_stats['duplicate_indices'] = df[duplicates].index.tolist()
        duplicate_stats['duplicate_percentage'] = (duplicates.sum() / len(df)) * 100
        
        if duplicate_stats['total_duplicates'] == 0:
            print("  ✓ No duplicate records found")
        else:
            print(f"  ⚠ Found {duplicate_stats['total_duplicates']} duplicates " +
                  f"({duplicate_stats['duplicate_percentage']:.2f}%)")
        
        return duplicate_stats
    
    def check_label_distribution(self, df: pd.DataFrame, label_column: str) -> Dict:
        """Check label distribution and class balance."""
        print("\n3. Checking Label Distribution...")
        print("-" * 50)
        
        if label_column not in df.columns:
            print(f"  ❌ Label column '{label_column}' not found")
            return {}
        
        label_stats = {
            'value_counts': df[label_column].value_counts().to_dict(),
            'percentages': (df[label_column].value_counts(normalize=True) * 100).to_dict(),
            'unique_values': df[label_column].nunique(),
            'imbalance_ratio': None
        }
        
        # Calculate imbalance ratio
        counts = list(label_stats['value_counts'].values())
        if len(counts) == 2:
            label_stats['imbalance_ratio'] = max(counts) / min(counts)
        
        # Print distribution
        for label, count in label_stats['value_counts'].items():
            percentage = label_stats['percentages'][label]
            print(f"  Label {label}: {count} samples ({percentage:.2f}%)")
        
        if label_stats['imbalance_ratio']:
            print(f"  Imbalance ratio: {label_stats['imbalance_ratio']:.2f}:1")
            if label_stats['imbalance_ratio'] > 3:
                print("  ⚠ Significant class imbalance detected")
            else:
                print("  ✓ Classes are reasonably balanced")
        
        return label_stats
    
    def check_feature_statistics(self, df: pd.DataFrame, numeric_columns: List[str] = None) -> Dict:
        """Calculate statistics for numeric features."""
        print("\n4. Feature Statistics...")
        print("-" * 50)
        
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        feature_stats = {}
        
        for col in numeric_columns[:10]:  # Limit to first 10 for display
            stats = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'skew': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
            feature_stats[col] = stats
            
            print(f"  {col}:")
            print(f"    Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
            print(f"    Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        
        if len(numeric_columns) > 10:
            print(f"  ... and {len(numeric_columns) - 10} more features")
        
        return feature_stats
    
    def check_outliers(self, df: pd.DataFrame, numeric_columns: List[str] = None,
                       threshold: float = 3.0) -> Dict:
        """Detect outliers using Z-score method."""
        print("\n5. Checking Outliers...")
        print("-" * 50)
        
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        outlier_stats = {
            'total_outliers': 0,
            'columns_with_outliers': {},
            'outlier_percentage': {}
        }
        
        for col in numeric_columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outliers = z_scores > threshold
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                outlier_stats['columns_with_outliers'][col] = outlier_count
                outlier_stats['outlier_percentage'][col] = (outlier_count / len(df)) * 100
                outlier_stats['total_outliers'] += outlier_count
        
        # Print summary
        if outlier_stats['total_outliers'] == 0:
            print(f"  ✓ No outliers detected (Z-score > {threshold})")
        else:
            print(f"  ⚠ Total outliers: {outlier_stats['total_outliers']}")
            for col, count in list(outlier_stats['columns_with_outliers'].items())[:5]:
                print(f"    {col}: {count} outliers ({outlier_stats['outlier_percentage'][col]:.2f}%)")
        
        return outlier_stats
    
    def check_feature_correlation(self, df: pd.DataFrame, numeric_columns: List[str] = None,
                                 threshold: float = 0.9) -> Dict:
        """Check for highly correlated features."""
        print("\n6. Checking Feature Correlation...")
        print("-" * 50)
        
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            print("  ⚠ Not enough numeric columns for correlation analysis")
            return {}
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_columns].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        correlation_stats = {
            'highly_correlated_pairs': high_corr_pairs,
            'total_pairs': len(high_corr_pairs),
            'threshold': threshold
        }
        
        if len(high_corr_pairs) == 0:
            print(f"  ✓ No highly correlated features (|r| > {threshold})")
        else:
            print(f"  ⚠ Found {len(high_corr_pairs)} highly correlated feature pairs:")
            for pair in high_corr_pairs[:5]:
                print(f"    {pair['feature1']} <-> {pair['feature2']}: {pair['correlation']:.3f}")
        
        return correlation_stats
    
    def check_data_consistency(self, df: pd.DataFrame) -> Dict:
        """Check for data consistency issues."""
        print("\n7. Checking Data Consistency...")
        print("-" * 50)
        
        consistency_issues = {
            'negative_values': {},
            'invalid_ranges': {},
            'type_mismatches': {}
        }
        
        # Check for negative values in columns that shouldn't have them
        for col in df.select_dtypes(include=[np.number]).columns:
            if 'count' in col.lower() or 'length' in col.lower() or 'num' in col.lower():
                neg_count = (df[col] < 0).sum()
                if neg_count > 0:
                    consistency_issues['negative_values'][col] = neg_count
        
        # Check for values outside expected ranges (0-1 for probabilities, scores)
        for col in df.columns:
            if 'score' in col.lower() or 'probability' in col.lower() or 'confidence' in col.lower():
                if col in df.select_dtypes(include=[np.number]).columns:
                    out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
                    if out_of_range > 0:
                        consistency_issues['invalid_ranges'][col] = out_of_range
        
        # Print findings
        total_issues = (len(consistency_issues['negative_values']) + 
                       len(consistency_issues['invalid_ranges']))
        
        if total_issues == 0:
            print("  ✓ No consistency issues found")
        else:
            print(f"  ⚠ Found {total_issues} consistency issues:")
            for col, count in consistency_issues['negative_values'].items():
                print(f"    {col}: {count} negative values")
            for col, count in consistency_issues['invalid_ranges'].items():
                print(f"    {col}: {count} values out of range")
        
        return consistency_issues
    
    def generate_quality_report(self, df: pd.DataFrame, label_column: str = 'label') -> Dict:
        """Generate comprehensive data quality report."""
        print("\n" + "="*70)
        print("DATA QUALITY VALIDATION REPORT")
        print("="*70)
        
        report = {
            'dataset_info': {
                'total_samples': len(df),
                'total_features': len(df.columns),
                'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
                'text_features': len(df.select_dtypes(include=[object]).columns)
            }
        }
        
        # Run all checks
        report['missing_values'] = self.check_missing_values(df)
        report['duplicates'] = self.check_duplicates(df)
        report['label_distribution'] = self.check_label_distribution(df, label_column)
        report['feature_statistics'] = self.check_feature_statistics(df)
        report['outliers'] = self.check_outliers(df)
        report['correlation'] = self.check_feature_correlation(df)
        report['consistency'] = self.check_data_consistency(df)
        
        # Calculate overall quality score
        quality_score = 100
        
        # Deduct points for issues
        if report['missing_values']['total_missing'] > 0:
            quality_score -= min(20, report['missing_values']['total_missing'] / len(df) * 100)
        
        if report['duplicates']['total_duplicates'] > 0:
            quality_score -= min(15, report['duplicates']['duplicate_percentage'])
        
        if report['outliers']['total_outliers'] > 0:
            quality_score -= min(10, report['outliers']['total_outliers'] / (len(df) * 0.01))
        
        if report['label_distribution'].get('imbalance_ratio', 1) > 3:
            quality_score -= 10
        
        report['quality_score'] = max(0, quality_score)
        
        # Print summary
        print("\n" + "="*70)
        print("QUALITY SUMMARY")
        print("="*70)
        print(f"Overall Data Quality Score: {report['quality_score']:.1f}/100")
        
        if report['quality_score'] >= 80:
            print("✓ Data quality is GOOD")
        elif report['quality_score'] >= 60:
            print("⚠ Data quality is MODERATE - some issues need attention")
        else:
            print("❌ Data quality is POOR - significant issues detected")
        
        # Save report
        self.save_report(report)
        
        return report
    
    def save_report(self, report: Dict):
        """Save validation report to file."""
        report_file = self.output_dir / 'data_quality_report.json'
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj
        
        report_serializable = convert_types(report)
        
        with open(report_file, 'w') as f:
            json.dump(report_serializable, f, indent=2)
        
        print(f"\n✓ Report saved to {report_file}")
    
    def visualize_data_quality(self, df: pd.DataFrame, label_column: str = 'label'):
        """Create visualizations for data quality."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Label distribution
        if label_column in df.columns:
            df[label_column].value_counts().plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Label Distribution')
            axes[0, 0].set_xlabel('Label')
            axes[0, 0].set_ylabel('Count')
        
        # 2. Missing values heatmap (top 20 columns)
        missing_cols = df.isnull().sum().nlargest(20)
        if missing_cols.sum() > 0:
            missing_cols.plot(kind='barh', ax=axes[0, 1])
            axes[0, 1].set_title('Top 20 Columns with Missing Values')
            axes[0, 1].set_xlabel('Missing Count')
        else:
            axes[0, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
            axes[0, 1].set_title('Missing Values')
        
        # 3. Feature distributions (first 4 numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]
        if len(numeric_cols) > 0:
            for i, col in enumerate(numeric_cols):
                df[col].hist(bins=30, alpha=0.5, ax=axes[1, 0], label=col)
            axes[1, 0].set_title('Feature Distributions')
            axes[1, 0].legend()
        
        # 4. Correlation heatmap (top 10 features)
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            im = axes[1, 1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 1].set_title('Feature Correlation Heatmap')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        
        # Save figure
        fig_file = self.output_dir / 'data_quality_visualization.png'
        plt.savefig(fig_file, dpi=100, bbox_inches='tight')
        print(f"✓ Visualizations saved to {fig_file}")
        
        plt.close()


def main():
    """Test the data validation system."""
    print("\n" + "="*70)
    print("DATA VALIDATION TEST")
    print("="*70)
    
    # Load sample data
    data_path = Path('data/raw/phishing_email.csv')
    
    if data_path.exists():
        print(f"\nLoading data from {data_path}...")
        df = pd.read_csv(data_path, nrows=5000)  # Use subset for testing
        print(f"  Loaded {len(df)} samples")
        
        # Initialize validator
        validator = DataValidator()
        
        # Generate quality report
        report = validator.generate_quality_report(df)
        
        # Create visualizations
        print("\nGenerating visualizations...")
        validator.visualize_data_quality(df)
        
    else:
        print(f"❌ Data file not found: {data_path}")
        print("   Please ensure datasets are downloaded first")


if __name__ == "__main__":
    main()