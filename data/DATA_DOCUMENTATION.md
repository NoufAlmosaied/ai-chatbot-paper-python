# Data Collection Documentation

## Overview
Successfully collected two comprehensive phishing datasets from Kaggle as specified in the project proposal (Sections 3.1.2 and 3.2).

## Primary Datasets

### 1. Phishing Dataset for Machine Learning
- **File**: `Phishing_Legitimate_full.csv`
- **Source**: Kaggle - [shashwatwork/phishing-dataset-for-machine-learning](https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning)
- **Size**: 1.34 MB
- **Records**: 10,000 (balanced: 5,000 phishing, 5,000 legitimate)
- **Features**: 50 URL-based features
- **Key Features**:
  - NumDots, SubdomainLevel, PathLevel
  - UrlLength, NumDash, NumDashInHostname
  - AtSymbol, TildeSymbol, NumUnderscore
  - NumPercent, NumQueryComponents
  - NumAmpersand, NumHash, NumNumericChars
  - NoHttps, RandomString, IpAddress
  - DomainInSubdomains, DomainInPaths
  - HttpsInHostname, HostnameLength
  - PathLength, QueryLength, DoubleSlashInPath
  - CLASS_LABEL (0=legitimate, 1=phishing)

### 2. Phishing Email Dataset Collection
- **Primary File**: `phishing_email.csv`
- **Source**: Kaggle - [naserabdullahalam/phishing-email-dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)
- **Total Size**: 246.55 MB across multiple files
- **Total Records**: 166,972 emails
- **Components**:
  
  | Dataset | Records | Description |
  |---------|---------|-------------|
  | phishing_email.csv | 82,486 | Combined text and labels |
  | CEAS_08.csv | 39,154 | Conference on Email and Anti-Spam dataset |
  | Enron.csv | 29,767 | Enron email corpus |
  | SpamAssasin.csv | 5,809 | SpamAssassin public corpus |
  | Nigerian_Fraud.csv | 3,332 | Nigerian fraud emails |
  | Ling.csv | 2,859 | Lingspam dataset |
  | Nazario.csv | 1,565 | Jose Nazario phishing corpus |

## Dataset Characteristics

### URL-Based Dataset (Phishing_Legitimate_full.csv)
- **Purpose**: Training models on URL structural features
- **Balance**: Perfectly balanced (50% phishing, 50% legitimate)
- **Feature Types**: Numerical features extracted from URLs
- **No missing values**: Complete dataset

### Email-Based Datasets
- **Purpose**: Training NLP models on email content
- **Content**: Full email text including headers, subject, body
- **Labels**: Binary classification (0=legitimate, 1=phishing/spam)
- **Language**: Primarily English
- **Temporal span**: Various periods from different sources

## Data Quality Assessment

### Strengths:
1. **Large sample size**: 166,972+ total email samples
2. **Diverse sources**: Multiple reputable datasets combined
3. **Balanced classes**: Good representation of both classes
4. **Real-world data**: Actual phishing and legitimate emails
5. **Multiple feature types**: Both URL and content features

### Considerations:
1. **Temporal bias**: Some datasets may be dated
2. **Language bias**: Primarily English content
3. **Feature engineering needed**: Raw email text requires processing
4. **Class imbalance**: Some subsets have imbalanced classes

## Suitability for Project Objectives

The collected datasets align perfectly with the proposal objectives:

1. **Objective 1**: Develop AI models for textual content analysis
   - ✓ 166,972 email samples provide rich text data
   
2. **Objective 2**: Achieve 95% detection accuracy
   - ✓ Large dataset size supports complex model training
   
3. **Objective 3**: Integration with email/messaging platforms
   - ✓ Real email data matches production use cases
   
4. **Objective 4**: Detect various phishing types
   - ✓ Multiple datasets cover different phishing tactics
   
5. **Objective 5**: Continuous learning mechanism
   - ✓ Diverse sources enable robust model generalization

## Data Storage Structure

```
data/
├── raw/                          # Original datasets
│   ├── Phishing_Legitimate_full.csv
│   ├── phishing_email.csv
│   ├── CEAS_08.csv
│   ├── Enron.csv
│   ├── SpamAssasin.csv
│   ├── Nigerian_Fraud.csv
│   ├── Ling.csv
│   └── Nazario.csv
├── processed/                    # Preprocessed data (Week 3-4)
├── interim/                      # Intermediate processing
├── dataset_summary.csv          # Summary statistics
└── data_inventory.json          # Detailed metadata
```

## Ethical Considerations

1. **Data Privacy**: All datasets are publicly available on Kaggle
2. **Consent**: Data has been anonymized and released for research
3. **No PII**: Personal information has been removed/anonymized
4. **Academic Use**: Datasets are used solely for research purposes
5. **Attribution**: Proper citations to original data sources

## Next Steps (Week 3-4: Data Preprocessing)

1. **Text Preprocessing**:
   - Email parsing and cleaning
   - Feature extraction from email headers
   - Text normalization and tokenization

2. **Feature Engineering**:
   - TF-IDF vectorization
   - N-gram extraction
   - Sentiment analysis features

3. **Data Splitting**:
   - Train/validation/test splits
   - Stratified sampling
   - Cross-validation folds

4. **Data Augmentation**:
   - Synthetic phishing email generation
   - Class balancing techniques

---

*Data collection completed: Week 1*  
*Total datasets: 8 CSV files*  
*Total records: 176,972*  
*Total size: 247.89 MB*