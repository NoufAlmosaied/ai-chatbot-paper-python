# Dataset Download Guide for AI-Powered Chatbot Project

## Required Datasets (as per Proposal)

### 1. Phishing Dataset for Machine Learning
- **URL**: https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning
- **Referenced in proposal**: Section 3.1.2 and Section 3.2
- **Description**: Contains labeled phishing examples suitable for training machine learning models

### 2. Phishing Email Dataset  
- **URL**: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
- **Referenced in proposal**: Section 3.1.2 and Section 3.2
- **Description**: Provides phishing email samples for analysis

## Automated Download (Recommended)

Run the provided Python script:
```bash
python scripts/download_datasets.py
```

This script will:
1. Check for Kaggle API credentials
2. Download both datasets automatically
3. Extract them to the `data/raw/` directory
4. Create a data inventory

## Manual Download Instructions

If the automated script doesn't work, follow these steps:

### Step 1: Create a Kaggle Account
1. Go to https://www.kaggle.com
2. Sign up for a free account if you don't have one

### Step 2: Download Dataset 1
1. Visit: https://www.kaggle.com/datasets/shashwatwork/phishing-dataset-for-machine-learning
2. Click the "Download" button (you may need to accept terms)
3. Save the ZIP file to your `data/raw/` directory
4. Extract the contents

### Step 3: Download Dataset 2
1. Visit: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
2. Click the "Download" button
3. Save the ZIP file to your `data/raw/` directory
4. Extract the contents

### Step 4: Verify Downloads
After downloading, you should have the following structure:
```
data/
├── raw/
│   ├── [phishing dataset files].csv
│   └── [email dataset files].csv
├── processed/
└── interim/
```

## Setting up Kaggle API (for automated downloads)

1. Go to https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. Save the downloaded `kaggle.json` to `~/.kaggle/`
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## Data Management Best Practices

1. **Keep raw data unchanged**: Never modify files in `data/raw/`
2. **Document everything**: Record data sources, download dates, and any issues
3. **Version control**: Use git but add `data/` to `.gitignore` (data too large for git)
4. **Create backups**: Keep a copy of downloaded data in a separate location

## Troubleshooting

### "Dataset not found" error
- Make sure you're logged into Kaggle
- Accept the dataset's terms on the Kaggle website
- Check the dataset URL is correct

### "403 Forbidden" error  
- Your Kaggle API credentials may be incorrect
- Regenerate your API token and update `kaggle.json`

### Large file issues
- Some datasets may be large; ensure you have sufficient disk space
- Use a stable internet connection for downloads

## Next Steps

After successful download:
1. Run data exploration notebooks to understand the data
2. Begin data preprocessing (Phase 2, Weeks 3-4)
3. Document data characteristics for the methodology section

---
*Last updated: Data collection phase (Week 1)*