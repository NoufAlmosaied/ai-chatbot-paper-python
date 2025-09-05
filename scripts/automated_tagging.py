#!/usr/bin/env python3
"""
Automated Tagging System for Phishing Detection
Phase 2: Data Annotation and Preprocessing
"""

import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import ipaddress
from typing import Dict, List, Tuple
import nltk
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class AutomatedTagger:
    """Automated tagging system for phishing detection."""
    
    def __init__(self):
        """Initialize the tagger with pattern dictionaries."""
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Phishing patterns
        self.urgency_keywords = [
            'urgent', 'immediate', 'act now', 'expire', 'suspend', 'deadline',
            'limited time', 'hurry', 'quick', 'fast', 'asap', 'important'
        ]
        
        self.authority_keywords = [
            'bank', 'security', 'official', 'government', 'irs', 'tax',
            'police', 'fbi', 'legal', 'court', 'administrator', 'support'
        ]
        
        self.financial_keywords = [
            'payment', 'invoice', 'refund', 'credit', 'debit', 'account',
            'transaction', 'money', 'dollar', 'bitcoin', 'prize', 'lottery'
        ]
        
        self.action_keywords = [
            'click', 'verify', 'confirm', 'update', 'validate', 'check',
            'review', 'download', 'install', 'open', 'view', 'see'
        ]
        
        self.threat_keywords = [
            'suspend', 'terminate', 'close', 'block', 'lock', 'disable',
            'deactivate', 'expire', 'cancel', 'delete', 'remove'
        ]
        
        # Suspicious TLDs often used in phishing
        self.suspicious_tlds = [
            '.tk', '.ml', '.ga', '.cf', '.info', '.biz', '.click',
            '.download', '.review', '.top', '.win', '.bid', '.loan'
        ]
        
    def extract_url_features(self, url: str) -> Dict:
        """Extract features from URL."""
        features = {
            'has_ip': 0,
            'url_length': len(url),
            'has_at_symbol': 1 if '@' in url else 0,
            'double_slash_redirect': 1 if '//' in url[8:] else 0,
            'has_hyphen': 1 if '-' in url else 0,
            'subdomain_count': 0,
            'suspicious_tld': 0,
            'has_https': 1 if url.startswith('https://') else 0,
            'port_in_url': 0,
            'shortening_service': 0
        }
        
        # Check for IP address
        try:
            parsed = urlparse(url)
            ipaddress.ip_address(parsed.hostname)
            features['has_ip'] = 1
        except:
            pass
        
        # Check for suspicious TLD
        for tld in self.suspicious_tlds:
            if tld in url.lower():
                features['suspicious_tld'] = 1
                break
        
        # Count subdomains
        try:
            parsed = urlparse(url)
            if parsed.hostname:
                features['subdomain_count'] = len(parsed.hostname.split('.')) - 2
        except:
            pass
        
        # Check for port
        if re.search(r':\d+', url):
            features['port_in_url'] = 1
        
        # Check for URL shorteners
        shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 'ow.ly', 't.co']
        for shortener in shorteners:
            if shortener in url.lower():
                features['shortening_service'] = 1
                break
        
        return features
    
    def extract_text_features(self, text: str) -> Dict:
        """Extract linguistic features from text."""
        if not text or pd.isna(text):
            return self._empty_text_features()
        
        text_lower = text.lower()
        
        features = {
            'urgency_score': self._calculate_keyword_score(text_lower, self.urgency_keywords),
            'authority_score': self._calculate_keyword_score(text_lower, self.authority_keywords),
            'financial_score': self._calculate_keyword_score(text_lower, self.financial_keywords),
            'action_score': self._calculate_keyword_score(text_lower, self.action_keywords),
            'threat_score': self._calculate_keyword_score(text_lower, self.threat_keywords),
            'sentiment_compound': 0,
            'sentiment_negative': 0,
            'sentiment_positive': 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
            'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1),
            'misspelling_score': 0
        }
        
        # Sentiment analysis
        try:
            sentiment = self.sia.polarity_scores(text)
            features['sentiment_compound'] = sentiment['compound']
            features['sentiment_negative'] = sentiment['neg']
            features['sentiment_positive'] = sentiment['pos']
        except:
            pass
        
        # Basic spell checking (using TextBlob)
        try:
            blob = TextBlob(text[:1000])  # Limit to first 1000 chars for speed
            corrected = blob.correct()
            if str(corrected) != str(blob):
                features['misspelling_score'] = 1
        except:
            pass
        
        return features
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate normalized keyword presence score."""
        score = sum(1 for keyword in keywords if keyword in text)
        return min(score / max(len(keywords), 1), 1.0)
    
    def _empty_text_features(self) -> Dict:
        """Return empty text features dict."""
        return {
            'urgency_score': 0,
            'authority_score': 0,
            'financial_score': 0,
            'action_score': 0,
            'threat_score': 0,
            'sentiment_compound': 0,
            'sentiment_negative': 0,
            'sentiment_positive': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'caps_ratio': 0,
            'special_char_ratio': 0,
            'misspelling_score': 0
        }
    
    def calculate_phishing_confidence(self, features: Dict) -> float:
        """Calculate overall phishing confidence score (0-1)."""
        weights = {
            'urgency_score': 0.15,
            'authority_score': 0.10,
            'financial_score': 0.10,
            'action_score': 0.10,
            'threat_score': 0.15,
            'has_ip': 0.10,
            'suspicious_tld': 0.05,
            'sentiment_negative': 0.05,
            'misspelling_score': 0.05,
            'special_char_ratio': 0.05,
            'has_at_symbol': 0.05,
            'shortening_service': 0.05
        }
        
        score = 0
        for feature, weight in weights.items():
            if feature in features:
                score += features[feature] * weight
        
        return min(max(score, 0), 1)
    
    def tag_email(self, text: str, url: str = None) -> Dict:
        """Tag an email with automated annotations."""
        tags = {
            'text_features': {},
            'url_features': {},
            'tags': [],
            'confidence': 0,
            'risk_level': 'low'
        }
        
        # Extract text features
        if text:
            tags['text_features'] = self.extract_text_features(text)
            
            # Add specific tags based on scores
            if tags['text_features']['urgency_score'] > 0.3:
                tags['tags'].append('high_urgency')
            if tags['text_features']['threat_score'] > 0.3:
                tags['tags'].append('contains_threats')
            if tags['text_features']['financial_score'] > 0.3:
                tags['tags'].append('financial_content')
            if tags['text_features']['misspelling_score'] > 0:
                tags['tags'].append('spelling_errors')
        
        # Extract URL features if provided
        if url:
            tags['url_features'] = self.extract_url_features(url)
            
            # Add URL-based tags
            if tags['url_features']['has_ip']:
                tags['tags'].append('ip_address_url')
            if tags['url_features']['suspicious_tld']:
                tags['tags'].append('suspicious_domain')
            if tags['url_features']['shortening_service']:
                tags['tags'].append('shortened_url')
        
        # Calculate overall confidence
        all_features = {**tags['text_features'], **tags['url_features']}
        tags['confidence'] = self.calculate_phishing_confidence(all_features)
        
        # Determine risk level
        if tags['confidence'] >= 0.7:
            tags['risk_level'] = 'high'
        elif tags['confidence'] >= 0.4:
            tags['risk_level'] = 'medium'
        else:
            tags['risk_level'] = 'low'
        
        return tags
    
    def process_dataset(self, df: pd.DataFrame, text_column: str, 
                       url_column: str = None, batch_size: int = 100) -> pd.DataFrame:
        """Process entire dataset with automated tagging."""
        print(f"Processing {len(df)} samples...")
        
        # Initialize new columns
        df['automated_tags'] = None
        df['confidence_score'] = 0.0
        df['risk_level'] = 'low'
        df['text_features'] = None
        df['url_features'] = None
        
        # Process in batches
        for i in range(0, len(df), batch_size):
            batch_end = min(i + batch_size, len(df))
            print(f"  Processing batch {i//batch_size + 1}: rows {i} to {batch_end}")
            
            for idx in range(i, batch_end):
                text = df.iloc[idx][text_column] if text_column else None
                url = df.iloc[idx][url_column] if url_column else None
                
                # Get tags
                result = self.tag_email(text, url)
                
                # Store results
                df.at[df.index[idx], 'automated_tags'] = result['tags']
                df.at[df.index[idx], 'confidence_score'] = result['confidence']
                df.at[df.index[idx], 'risk_level'] = result['risk_level']
                df.at[df.index[idx], 'text_features'] = result['text_features']
                df.at[df.index[idx], 'url_features'] = result['url_features']
        
        print(f"✓ Tagging complete for {len(df)} samples")
        return df


def main():
    """Test the automated tagging system."""
    print("\n" + "="*70)
    print("AUTOMATED TAGGING SYSTEM TEST")
    print("="*70)
    
    # Initialize tagger
    tagger = AutomatedTagger()
    
    # Test with sample phishing email
    sample_phishing = """
    URGENT: Your account will be suspended!
    
    Dear Customer,
    
    We have detected suspicious activity on your account. 
    Please click here immediately to verify your identity or your account will be terminated.
    
    This is a time-sensitive matter. Act now!
    
    Security Department
    """
    
    sample_url = "http://192.168.1.1/verify-account.html"
    
    print("\nSample Phishing Email Test:")
    print("-" * 50)
    result = tagger.tag_email(sample_phishing, sample_url)
    
    print(f"Tags: {result['tags']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"\nText Features:")
    for feature, value in result['text_features'].items():
        if value > 0:
            print(f"  - {feature}: {value:.3f}")
    
    # Test with legitimate email
    sample_legitimate = """
    Meeting Reminder
    
    Hi Team,
    
    This is a reminder about our weekly team meeting scheduled for tomorrow at 2 PM.
    Please review the attached agenda before the meeting.
    
    Best regards,
    John
    """
    
    print("\n" + "="*70)
    print("Sample Legitimate Email Test:")
    print("-" * 50)
    result = tagger.tag_email(sample_legitimate)
    
    print(f"Tags: {result['tags']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    
    print("\n✓ Automated tagging system ready for use")


if __name__ == "__main__":
    main()