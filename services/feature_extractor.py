"""
Feature Extractor Service
Extracts the 100 features from URLs for phishing detection
"""

import numpy as np
from urllib.parse import urlparse, parse_qs
import re
import ipaddress
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from URLs for phishing detection."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        self.suspicious_words = [
            'secure', 'account', 'update', 'suspend', 'verify',
            'confirm', 'banking', 'paypal', 'amazon', 'microsoft',
            'apple', 'google', 'facebook', 'instagram', 'twitter'
        ]
        
    def extract(self, content: str, content_type: str = 'url') -> Optional[np.ndarray]:
        """
        Extract features from content.
        
        Args:
            content: URL or email text to analyze
            content_type: Type of content ('url' or 'email')
            
        Returns:
            Feature vector of shape (100,) or None if extraction fails
        """
        try:
            if content_type == 'url':
                return self.extract_url_features(content)
            else:
                # For email, extract URLs and analyze them
                urls = self.extract_urls_from_text(content)
                if urls:
                    return self.extract_url_features(urls[0])
                else:
                    # Return default features for text without URLs
                    return self.extract_text_features(content)
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return None
    
    def extract_url_features(self, url: str) -> np.ndarray:
        """
        Extract 48 features from a URL.

        Features based on the Phishing_Legitimate_full dataset structure.
        """
        features = []
        
        # Add http if not present
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path
            query = parsed.query
            
            # Feature 1-15: URL Structural Features
            features.append(self.count_dots(url))  # NumDots
            features.append(self.subdomain_level(domain))  # SubdomainLevel
            features.append(self.path_level(path))  # PathLevel
            features.append(len(url))  # UrlLength
            features.append(url.count('-'))  # NumDash
            features.append(domain.count('-'))  # NumDashInHostname
            features.append(1 if '@' in url else 0)  # AtSymbol
            features.append(1 if '~' in url else 0)  # TildeSymbol
            features.append(url.count('_'))  # NumUnderscore
            features.append(url.count('%'))  # NumPercent
            features.append(len(parse_qs(query)))  # NumQueryComponents
            features.append(url.count('&'))  # NumAmpersand
            features.append(url.count('#'))  # NumHash
            features.append(sum(c.isdigit() for c in url))  # NumNumericChars
            features.append(0 if url.startswith('https://') else 1)  # NoHttps
            
            # Feature 16-25: Special Character Features
            features.append(self.has_random_string(domain))  # RandomString
            features.append(self.is_ip_address(domain))  # IpAddress
            features.append(self.domain_in_subdomains(domain))  # DomainInSubdomains
            features.append(self.domain_in_paths(domain, path))  # DomainInPaths
            features.append(1 if 'https' in domain else 0)  # HttpsInHostname
            features.append(len(domain))  # HostnameLength
            features.append(len(path))  # PathLength
            features.append(len(query))  # QueryLength
            features.append(1 if '//' in path else 0)  # DoubleSlashInPath
            features.append(self.count_sensitive_words(url))  # NumSensitiveWords
            
            # Feature 26-35: Content and Security Features
            features.append(self.has_embedded_brand(domain))  # EmbeddedBrandName
            features.append(0.0)  # PctExtHyperlinks (would need page content)
            features.append(0.0)  # PctExtResourceUrls (would need page content)
            features.append(0)  # ExtFavicon (would need page content)
            features.append(0)  # InsecureForms (would need page content)
            features.append(0)  # RelativeFormAction (would need page content)
            features.append(0)  # ExtFormAction (would need page content)
            features.append(0)  # AbnormalFormAction (would need page content)
            features.append(0.0)  # PctNullSelfRedirectHyperlinks
            features.append(0)  # FrequentDomainNameMismatch
            
            # Feature 36-48: Statistical and Behavioral Features
            features.append(0)  # FakeLinkInStatusBar
            features.append(0)  # RightClickDisabled
            features.append(0)  # PopUpWindow
            features.append(0)  # SubmitInfoToEmail
            features.append(0)  # IframeOrFrame
            features.append(0)  # MissingTitle
            features.append(0)  # ImagesOnlyInForm
            features.append(self.subdomain_level_rt(domain))  # SubdomainLevelRT
            features.append(self.url_length_rt(url))  # UrlLengthRT
            features.append(0)  # PctExtResourceUrlsRT
            features.append(0)  # AbnormalExtFormActionR
            features.append(0)  # ExtMetaScriptLinkRT
            features.append(0)  # PctExtNullSelfRedirectHyperlinksRT
            
        except Exception as e:
            logger.error(f"Error extracting URL features: {str(e)}")
            # Return default features on error
            features = [0] * 48

        # Ensure exactly 48 features
        return np.array(features[:48], dtype=np.float32)
    
    def extract_text_features(self, text: str) -> np.ndarray:
        """Extract features from email text."""
        # For emails without URLs, create basic features (48 features)
        features = np.zeros(48, dtype=np.float32)

        # Set some text-based features
        features[3] = min(len(text), 1000)  # Length indicator
        features[9] = text.count('%')  # Encoded characters
        features[13] = sum(c.isdigit() for c in text)  # Numeric chars
        features[25] = self.count_sensitive_words(text)  # Sensitive words

        return features
    
    def count_dots(self, url: str) -> int:
        """Count dots in URL."""
        return url.count('.')
    
    def subdomain_level(self, domain: str) -> int:
        """Count subdomain levels."""
        parts = domain.split('.')
        # Typical domain has 2 parts (example.com)
        return max(0, len(parts) - 2)
    
    def path_level(self, path: str) -> int:
        """Count path depth."""
        if not path or path == '/':
            return 0
        return path.count('/')
    
    def has_random_string(self, domain: str) -> int:
        """Check if domain appears to have random string."""
        # Check for long sequences of consonants or mixed case
        consonants = sum(1 for c in domain if c.lower() in 'bcdfghjklmnpqrstvwxyz')
        if len(domain) > 0:
            ratio = consonants / len(domain)
            return 1 if ratio > 0.8 else 0
        return 0
    
    def is_ip_address(self, domain: str) -> int:
        """Check if domain is an IP address."""
        try:
            ipaddress.ip_address(domain)
            return 1
        except ValueError:
            return 0
    
    def domain_in_subdomains(self, domain: str) -> int:
        """Check if brand names appear in subdomain."""
        parts = domain.lower().split('.')
        if len(parts) > 2:
            subdomain = '.'.join(parts[:-2])
            for word in self.suspicious_words:
                if word in subdomain:
                    return 1
        return 0
    
    def domain_in_paths(self, domain: str, path: str) -> int:
        """Check if domain keywords appear in path."""
        for word in self.suspicious_words:
            if word in path.lower():
                return 1
        return 0
    
    def count_sensitive_words(self, text: str) -> int:
        """Count sensitive/suspicious words."""
        text_lower = text.lower()
        count = 0
        suspicious_terms = [
            'verify', 'update', 'confirm', 'suspend', 'restrict',
            'urgent', 'expire', 'click here', 'act now', 'limited time'
        ]
        for term in suspicious_terms:
            count += text_lower.count(term)
        return count
    
    def has_embedded_brand(self, domain: str) -> int:
        """Check if brand name is embedded in suspicious way."""
        domain_lower = domain.lower()
        brands = ['paypal', 'amazon', 'google', 'microsoft', 'apple', 'facebook']
        
        for brand in brands:
            if brand in domain_lower:
                # Check if it's not the actual brand domain
                if not domain_lower.startswith(brand + '.') and not domain_lower == brand + '.com':
                    return 1
        return 0
    
    def subdomain_level_rt(self, domain: str) -> int:
        """Relative threshold for subdomain level."""
        level = self.subdomain_level(domain)
        if level > 3:
            return 1  # High
        elif level > 1:
            return 0  # Medium
        else:
            return -1  # Low
    
    def url_length_rt(self, url: str) -> int:
        """Relative threshold for URL length."""
        length = len(url)
        if length > 75:
            return 1  # Long
        elif length > 50:
            return 0  # Medium
        else:
            return -1  # Short
    
    def extract_urls_from_text(self, text: str) -> list:
        """Extract URLs from email text."""
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return urls