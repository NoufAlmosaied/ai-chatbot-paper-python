#!/usr/bin/env python3
"""
Unit Tests for Feature Extraction
Tests the corrected feature extraction logic for phishing detection
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from services.feature_extractor import FeatureExtractor


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for FeatureExtractor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()

    # Test has_embedded_brand() method

    def test_embedded_brand_legitimate_google(self):
        """Test legitimate Google domains return 0."""
        legitimate_domains = [
            'google.com',
            'www.google.com',
            'accounts.google.com',
            'mail.google.com',
            'drive.google.com'
        ]
        for domain in legitimate_domains:
            with self.subTest(domain=domain):
                result = self.extractor.has_embedded_brand(domain)
                self.assertEqual(result, 0, f"Failed for {domain}")

    def test_embedded_brand_legitimate_amazon(self):
        """Test legitimate Amazon domains return 0."""
        legitimate_domains = [
            'amazon.com',
            'www.amazon.com',
            'aws.amazon.com',
            'smile.amazon.com'
        ]
        for domain in legitimate_domains:
            with self.subTest(domain=domain):
                result = self.extractor.has_embedded_brand(domain)
                self.assertEqual(result, 0, f"Failed for {domain}")

    def test_embedded_brand_legitimate_paypal(self):
        """Test legitimate PayPal domains return 0."""
        legitimate_domains = [
            'paypal.com',
            'www.paypal.com',
            'business.paypal.com'
        ]
        for domain in legitimate_domains:
            with self.subTest(domain=domain):
                result = self.extractor.has_embedded_brand(domain)
                self.assertEqual(result, 0, f"Failed for {domain}")

    def test_embedded_brand_phishing_suspicious_placement(self):
        """Test phishing URLs with brand names in suspicious positions."""
        phishing_domains = [
            'paypal-verify.com',
            'secure-paypal.com',
            'paypal-login.tk',
            'google-login.net',
            'amazon-account.com',
            'verify-amazon.org',
            'facebook-security.info',
            'apple-support-center.com'
        ]
        for domain in phishing_domains:
            with self.subTest(domain=domain):
                result = self.extractor.has_embedded_brand(domain)
                self.assertEqual(result, 1, f"Failed to detect phishing in {domain}")

    def test_embedded_brand_typosquatting(self):
        """Test detection of typosquatting variants."""
        typosquat_domains = [
            'paypa1.com',  # paypal with 1 instead of l
            'g00gle.com',  # google with 0s
            'amaz0n.com',  # amazon with 0
            'micros0ft.com',  # microsoft with 0
            'paypai.com',  # paypal misspelled
            'gooogle.com',  # google with extra o
        ]
        for domain in typosquat_domains:
            with self.subTest(domain=domain):
                result = self.extractor.has_embedded_brand(domain)
                self.assertEqual(result, 1, f"Failed to detect typosquatting in {domain}")

    def test_embedded_brand_no_brand(self):
        """Test domains without brand names return 0."""
        no_brand_domains = [
            'example.com',
            'test.org',
            'mywebsite.net',
            'random-domain.co.uk'
        ]
        for domain in no_brand_domains:
            with self.subTest(domain=domain):
                result = self.extractor.has_embedded_brand(domain)
                self.assertEqual(result, 0, f"Failed for {domain}")

    # Test other feature extraction methods

    def test_count_dots(self):
        """Test dot counting."""
        test_cases = [
            ('http://example.com', 1),
            ('http://www.example.com', 2),
            ('http://sub.domain.example.com', 3),
            ('http://a.b.c.d.com', 4)
        ]
        for url, expected in test_cases:
            with self.subTest(url=url):
                result = self.extractor.count_dots(url)
                self.assertEqual(result, expected)

    def test_subdomain_level(self):
        """Test subdomain level calculation."""
        test_cases = [
            ('example.com', 0),  # No subdomain
            ('www.example.com', 1),  # 1 subdomain
            ('mail.google.com', 1),  # 1 subdomain
            ('sub1.sub2.example.com', 2),  # 2 subdomains
            ('a.b.c.example.com', 3)  # 3 subdomains
        ]
        for domain, expected in test_cases:
            with self.subTest(domain=domain):
                result = self.extractor.subdomain_level(domain)
                self.assertEqual(result, expected)

    def test_path_level(self):
        """Test path level calculation."""
        test_cases = [
            ('/', 0),
            ('/login', 1),
            ('/account/settings', 2),
            ('/users/profile/edit', 3),
            ('', 0)
        ]
        for path, expected in test_cases:
            with self.subTest(path=path):
                result = self.extractor.path_level(path)
                self.assertEqual(result, expected)

    def test_is_ip_address(self):
        """Test IP address detection."""
        test_cases = [
            ('192.168.1.1', 1),
            ('127.0.0.1', 1),
            ('10.0.0.1', 1),
            ('example.com', 0),
            ('www.google.com', 0),
            ('invalid.ip', 0)
        ]
        for domain, expected in test_cases:
            with self.subTest(domain=domain):
                result = self.extractor.is_ip_address(domain)
                self.assertEqual(result, expected)

    def test_count_sensitive_words(self):
        """Test sensitive word counting."""
        test_cases = [
            ('Please verify your account', 1),  # verify
            ('Update your password now', 1),  # update
            ('Click here to confirm', 2),  # confirm + click here
            ('Urgent: Your account will expire', 2),  # urgent + expire
            ('Normal text with no triggers', 0)
        ]
        for text, min_expected in test_cases:
            with self.subTest(text=text):
                result = self.extractor.count_sensitive_words(text)
                self.assertGreaterEqual(result, min_expected)

    def test_url_length_rt(self):
        """Test URL length relative threshold."""
        test_cases = [
            ('http://short.com', -1),  # Short (<= 50)
            ('http://medium-length-url-that-is-somewhat-longish.org', 0),  # Medium (> 50 and <= 75)
            ('http://very-long-url-that-exceeds-the-threshold-and-keeps-going-on-and-on.com', 1)  # Long (> 75)
        ]
        for url, expected in test_cases:
            with self.subTest(url=url):
                result = self.extractor.url_length_rt(url)
                self.assertEqual(result, expected)

    def test_subdomain_level_rt(self):
        """Test subdomain level relative threshold."""
        test_cases = [
            ('example.com', -1),  # Low (0-1)
            ('www.example.com', -1),  # Low (0-1)
            ('a.b.example.com', 0),  # Medium (2-3)
            ('a.b.c.example.com', 0),  # Medium (2-3)
            ('a.b.c.d.example.com', 1)  # High (4+)
        ]
        for domain, expected in test_cases:
            with self.subTest(domain=domain):
                result = self.extractor.subdomain_level_rt(domain)
                self.assertEqual(result, expected)

    # Test full feature extraction

    def test_extract_url_features_shape(self):
        """Test that feature extraction returns correct shape."""
        url = 'http://example.com/test'
        features = self.extractor.extract_url_features(url)
        self.assertEqual(features.shape, (48,), "Feature vector should have 48 dimensions")
        self.assertEqual(features.dtype, np.float32, "Features should be float32")

    def test_extract_url_features_legitimate(self):
        """Test feature extraction for legitimate URL."""
        url = 'https://www.google.com'
        features = self.extractor.extract_url_features(url)

        # Check specific features
        self.assertEqual(features[14], 0, "NoHttps should be 0 for HTTPS")  # NoHttps
        self.assertEqual(features[25], 0, "EmbeddedBrandName should be 0 for google.com")  # EmbeddedBrandName

    def test_extract_url_features_phishing(self):
        """Test feature extraction for phishing URL."""
        url = 'http://paypa1-verify.suspicious.com/login'
        features = self.extractor.extract_url_features(url)

        # Check specific features
        self.assertEqual(features[14], 1, "NoHttps should be 1 for HTTP")  # NoHttps
        self.assertEqual(features[25], 1, "EmbeddedBrandName should be 1 for paypa1")  # EmbeddedBrandName
        self.assertGreater(features[1], 0, "SubdomainLevel should be > 0")  # SubdomainLevel

    def test_extract_handles_missing_protocol(self):
        """Test that URLs without protocol are handled correctly."""
        url = 'example.com'
        features = self.extractor.extract_url_features(url)
        self.assertEqual(features.shape, (48,), "Should handle URLs without protocol")

    def test_extract_handles_errors(self):
        """Test that extraction handles errors gracefully."""
        invalid_urls = ['', 'not-a-url', '://invalid', 'ht tp://broken.com']
        for url in invalid_urls:
            with self.subTest(url=url):
                features = self.extractor.extract_url_features(url)
                self.assertEqual(features.shape, (48,), f"Should return default features for '{url}'")


class TestFeatureConsistency(unittest.TestCase):
    """Test that features are consistent with training data format."""

    def setUp(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()

    def test_feature_order_matches_dataset(self):
        """Verify feature extraction order matches Phishing_Legitimate_full.csv."""
        url = 'http://test.example.com/path'
        features = self.extractor.extract_url_features(url)

        # Verify feature positions (as documented in dataset)
        # Feature 0: NumDots
        # Feature 1: SubdomainLevel
        # Feature 2: PathLevel
        # Feature 3: UrlLength
        # Feature 14: NoHttps
        # Feature 25: EmbeddedBrandName

        self.assertIsInstance(features[0], (int, float, np.floating, np.integer))
        self.assertIsInstance(features[1], (int, float, np.floating, np.integer))
        self.assertIsInstance(features[14], (int, float, np.floating, np.integer))
        self.assertIsInstance(features[25], (int, float, np.floating, np.integer))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
