#!/usr/bin/env python3
"""
API Endpoint Tests - Validate Retrained Model Performance
Tests the Flask API with legitimate and phishing URLs to verify
that the retrained model performs correctly.
"""

import requests
import json


class APITester:
    """Test API endpoints with various URLs."""

    def __init__(self, base_url="http://localhost:5001"):
        """Initialize API tester."""
        self.base_url = base_url
        self.results = {
            'legitimate': [],
            'phishing': []
        }

    def test_url(self, url, expected_phishing, label):
        """Test a single URL."""
        try:
            response = requests.post(
                f"{self.base_url}/api/analyze",
                json={"content": url, "type": "url"},
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                analysis = data['analysis']

                result = {
                    'url': url,
                    'label': label,
                    'is_phishing': analysis['is_phishing'],
                    'confidence': analysis['confidence'] * 100,
                    'risk_level': analysis['risk_level'],
                    'risk_score': analysis['risk_score'],
                    'expected': expected_phishing,
                    'correct': analysis['is_phishing'] == expected_phishing
                }

                # Store result
                if expected_phishing:
                    self.results['phishing'].append(result)
                else:
                    self.results['legitimate'].append(result)

                return result
            else:
                print(f"  ERROR: Status code {response.status_code}")
                return None

        except Exception as e:
            print(f"  ERROR: {str(e)}")
            return None

    def run_all_tests(self):
        """Run all test cases."""
        print("\n" + "="*70)
        print("API ENDPOINT TESTS - RETRAINED MODEL VALIDATION")
        print("="*70)

        # Test legitimate URLs
        print("\n[1] Testing LEGITIMATE URLs")
        print("-"*70)

        legitimate_urls = [
            ("https://www.google.com", "Google Search"),
            ("https://www.amazon.com", "Amazon"),
            ("https://www.paypal.com", "PayPal"),
            ("https://www.microsoft.com", "Microsoft"),
            ("https://www.apple.com", "Apple"),
            ("https://www.facebook.com", "Facebook"),
            ("https://www.netflix.com", "Netflix"),
            ("https://accounts.google.com", "Google Accounts"),
            ("https://mail.google.com", "Gmail"),
            ("https://aws.amazon.com", "AWS")
        ]

        for url, label in legitimate_urls:
            result = self.test_url(url, expected_phishing=False, label=label)
            if result:
                status = "✓ PASS" if result['correct'] else "✗ FAIL"
                print(f"{status} | {label:20s} | Phishing={result['is_phishing']} | "
                      f"Confidence={result['confidence']:.1f}% | Risk={result['risk_level']}")

        # Test phishing URLs
        print("\n[2] Testing PHISHING URLs")
        print("-"*70)

        phishing_urls = [
            ("http://paypa1-verify.com/login", "PayPal typosquatting"),
            ("http://secure-paypal-login.tk", "Fake PayPal login"),
            ("http://amazon-account-update.net", "Fake Amazon update"),
            ("http://google-login-verify.com", "Fake Google login"),
            ("http://microsoft-security-alert.org", "Fake Microsoft alert"),
            ("http://apple-support-center.info", "Fake Apple support"),
            ("http://192.168.1.100/paypal", "IP address phishing"),
            ("http://g00gle-login.com", "Google typosquatting"),
            ("http://amaz0n-verify.com", "Amazon typosquatting"),
            ("http://www.paypal-secure-login-verify-account.tk", "Long suspicious URL")
        ]

        for url, label in phishing_urls:
            result = self.test_url(url, expected_phishing=True, label=label)
            if result:
                status = "✓ PASS" if result['correct'] else "✗ FAIL"
                print(f"{status} | {label:25s} | Phishing={result['is_phishing']} | "
                      f"Confidence={result['confidence']:.1f}% | Risk={result['risk_level']}")

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)

        # Legitimate URLs
        legit_total = len(self.results['legitimate'])
        legit_correct = sum(1 for r in self.results['legitimate'] if r['correct'])
        legit_accuracy = (legit_correct / legit_total * 100) if legit_total > 0 else 0

        print(f"\nLegitimate URLs:")
        print(f"  Total tested: {legit_total}")
        print(f"  Correct: {legit_correct}")
        print(f"  Accuracy: {legit_accuracy:.1f}%")

        if legit_total > 0:
            avg_confidence = sum(r['confidence'] for r in self.results['legitimate']) / legit_total
            print(f"  Average confidence: {avg_confidence:.1f}%")

        # Phishing URLs
        phish_total = len(self.results['phishing'])
        phish_correct = sum(1 for r in self.results['phishing'] if r['correct'])
        phish_accuracy = (phish_correct / phish_total * 100) if phish_total > 0 else 0

        print(f"\nPhishing URLs:")
        print(f"  Total tested: {phish_total}")
        print(f"  Correct: {phish_correct}")
        print(f"  Accuracy: {phish_accuracy:.1f}%")

        if phish_total > 0:
            avg_confidence = sum(r['confidence'] for r in self.results['phishing']) / phish_total
            print(f"  Average confidence: {avg_confidence:.1f}%")

        # Overall
        total = legit_total + phish_total
        correct = legit_correct + phish_correct
        overall_accuracy = (correct / total * 100) if total > 0 else 0

        print(f"\nOverall:")
        print(f"  Total tested: {total}")
        print(f"  Correct: {correct}")
        print(f"  Accuracy: {overall_accuracy:.1f}%")

        print("\n" + "="*70)

        # Save results
        self.save_results()

    def save_results(self):
        """Save test results to JSON."""
        import os
        from pathlib import Path
        from datetime import datetime

        output_dir = Path(__file__).parent.parent / "reports" / "api_tests"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"api_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to {output_file}")


def main():
    """Run API tests."""
    tester = APITester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
