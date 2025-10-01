#!/usr/bin/env python3
"""
Comprehensive System Testing
Validates the entire PhishGuard AI system end-to-end
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from services.feature_extractor import FeatureExtractor
from services.ml_service import MLService
from chatbot.risk_analyzer import RiskAnalyzer
from chatbot.conversation import ChatbotEngine


class ComprehensiveSystemTest:
    """Comprehensive testing of the entire system."""

    def __init__(self):
        """Initialize test components."""
        self.feature_extractor = FeatureExtractor()
        self.ml_service = MLService()
        self.risk_analyzer = RiskAnalyzer()
        self.chatbot = ChatbotEngine()
        self.test_results = {
            'feature_extraction': [],
            'model_prediction': [],
            'risk_assessment': [],
            'end_to_end': []
        }

    def test_feature_extraction(self):
        """Test feature extraction functionality."""
        print("\n" + "="*70)
        print("TEST 1: FEATURE EXTRACTION")
        print("="*70)

        test_cases = [
            {
                'url': 'https://www.google.com',
                'expected_embedded_brand': 0,
                'expected_no_https': 0,
                'label': 'Legitimate Google'
            },
            {
                'url': 'http://paypa1-verify.com',
                'expected_embedded_brand': 1,
                'expected_no_https': 1,
                'label': 'Phishing PayPal variant'
            },
            {
                'url': 'https://www.amazon.com',
                'expected_embedded_brand': 0,
                'expected_no_https': 0,
                'label': 'Legitimate Amazon'
            },
            {
                'url': 'http://secure-paypal-login.tk',
                'expected_embedded_brand': 1,
                'expected_no_https': 1,
                'label': 'Phishing fake PayPal'
            }
        ]

        passed = 0
        failed = 0

        for test in test_cases:
            features = self.feature_extractor.extract_url_features(test['url'])

            # Check feature vector shape
            shape_ok = features.shape == (48,)

            # Check specific features
            embedded_brand_ok = features[25] == test['expected_embedded_brand']
            no_https_ok = features[14] == test['expected_no_https']

            all_ok = shape_ok and embedded_brand_ok and no_https_ok

            status = "✓ PASS" if all_ok else "✗ FAIL"

            print(f"\n{status} | {test['label']}")
            print(f"  URL: {test['url']}")
            print(f"  Shape: {features.shape} {'✓' if shape_ok else '✗'}")
            print(f"  EmbeddedBrandName: {features[25]} (expected {test['expected_embedded_brand']}) {'✓' if embedded_brand_ok else '✗'}")
            print(f"  NoHttps: {features[14]} (expected {test['expected_no_https']}) {'✓' if no_https_ok else '✗'}")

            if all_ok:
                passed += 1
            else:
                failed += 1

            self.test_results['feature_extraction'].append({
                'url': test['url'],
                'passed': all_ok
            })

        print(f"\n{'='*70}")
        print(f"Feature Extraction Results: {passed} passed, {failed} failed")
        print(f"{'='*70}")

        return passed, failed

    def test_model_prediction(self):
        """Test model prediction functionality."""
        print("\n" + "="*70)
        print("TEST 2: MODEL PREDICTION")
        print("="*70)

        # Load test data from dataset
        dataset_path = Path(__file__).parent.parent / "data" / "raw" / "Phishing_Legitimate_full.csv"
        df = pd.read_csv(dataset_path)

        # Take 100 random samples
        test_samples = df.sample(n=100, random_state=42)
        X_test = test_samples.drop(['id', 'CLASS_LABEL'], axis=1).values
        y_test = test_samples['CLASS_LABEL'].values

        print(f"\nTesting with 100 random samples from dataset...")
        print(f"  Legitimate: {(y_test==0).sum()}")
        print(f"  Phishing: {(y_test==1).sum()}")

        # Make predictions
        predictions = []
        correct = 0

        for i, (features, true_label) in enumerate(zip(X_test, y_test)):
            result = self.ml_service.predict(features)
            predicted_label = 1 if result['is_phishing'] else 0

            if predicted_label == true_label:
                correct += 1

            predictions.append({
                'true_label': int(true_label),
                'predicted_label': predicted_label,
                'confidence': result['confidence'],
                'correct': predicted_label == true_label
            })

        accuracy = correct / len(y_test) * 100

        print(f"\n{'='*70}")
        print(f"Model Prediction Results:")
        print(f"  Accuracy: {accuracy:.2f}% ({correct}/{len(y_test)})")
        print(f"{'='*70}")

        self.test_results['model_prediction'] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(y_test)
        }

        return accuracy >= 95.0  # Pass if accuracy >= 95%

    def test_risk_assessment(self):
        """Test risk assessment functionality."""
        print("\n" + "="*70)
        print("TEST 3: RISK ASSESSMENT")
        print("="*70)

        test_cases = [
            {'probability': 0.05, 'expected_level': 'low', 'label': 'Very low risk'},
            {'probability': 0.25, 'expected_level': 'low', 'label': 'Low risk'},
            {'probability': 0.50, 'expected_level': 'medium', 'label': 'Medium risk'},
            {'probability': 0.65, 'expected_level': 'medium', 'label': 'High medium risk'},
            {'probability': 0.85, 'expected_level': 'high', 'label': 'High risk'},
            {'probability': 0.95, 'expected_level': 'high', 'label': 'Very high risk'}
        ]

        passed = 0
        failed = 0

        for test in test_cases:
            result = self.risk_analyzer.assess(test['probability'])

            level_ok = result['level'] == test['expected_level']
            score_ok = abs(result['score'] - int(test['probability'] * 100)) <= 1

            all_ok = level_ok and score_ok

            status = "✓ PASS" if all_ok else "✗ FAIL"

            print(f"\n{status} | {test['label']}")
            print(f"  Probability: {test['probability']:.2f}")
            print(f"  Risk Level: {result['level']} (expected {test['expected_level']}) {'✓' if level_ok else '✗'}")
            print(f"  Risk Score: {result['score']} {'✓' if score_ok else '✗'}")

            if all_ok:
                passed += 1
            else:
                failed += 1

        print(f"\n{'='*70}")
        print(f"Risk Assessment Results: {passed} passed, {failed} failed")
        print(f"{'='*70}")

        self.test_results['risk_assessment'] = {
            'passed': passed,
            'failed': failed
        }

        return passed, failed

    def test_end_to_end(self):
        """Test complete end-to-end pipeline."""
        print("\n" + "="*70)
        print("TEST 4: END-TO-END PIPELINE")
        print("="*70)

        test_cases = [
            {
                'url': 'https://www.google.com',
                'expected_phishing': False,
                'label': 'Google (Legitimate)'
            },
            {
                'url': 'https://www.paypal.com',
                'expected_phishing': False,
                'label': 'PayPal (Legitimate)'
            },
            {
                'url': 'https://www.amazon.com',
                'expected_phishing': False,
                'label': 'Amazon (Legitimate)'
            }
        ]

        passed = 0
        failed = 0

        for test in test_cases:
            # Full pipeline
            features = self.feature_extractor.extract(test['url'], 'url')
            prediction = self.ml_service.predict(features)
            risk = self.risk_analyzer.assess(prediction['probability'])
            response = self.chatbot.generate_response(test['url'], prediction, risk)

            # Validate
            prediction_ok = prediction['is_phishing'] == test['expected_phishing']
            risk_exists = 'level' in risk
            response_exists = len(response) > 0

            all_ok = prediction_ok and risk_exists and response_exists

            status = "✓ PASS" if all_ok else "✗ FAIL"

            print(f"\n{status} | {test['label']}")
            print(f"  URL: {test['url']}")
            print(f"  Is Phishing: {prediction['is_phishing']} (expected {test['expected_phishing']}) {'✓' if prediction_ok else '✗'}")
            print(f"  Confidence: {prediction['confidence']*100:.1f}%")
            print(f"  Risk Level: {risk['level']}")
            print(f"  Risk Score: {risk['score']}")
            print(f"  Response Length: {len(response)} chars")

            if all_ok:
                passed += 1
            else:
                failed += 1

            self.test_results['end_to_end'].append({
                'url': test['url'],
                'passed': all_ok,
                'prediction': prediction['is_phishing'],
                'risk_level': risk['level']
            })

        print(f"\n{'='*70}")
        print(f"End-to-End Results: {passed} passed, {failed} failed")
        print(f"{'='*70}")

        return passed, failed

    def test_model_loaded(self):
        """Test that model is properly loaded."""
        print("\n" + "="*70)
        print("TEST 0: MODEL LOADING")
        print("="*70)

        model_loaded = self.ml_service.is_loaded()

        if model_loaded:
            model_info = self.ml_service.get_model_info()
            print(f"\n✓ Model loaded successfully")
            print(f"  Model Type: {model_info['model_type']}")
            print(f"  Model Name: {model_info['model_name']}")
            print(f"  Features Expected: {model_info['features_expected']}")

            if 'n_estimators' in model_info:
                print(f"  Number of Trees: {model_info['n_estimators']}")
            if 'max_depth' in model_info:
                print(f"  Max Depth: {model_info['max_depth']}")

            print(f"\n✓ Model is ready for predictions")
        else:
            print(f"\n✗ FAIL: Model not loaded!")
            return False

        print(f"\n{'='*70}")
        return True

    def run_all_tests(self):
        """Run all tests."""
        print("\n" + "="*70)
        print("COMPREHENSIVE SYSTEM TEST")
        print("PhishGuard AI - Complete Validation")
        print("="*70)

        all_passed = True

        # Test 0: Model loading
        if not self.test_model_loaded():
            print("\n❌ CRITICAL: Model not loaded. Cannot continue.")
            return False

        # Test 1: Feature extraction
        fe_passed, fe_failed = self.test_feature_extraction()
        if fe_failed > 0:
            all_passed = False

        # Test 2: Model prediction
        mp_passed = self.test_model_prediction()
        if not mp_passed:
            all_passed = False

        # Test 3: Risk assessment
        ra_passed, ra_failed = self.test_risk_assessment()
        if ra_failed > 0:
            all_passed = False

        # Test 4: End-to-end
        e2e_passed, e2e_failed = self.test_end_to_end()
        if e2e_failed > 0:
            all_passed = False

        # Final summary
        print("\n" + "="*70)
        print("FINAL TEST SUMMARY")
        print("="*70)

        print(f"\n✓ Model Loading: PASSED")
        print(f"{'✓' if fe_failed == 0 else '✗'} Feature Extraction: {fe_passed} passed, {fe_failed} failed")
        print(f"{'✓' if mp_passed else '✗'} Model Prediction: {self.test_results['model_prediction']['accuracy']:.2f}% accuracy")
        print(f"{'✓' if ra_failed == 0 else '✗'} Risk Assessment: {ra_passed} passed, {ra_failed} failed")
        print(f"{'✓' if e2e_failed == 0 else '✗'} End-to-End Pipeline: {e2e_passed} passed, {e2e_failed} failed")

        if all_passed:
            print(f"\n{'='*70}")
            print(f"🎉 ALL TESTS PASSED - SYSTEM IS WORKING PERFECTLY!")
            print(f"{'='*70}\n")
        else:
            print(f"\n{'='*70}")
            print(f"⚠️  SOME TESTS FAILED - REVIEW RESULTS ABOVE")
            print(f"{'='*70}\n")

        return all_passed


def main():
    """Run comprehensive system test."""
    tester = ComprehensiveSystemTest()
    success = tester.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
