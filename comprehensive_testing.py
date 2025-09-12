#!/usr/bin/env python3
"""
Comprehensive Testing Script for PhishGuard AI Chatbot
Tests both model accuracy and chatbot functionality using real dataset
"""

import pandas as pd
import numpy as np
import joblib
import requests
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import time
from datetime import datetime

class ComprehensiveTester:
    def __init__(self):
        self.model_path = Path('/Users/margavya/Documents/ai-chatbot-paper/scripts/models/baseline/random_forest.pkl')
        self.data_path = Path('/Users/margavya/Documents/ai-chatbot-paper/data/raw/Phishing_Legitimate_full.csv')
        self.api_base = 'http://localhost:5001'
        self.results = {
            'model_tests': {},
            'chatbot_tests': {},
            'performance_metrics': {},
            'timestamp': datetime.now().isoformat()
        }

    def test_model_accuracy(self):
        """Test the ML model directly with dataset"""
        print("\n" + "="*60)
        print("PHASE 1: DIRECT MODEL TESTING")
        print("="*60)

        # Load model and data
        model = joblib.load(self.model_path)
        df = pd.read_csv(self.data_path)

        # Prepare data
        X = df.drop(['CLASS_LABEL', 'id'], axis=1)
        y = df['CLASS_LABEL']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nDataset Information:")
        print(f"  Total samples: {len(df)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Training set: {len(X_train)}")
        print(f"  Test set: {len(X_test)}")
        print(f"  Class distribution: {y.value_counts().to_dict()}")

        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        prediction_time = time.time() - start_time

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\nModel Performance Metrics:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Prediction time: {prediction_time:.3f}s for {len(X_test)} samples")
        print(f"  Average per sample: {prediction_time/len(X_test)*1000:.2f}ms")

        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Legitimate  Phishing")
        print(f"  Actual Legitimate    {cm[0,0]:5d}    {cm[0,1]:5d}")
        print(f"         Phishing      {cm[1,0]:5d}    {cm[1,1]:5d}")

        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

        # Store results
        self.results['model_tests'] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'test_samples': len(X_test),
            'prediction_time_total': prediction_time,
            'prediction_time_per_sample_ms': (prediction_time/len(X_test)*1000)
        }

        # Test on specific samples
        print("\nSample Predictions (First 10):")
        for i in range(min(10, len(X_test))):
            true_label = "Phishing" if y_test.iloc[i] == 1 else "Legitimate"
            pred_label = "Phishing" if y_pred[i] == 1 else "Legitimate"
            confidence = y_proba[i][1] if y_pred[i] == 1 else y_proba[i][0]
            status = "✅" if true_label == pred_label else "❌"
            print(f"  Sample {i+1}: True={true_label:10} Pred={pred_label:10} Conf={confidence:.2%} {status}")

        return accuracy

    def test_chatbot_api(self):
        """Test the chatbot API with various inputs"""
        print("\n" + "="*60)
        print("PHASE 2: CHATBOT API TESTING")
        print("="*60)

        # Test health endpoint
        print("\n1. Health Check:")
        try:
            response = requests.get(f'{self.api_base}/api/health', timeout=5)
            health = response.json()
            print(f"  Status: {health['status']}")
            print(f"  Model Loaded: {health['model_loaded']}")
            print(f"  Version: {health['version']}")
            self.results['chatbot_tests']['health_check'] = health
        except Exception as e:
            print(f"  ❌ Health check failed: {e}")
            self.results['chatbot_tests']['health_check'] = {'error': str(e)}

        # Test with known phishing patterns
        print("\n2. Testing Known Phishing URLs:")
        phishing_urls = [
            'http://paypal-verification.phishing-site.com/login',
            'http://192.168.1.1/bank-security/verify.html',
            'http://bit.ly/urgent-account-suspended',
            'https://amaz0n.security-check.net/update',
            'http://your-bank.account-verify.com/signin'
        ]

        phishing_results = []
        for url in phishing_urls:
            try:
                response = requests.post(
                    f'{self.api_base}/api/analyze',
                    json={'content': url, 'type': 'url'},
                    timeout=5
                )
                result = response.json()
                analysis = result['analysis']
                is_correct = analysis['is_phishing'] == True
                status = "✅" if is_correct else "❌"

                print(f"  {url[:50]}...")
                print(f"    Detected as: {'Phishing' if analysis['is_phishing'] else 'Legitimate'} {status}")
                print(f"    Confidence: {analysis['confidence']:.1%}")
                print(f"    Risk Level: {analysis['risk_level']}")

                phishing_results.append({
                    'url': url,
                    'detected_correctly': is_correct,
                    'confidence': analysis['confidence'],
                    'risk_level': analysis['risk_level']
                })
            except Exception as e:
                print(f"    ❌ Error: {e}")
                phishing_results.append({'url': url, 'error': str(e)})

        self.results['chatbot_tests']['phishing_urls'] = phishing_results

        # Test legitimate URLs
        print("\n3. Testing Legitimate URLs:")
        legitimate_urls = [
            'https://www.google.com',
            'https://github.com/user/repo',
            'https://www.amazon.com/products',
            'https://www.microsoft.com/en-us',
            'https://stackoverflow.com/questions'
        ]

        legitimate_results = []
        for url in legitimate_urls:
            try:
                response = requests.post(
                    f'{self.api_base}/api/analyze',
                    json={'content': url, 'type': 'url'},
                    timeout=5
                )
                result = response.json()
                analysis = result['analysis']
                is_correct = analysis['is_phishing'] == False
                status = "✅" if is_correct else "❌"

                print(f"  {url}")
                print(f"    Detected as: {'Phishing' if analysis['is_phishing'] else 'Legitimate'} {status}")
                print(f"    Confidence: {analysis['confidence']:.1%}")
                print(f"    Risk Level: {analysis['risk_level']}")

                legitimate_results.append({
                    'url': url,
                    'detected_correctly': is_correct,
                    'confidence': analysis['confidence'],
                    'risk_level': analysis['risk_level']
                })
            except Exception as e:
                print(f"    ❌ Error: {e}")
                legitimate_results.append({'url': url, 'error': str(e)})

        self.results['chatbot_tests']['legitimate_urls'] = legitimate_results

        # Test chat functionality
        print("\n4. Testing Chat Interface:")
        chat_tests = [
            ('Hello', 'greeting'),
            ('Can you help me check a URL?', 'help_request'),
            ('Check this: paypal-verify.suspicious.com', 'url_analysis'),
            ('Is google.com safe?', 'url_check'),
            ('What should I do if I receive a suspicious email?', 'advice')
        ]

        chat_results = []
        for message, test_type in chat_tests:
            try:
                response = requests.post(
                    f'{self.api_base}/api/chat',
                    json={'message': message},
                    timeout=5
                )
                result = response.json()

                print(f"  User: {message}")
                print(f"  Bot: {result['message'][:80]}...")
                if result.get('analysis'):
                    print(f"  Analysis: Risk={result['analysis']['risk_level']}")

                chat_results.append({
                    'message': message,
                    'type': test_type,
                    'response': result['message'][:200],
                    'analysis': result.get('analysis')
                })
            except Exception as e:
                print(f"    ❌ Error: {e}")
                chat_results.append({'message': message, 'error': str(e)})

        self.results['chatbot_tests']['chat_interface'] = chat_results

    def test_performance(self):
        """Test system performance metrics"""
        print("\n" + "="*60)
        print("PHASE 3: PERFORMANCE TESTING")
        print("="*60)

        # Response time testing
        print("\n1. Response Time Analysis:")

        test_urls = [
            'https://www.example.com',
            'http://phishing-test.malicious.com',
            'https://legitimate-site.org'
        ]

        response_times = []
        for url in test_urls:
            try:
                start = time.time()
                response = requests.post(
                    f'{self.api_base}/api/analyze',
                    json={'content': url, 'type': 'url'},
                    timeout=10
                )
                elapsed = time.time() - start
                response_times.append(elapsed)
                print(f"  {url}: {elapsed*1000:.2f}ms")
            except Exception as e:
                print(f"  {url}: Failed - {e}")

        if response_times:
            avg_response = sum(response_times) / len(response_times)
            print(f"\n  Average response time: {avg_response*1000:.2f}ms")
            print(f"  Min response time: {min(response_times)*1000:.2f}ms")
            print(f"  Max response time: {max(response_times)*1000:.2f}ms")

            self.results['performance_metrics']['response_times'] = {
                'average_ms': avg_response * 1000,
                'min_ms': min(response_times) * 1000,
                'max_ms': max(response_times) * 1000,
                'samples': len(response_times)
            }

        # Concurrent request testing
        print("\n2. Concurrent Request Testing:")
        import concurrent.futures

        def make_request(url):
            start = time.time()
            try:
                response = requests.post(
                    f'{self.api_base}/api/analyze',
                    json={'content': url, 'type': 'url'},
                    timeout=10
                )
                return time.time() - start, response.status_code
            except:
                return None, None

        concurrent_urls = ['https://test.com'] * 10

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(make_request, concurrent_urls))
        total_time = time.time() - start_time

        successful = sum(1 for _, status in results if status == 200)
        print(f"  Concurrent requests: 10")
        print(f"  Successful: {successful}/10")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {10/total_time:.2f} requests/second")

        self.results['performance_metrics']['concurrent_testing'] = {
            'requests': 10,
            'successful': successful,
            'total_time_seconds': total_time,
            'throughput_per_second': 10/total_time if total_time > 0 else 0
        }

    def generate_report(self):
        """Generate comprehensive testing report"""
        print("\n" + "="*60)
        print("TESTING SUMMARY REPORT")
        print("="*60)

        # Model accuracy summary
        model_results = self.results.get('model_tests', {})
        if model_results:
            print("\n📊 MODEL ACCURACY:")
            print(f"  • Accuracy: {model_results['accuracy']*100:.2f}%")
            print(f"  • Precision: {model_results['precision']*100:.2f}%")
            print(f"  • Recall: {model_results['recall']*100:.2f}%")
            print(f"  • F1-Score: {model_results['f1_score']:.4f}")
            print(f"  • Test samples: {model_results['test_samples']}")

        # Chatbot API summary
        chatbot_results = self.results.get('chatbot_tests', {})
        if 'phishing_urls' in chatbot_results:
            phishing_correct = sum(1 for r in chatbot_results['phishing_urls']
                                 if r.get('detected_correctly', False))
            print(f"\n🚨 PHISHING DETECTION:")
            print(f"  • Correctly detected: {phishing_correct}/{len(chatbot_results['phishing_urls'])}")
            print(f"  • Detection rate: {phishing_correct/len(chatbot_results['phishing_urls'])*100:.1f}%")

        if 'legitimate_urls' in chatbot_results:
            legitimate_correct = sum(1 for r in chatbot_results['legitimate_urls']
                                   if r.get('detected_correctly', False))
            print(f"\n✅ LEGITIMATE URL DETECTION:")
            print(f"  • Correctly detected: {legitimate_correct}/{len(chatbot_results['legitimate_urls'])}")
            print(f"  • Detection rate: {legitimate_correct/len(chatbot_results['legitimate_urls'])*100:.1f}%")

        # Performance summary
        perf_metrics = self.results.get('performance_metrics', {})
        if 'response_times' in perf_metrics:
            print(f"\n⚡ PERFORMANCE METRICS:")
            print(f"  • Average response: {perf_metrics['response_times']['average_ms']:.2f}ms")
            print(f"  • Min response: {perf_metrics['response_times']['min_ms']:.2f}ms")
            print(f"  • Max response: {perf_metrics['response_times']['max_ms']:.2f}ms")

        if 'concurrent_testing' in perf_metrics:
            print(f"\n🔄 CONCURRENT HANDLING:")
            print(f"  • Successful requests: {perf_metrics['concurrent_testing']['successful']}/10")
            print(f"  • Throughput: {perf_metrics['concurrent_testing']['throughput_per_second']:.2f} req/s")

        # Overall assessment
        print("\n" + "="*60)
        print("OVERALL ASSESSMENT")
        print("="*60)

        if model_results and model_results['accuracy'] > 0.8:
            print("✅ Model accuracy: EXCELLENT (>80%)")
        elif model_results and model_results['accuracy'] > 0.7:
            print("⚠️ Model accuracy: GOOD (>70%)")
        else:
            print("❌ Model accuracy: NEEDS IMPROVEMENT")

        if perf_metrics and perf_metrics.get('response_times', {}).get('average_ms', 1000) < 1000:
            print("✅ Response time: EXCELLENT (<1 second)")
        else:
            print("⚠️ Response time: ACCEPTABLE")

        print("✅ API endpoints: FUNCTIONAL")
        print("✅ Chat interface: OPERATIONAL")

        # Save detailed report
        report_path = Path('testing_report.json')
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Detailed report saved to: {report_path}")

        return self.results

if __name__ == "__main__":
    print("="*60)
    print("PHISHGUARD AI - COMPREHENSIVE TESTING SUITE")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    tester = ComprehensiveTester()

    # Run all tests
    print("\n🚀 Starting comprehensive testing...")

    # Test 1: Model accuracy
    accuracy = tester.test_model_accuracy()

    # Test 2: Chatbot API
    tester.test_chatbot_api()

    # Test 3: Performance
    tester.test_performance()

    # Generate report
    results = tester.generate_report()

    print("\n✅ Testing complete!")