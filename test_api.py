#!/usr/bin/env python3
"""
Test script for the Phase 4 chatbot API
"""

import requests
import json
import time
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

def test_health_endpoint():
    """Test the health check endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        print(f"Health Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_analyze_endpoint():
    """Test the analyze endpoint with sample data."""
    print("\nTesting analyze endpoint...")
    
    test_cases = [
        {
            "name": "Suspicious URL",
            "data": {
                "content": "bit.ly/free-prize-winner",
                "type": "url"
            }
        },
        {
            "name": "Legitimate URL", 
            "data": {
                "content": "https://www.google.com",
                "type": "url"
            }
        },
        {
            "name": "Phishing email content",
            "data": {
                "content": "Urgent: Your account has been suspended. Click here to verify immediately or lose access forever.",
                "type": "email"
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['name']} ---")
        try:
            response = requests.post(
                'http://localhost:5000/api/analyze',
                json=test_case['data'],
                timeout=10
            )
            
            print(f"Status: {response.status_code}")
            result = response.json()
            
            if response.status_code == 200:
                analysis = result['analysis']
                print(f"Is Phishing: {analysis['is_phishing']}")
                print(f"Confidence: {analysis['confidence']:.2f}")
                print(f"Risk Level: {analysis['risk_level']}")
                print(f"Recommendation: {analysis['recommendation']}")
            else:
                print(f"Error: {result}")
                
        except Exception as e:
            print(f"Test failed: {e}")

def test_chat_endpoint():
    """Test the chat endpoint."""
    print("\nTesting chat endpoint...")
    
    test_messages = [
        "Hello",
        "Help me check this URL: phishing-site.suspicious.com",
        "How do I know if an email is phishing?"
    ]
    
    for message in test_messages:
        print(f"\n--- Testing message: '{message}' ---")
        try:
            response = requests.post(
                'http://localhost:5000/api/chat',
                json={"message": message},
                timeout=10
            )
            
            print(f"Status: {response.status_code}")
            result = response.json()
            
            if response.status_code == 200:
                print(f"Bot Response: {result['message']}")
                if result.get('analysis'):
                    print(f"Analysis included: {result['analysis']}")
            else:
                print(f"Error: {result}")
                
        except Exception as e:
            print(f"Test failed: {e}")

def load_test():
    """Perform simple load test."""
    print("\nPerforming load test (10 concurrent requests)...")
    
    import threading
    import time
    
    results = []
    start_time = time.time()
    
    def make_request():
        try:
            response = requests.post(
                'http://localhost:5000/api/analyze',
                json={
                    "content": "https://suspicious-phishing-site.fake.com/verify-account",
                    "type": "url"
                },
                timeout=10
            )
            results.append(response.status_code == 200)
        except:
            results.append(False)
    
    # Create threads
    threads = []
    for i in range(10):
        thread = threading.Thread(target=make_request)
        threads.append(thread)
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    success_rate = sum(results) / len(results) * 100
    total_time = end_time - start_time
    
    print(f"Load test results:")
    print(f"  Total requests: {len(results)}")
    print(f"  Successful: {sum(results)}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average response time: {total_time/len(results):.2f} seconds")

if __name__ == "__main__":
    print("🤖 PhishGuard AI Chatbot - API Testing Suite")
    print("=" * 50)
    
    # Test if server is running
    if not test_health_endpoint():
        print("\n❌ Server not running. Please start the Flask app first:")
        print("   cd scripts/phase4_chatbot")
        print("   python3 app.py")
        sys.exit(1)
    
    # Run all tests
    test_analyze_endpoint()
    test_chat_endpoint()
    load_test()
    
    print("\n✅ All tests completed!")
    print("\nTo test the web interface:")
    print("1. Start the Flask app: python3 app.py")
    print("2. Open browser to: http://localhost:5000")