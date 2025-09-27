#!/usr/bin/env python3
"""
Quick deployment test script
Tests the application locally to ensure it's ready for production deployment
"""

import requests
import json
import time
import subprocess
import threading
import sys
from pathlib import Path

def start_local_server():
    """Start the local Flask server"""
    try:
        # Change to the correct directory
        import os
        os.chdir(Path(__file__).parent)

        # Start gunicorn server
        process = subprocess.Popen([
            'gunicorn', '--config', 'gunicorn.conf.py', 'app:app',
            '--bind', '0.0.0.0:8080'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for server to start
        time.sleep(5)

        return process
    except Exception as e:
        print(f"Failed to start server: {e}")
        return None

def test_endpoints():
    """Test all API endpoints"""
    base_url = "http://localhost:8080"

    tests = [
        {
            'name': 'Health Check',
            'url': f'{base_url}/api/health',
            'method': 'GET',
            'expected_status': 200
        },
        {
            'name': 'Analyze Phishing URL',
            'url': f'{base_url}/api/analyze',
            'method': 'POST',
            'data': {'content': 'http://phishing-site.suspicious.com/login', 'type': 'url'},
            'expected_status': 200
        },
        {
            'name': 'Analyze Legitimate URL',
            'url': f'{base_url}/api/analyze',
            'method': 'POST',
            'data': {'content': 'https://www.google.com', 'type': 'url'},
            'expected_status': 200
        },
        {
            'name': 'Chat Interface',
            'url': f'{base_url}/api/chat',
            'method': 'POST',
            'data': {'message': 'Hello, can you help me check a URL?'},
            'expected_status': 200
        }
    ]

    results = []

    for test in tests:
        try:
            print(f"\n🧪 Testing: {test['name']}")

            if test['method'] == 'GET':
                response = requests.get(test['url'], timeout=10)
            else:
                response = requests.post(
                    test['url'],
                    json=test['data'],
                    timeout=10
                )

            status = response.status_code
            success = status == test['expected_status']

            print(f"   Status: {status} {'✅' if success else '❌'}")

            if success and response.headers.get('content-type', '').startswith('application/json'):
                data = response.json()
                print(f"   Response: {json.dumps(data, indent=2)[:200]}...")

            results.append({
                'test': test['name'],
                'success': success,
                'status': status,
                'response_time': response.elapsed.total_seconds()
            })

        except Exception as e:
            print(f"   Error: {e} ❌")
            results.append({
                'test': test['name'],
                'success': False,
                'error': str(e)
            })

    return results

def print_summary(results):
    """Print test summary"""
    print("\n" + "="*60)
    print("DEPLOYMENT READINESS TEST SUMMARY")
    print("="*60)

    passed = sum(1 for r in results if r['success'])
    total = len(results)

    for result in results:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"  {result['test']:<25} {status}")
        if 'response_time' in result:
            print(f"    Response time: {result['response_time']:.3f}s")
        if 'error' in result:
            print(f"    Error: {result['error']}")

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Application is ready for deployment.")
        print("   You can now deploy to Render.com using the dashboard or CLI.")
    else:
        print("\n⚠️  Some tests failed. Please fix issues before deploying.")

    return passed == total

def main():
    """Main test function"""
    print("PhishGuard AI - Deployment Readiness Test")
    print("="*50)

    # Check if required files exist
    required_files = [
        'app.py', 'requirements.txt', 'gunicorn.conf.py',
        'Procfile', 'render.yaml'
    ]

    print("\n📋 Checking deployment files...")
    for file in required_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - Missing!")
            return False

    # Start local server
    print("\n🚀 Starting local test server...")
    server = start_local_server()

    if not server:
        print("❌ Failed to start local server")
        return False

    try:
        # Wait for server to be ready
        print("   Waiting for server to initialize...")
        time.sleep(3)

        # Test endpoints
        results = test_endpoints()

        # Print summary
        success = print_summary(results)

        return success

    finally:
        # Cleanup
        if server:
            server.terminate()
            server.wait()
            print("\n🛑 Test server stopped")

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)