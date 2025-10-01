#!/bin/bash
# Live API Testing Script

echo "=========================================="
echo "LIVE API TESTING"
echo "=========================================="
echo ""

# Test 1: Health Check
echo "1. Health Check"
echo "----------------"
curl -s http://localhost:5001/api/health | python3 -m json.tool | grep -E "(status|model_loaded)"
echo ""

# Test 2: Google.com
echo "2. Test: https://www.google.com (Should be NOT phishing)"
echo "--------------------------------------------------------"
curl -s -X POST http://localhost:5001/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"content":"https://www.google.com","type":"url"}' | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Is Phishing: {d[\"analysis\"][\"is_phishing\"]}'); print(f'  Confidence: {d[\"analysis\"][\"confidence\"]*100:.1f}%'); print(f'  Risk Level: {d[\"analysis\"][\"risk_level\"]}'); print(f'  Risk Score: {d[\"analysis\"][\"risk_score\"]}')"
echo ""

# Test 3: Amazon.com
echo "3. Test: https://www.amazon.com (Should be NOT phishing)"
echo "---------------------------------------------------------"
curl -s -X POST http://localhost:5001/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"content":"https://www.amazon.com","type":"url"}' | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Is Phishing: {d[\"analysis\"][\"is_phishing\"]}'); print(f'  Confidence: {d[\"analysis\"][\"confidence\"]*100:.1f}%'); print(f'  Risk Level: {d[\"analysis\"][\"risk_level\"]}'); print(f'  Risk Score: {d[\"analysis\"][\"risk_score\"]}')"
echo ""

# Test 4: PayPal.com
echo "4. Test: https://www.paypal.com (Should be NOT phishing)"
echo "---------------------------------------------------------"
curl -s -X POST http://localhost:5001/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"content":"https://www.paypal.com","type":"url"}' | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Is Phishing: {d[\"analysis\"][\"is_phishing\"]}'); print(f'  Confidence: {d[\"analysis\"][\"confidence\"]*100:.1f}%'); print(f'  Risk Level: {d[\"analysis\"][\"risk_level\"]}'); print(f'  Risk Score: {d[\"analysis\"][\"risk_score\"]}')"
echo ""

echo "=========================================="
echo "All API tests completed!"
echo "=========================================="
