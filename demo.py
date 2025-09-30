#!/usr/bin/env python3
"""
Demo script for Phase 4 Chatbot functionality
Tests the core components without running the web server
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory for imports
sys.path.append('..')

from services.ml_service import MLService
from services.feature_extractor import FeatureExtractor
from chatbot.conversation import ChatbotEngine
from chatbot.risk_analyzer import RiskAnalyzer

def demo_feature_extraction():
    """Demo feature extraction from URLs."""
    print("🔧 Testing Feature Extraction")
    print("=" * 40)
    
    extractor = FeatureExtractor()
    
    test_urls = [
        "https://www.google.com",
        "http://bit.ly/free-gift-winner",
        "https://paypal-security-verify.suspicious.com/login",
        "192.168.1.1/phishing-page"
    ]
    
    for url in test_urls:
        print(f"\nURL: {url}")
        features = extractor.extract(url, 'url')
        if features is not None:
            print(f"  Features extracted: {features.shape}")
            print(f"  Sample features: {features[:5]}")
        else:
            print("  ❌ Feature extraction failed")

def demo_risk_analysis():
    """Demo risk analysis."""
    print("\n🎯 Testing Risk Analysis")
    print("=" * 40)
    
    analyzer = RiskAnalyzer()
    
    test_probabilities = [0.1, 0.5, 0.9]
    
    for prob in test_probabilities:
        print(f"\nPhishing Probability: {prob}")
        assessment = analyzer.assess(prob)
        print(f"  Risk Level: {assessment['level']}")
        print(f"  Risk Score: {assessment['score']}")
        print(f"  Recommendation: {assessment['recommendation']}")

def demo_chatbot():
    """Demo chatbot conversation."""
    print("\n🤖 Testing Chatbot Engine")
    print("=" * 40)
    
    chatbot = ChatbotEngine()
    
    test_messages = [
        "Hello",
        "Help",
        "Check this URL: bit.ly/suspicious-link",
        "Is this phishing: Your account will be suspended unless you verify immediately"
    ]
    
    for message in test_messages:
        print(f"\nUser: {message}")
        response = chatbot.process_message(message)
        print(f"Bot: {response['message']}")
        if response.get('requires_analysis'):
            print(f"  -> Analysis needed for: {response['extracted_content']}")

def demo_ml_service():
    """Demo ML service with synthetic data."""
    print("\n🧠 Testing ML Service")
    print("=" * 40)
    
    # Try to load ML service
    try:
        ml_service = MLService()
        
        if ml_service.is_loaded():
            print("✅ ML Model loaded successfully")
            print(f"Model info: {ml_service.get_model_info()}")
            
            # Test with synthetic features (48 features)
            synthetic_features = np.random.rand(48)
            prediction = ml_service.predict(synthetic_features)
            
            print(f"\nTest Prediction:")
            print(f"  Is Phishing: {prediction['is_phishing']}")
            print(f"  Confidence: {prediction['confidence']:.3f}")
            print(f"  Model Used: {prediction['model_used']}")
            
        else:
            print("❌ ML Model not loaded")
            
    except Exception as e:
        print(f"❌ ML Service error: {e}")
        print("Note: This is expected if no trained models are available")

def demo_full_pipeline():
    """Demo the complete pipeline."""
    print("\n🚀 Testing Complete Pipeline")
    print("=" * 40)
    
    try:
        # Initialize components
        extractor = FeatureExtractor()
        ml_service = MLService()
        analyzer = RiskAnalyzer()
        chatbot = ChatbotEngine()
        
        test_url = "http://phishing-site.suspicious.com/verify-account"
        
        print(f"Testing URL: {test_url}")
        
        # Extract features
        features = extractor.extract(test_url, 'url')
        if features is None:
            print("❌ Feature extraction failed")
            return
        
        print(f"✅ Features extracted: {features.shape}")
        
        # Make prediction (if model available)
        if ml_service.is_loaded():
            prediction = ml_service.predict(features)
            print(f"✅ Prediction: {prediction['is_phishing']} (confidence: {prediction['confidence']:.3f})")
            
            # Analyze risk
            risk_assessment = analyzer.assess(prediction['probability'])
            print(f"✅ Risk Level: {risk_assessment['level']} (score: {risk_assessment['score']})")
            
            # Generate chatbot response
            response = chatbot.generate_response(test_url, prediction, risk_assessment)
            print(f"✅ Chatbot response generated ({len(response)} chars)")
            print(f"\nBot Response Preview:")
            print(response[:200] + "..." if len(response) > 200 else response)
            
        else:
            print("⚠️ ML model not available - using synthetic prediction")
            synthetic_prediction = {
                'is_phishing': True,
                'probability': 0.85,
                'confidence': 0.85,
                'model_used': 'synthetic',
                'top_features': []
            }
            risk_assessment = analyzer.assess(synthetic_prediction['probability'])
            response = chatbot.generate_response(test_url, synthetic_prediction, risk_assessment)
            print(f"✅ Synthetic response generated")
            print(f"\nBot Response Preview:")
            print(response[:200] + "..." if len(response) > 200 else response)
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🤖 PhishGuard AI - Phase 4 Demo")
    print("=" * 50)
    
    # Run all demos
    demo_feature_extraction()
    demo_risk_analysis()
    demo_chatbot()
    demo_ml_service()
    demo_full_pipeline()
    
    print("\n" + "=" * 50)
    print("✅ Demo completed!")
    print("\nNext steps:")
    print("1. Start the web server: python3 app.py")
    print("2. Open browser to: http://localhost:5000")
    print("3. Test the chatbot interface!")