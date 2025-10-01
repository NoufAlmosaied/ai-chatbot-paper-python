#!/usr/bin/env python3
"""
Phase 4: Phishing Detection Chatbot API
Main Flask application for serving the phishing detection models
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path for model imports
sys.path.append(str(Path(__file__).parent))

from services.ml_service import MLService
from services.feature_extractor import FeatureExtractor
from chatbot.conversation import ChatbotEngine
from chatbot.risk_analyzer import RiskAnalyzer

def create_app():
    """Application factory pattern for better deployment flexibility"""
    app = Flask(__name__)

    # Configuration from environment variables
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    app.config['FLASK_ENV'] = os.getenv('FLASK_ENV', 'production')

    # CORS configuration
    cors_origins = os.getenv('CORS_ORIGINS', '*')
    if cors_origins == '*':
        CORS(app)
    else:
        CORS(app, origins=cors_origins.split(','))

    return app

# Initialize Flask app
app = create_app()

# Initialize services with error handling
try:
    ml_service = MLService()
    feature_extractor = FeatureExtractor()
    chatbot = ChatbotEngine()
    risk_analyzer = RiskAnalyzer()
    logger.info("All services initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize services: {e}")
    raise

@app.route('/')
def home():
    """Serve the chatbot interface."""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        model_loaded = ml_service.is_loaded()
        return jsonify({
            'status': 'healthy' if model_loaded else 'unhealthy',
            'model_loaded': model_loaded,
            'version': '1.0.0',
            'environment': os.getenv('FLASK_ENV', 'unknown'),
            'timestamp': __import__('datetime').datetime.utcnow().isoformat()
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'version': '1.0.0'
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Main endpoint for phishing detection.
    Accepts URL or email text for analysis.
    """
    try:
        data = request.get_json()
        
        if not data or 'content' not in data:
            return jsonify({
                'error': 'No content provided',
                'message': 'Please provide URL or email content to analyze'
            }), 400
        
        content = data['content']
        content_type = data.get('type', 'url')  # Default to URL
        
        # Extract features from content
        features = feature_extractor.extract(content, content_type)
        
        if features is None:
            return jsonify({
                'error': 'Feature extraction failed',
                'message': 'Could not extract features from the provided content'
            }), 400
        
        # Get prediction from ML model
        prediction = ml_service.predict(features)
        
        # Analyze risk level
        risk_assessment = risk_analyzer.assess(prediction['probability'])
        
        # Generate chatbot response
        response = chatbot.generate_response(
            content=content,
            prediction=prediction,
            risk_assessment=risk_assessment
        )
        
        return jsonify({
            'success': True,
            'analysis': {
                'content': content,
                'type': content_type,
                'is_phishing': prediction['is_phishing'],
                'confidence': prediction['probability'],
                'risk_level': risk_assessment['level'],
                'risk_score': risk_assessment['score'],
                'indicators': prediction.get('top_features', []),
                'recommendation': risk_assessment['recommendation']
            },
            'chatbot_response': response
        })
        
    except Exception as e:
        app.logger.error(f"Analysis error: {str(e)}")
        return jsonify({
            'error': 'Analysis failed',
            'message': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat endpoint for conversational interaction.
    """
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({
                'error': 'No message provided'
            }), 400
        
        # Process chat message
        response = chatbot.process_message(message)

        # If URL/email detected in message, analyze it
        if response.get('requires_analysis'):
            content = response['extracted_content']
            features = feature_extractor.extract(content, response['content_type'])

            if features is not None:
                prediction = ml_service.predict(features)
                risk_assessment = risk_analyzer.assess(prediction['probability'])

                response['analysis'] = {
                    'is_phishing': prediction['is_phishing'],
                    'confidence': prediction['probability'],
                    'risk_level': risk_assessment['level'],
                    'risk_score': risk_assessment['score'],
                    'recommendation': risk_assessment['recommendation']
                }

        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Chat error: {str(e)}")
        return jsonify({
            'error': 'Chat processing failed',
            'message': str(e)
        }), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    """
    Collect user feedback on predictions.
    """
    try:
        data = request.get_json()
        
        # Log feedback for model improvement
        feedback = {
            'content': data.get('content'),
            'predicted': data.get('predicted'),
            'actual': data.get('actual'),
            'user_comment': data.get('comment')
        }
        
        # In production, save to database
        app.logger.info(f"Feedback received: {feedback}")
        
        return jsonify({
            'success': True,
            'message': 'Thank you for your feedback!'
        })
        
    except Exception as e:
        app.logger.error(f"Feedback error: {str(e)}")
        return jsonify({
            'error': 'Feedback submission failed'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get detection statistics.
    """
    try:
        stats = ml_service.get_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({
            'error': 'Failed to retrieve statistics'
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    # Development server (production uses gunicorn)
    port = int(os.getenv('PORT', 5001))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)