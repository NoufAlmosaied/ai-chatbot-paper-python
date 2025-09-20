#!/bin/bash
# PhishGuard AI Chatbot Startup Script

echo "🤖 PhishGuard AI - Starting Chatbot Interface"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run from the phase4_chatbot directory"
    exit 1
fi

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 not found. Please install Python 3"
    exit 1
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
python3 -c "import flask, flask_cors, numpy, pandas, sklearn, joblib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Installing missing dependencies..."
    pip3 install flask flask-cors numpy pandas scikit-learn joblib
fi

# Check if models are available
echo "🧠 Checking ML models..."
if [ -f "../models/baseline/random_forest.pkl" ]; then
    echo "✅ Random Forest model found"
elif [ -f "../models/baseline/logistic_regression.pkl" ]; then
    echo "✅ Logistic Regression model found"
else
    echo "⚠️  No trained models found. Run Phase 3 first:"
    echo "   cd ../scripts && python3 run_phase3_pipeline.py"
fi

# Test components
echo "🔧 Testing components..."
python3 -c "
import sys
sys.path.append('..')
try:
    from services.ml_service import MLService
    from services.feature_extractor import FeatureExtractor
    from chatbot.conversation import ChatbotEngine
    from chatbot.risk_analyzer import RiskAnalyzer
    print('✅ All components imported successfully')
except Exception as e:
    print(f'❌ Component test failed: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Component test failed. Please check the installation."
    exit 1
fi

# Start the server
echo ""
echo "🚀 Starting Flask server..."
echo "📱 Web interface will be available at: http://localhost:5000"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

python3 app.py