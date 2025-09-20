#!/bin/bash
# Build script for Render.com deployment

echo "🚀 Starting PhishGuard AI build process..."

# Set working directory
cd /opt/render/project/src/scripts/phase4_chatbot

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Verify critical files exist
echo "📋 Verifying deployment files..."
required_files=("app.py" "gunicorn.conf.py" "requirements.txt")

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "   ✅ $file found"
    else
        echo "   ❌ $file missing!"
        exit 1
    fi
done

# Test import of main application
echo "🧪 Testing application imports..."
python -c "
import sys
sys.path.append('/opt/render/project/src')
try:
    from scripts.phase4_chatbot.app import app
    print('   ✅ App imports successfully')
except Exception as e:
    print(f'   ❌ App import failed: {e}')
    sys.exit(1)
"

# Check model files exist (they should be in the git repo)
echo "🤖 Checking ML model availability..."
if [[ -f "../models/baseline/random_forest.pkl" ]]; then
    echo "   ✅ ML model found"
else
    echo "   ⚠️  ML model not found at expected path"
    echo "   📁 Available files in models directory:"
    find .. -name "*.pkl" -type f 2>/dev/null || echo "   No .pkl files found"
fi

echo "✅ Build completed successfully!"
echo "🌐 Ready for production deployment!"