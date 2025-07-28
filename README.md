# PhishGuard AI - AI-Powered Phishing Detection Chatbot

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An intelligent chatbot system for real-time phishing detection using machine learning. Built with Flask, scikit-learn, and deployed on Render.com.

## 🎯 Features

- **Real-time Phishing Detection**: Analyzes URLs and email content instantly using ML models
- **98.4% Accuracy**: Powered by Random Forest classifier trained on comprehensive phishing datasets
- **Interactive Chat Interface**: User-friendly web interface for natural conversations
- **Tiered Risk Assessment**: Low/Medium/High risk levels with detailed explanations
- **RESTful API**: Programmatic access for integration with other systems
- **Production-Ready**: Optimized for deployment with gunicorn WSGI server

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/NoufAlmosaied/ai-chatbot-paper-python.git
cd ai-chatbot-paper-python
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
python app.py
```

4. **Access the chatbot**
Open your browser and navigate to: `http://localhost:5001`

### Using Docker (Optional)

```bash
docker build -t phishguard-ai .
docker run -p 5001:5001 phishguard-ai
```

## 📊 Project Structure

```
ai-chatbot-paper-python/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── gunicorn.conf.py           # Production WSGI configuration
├── render.yaml                # Render.com deployment config
├── Procfile                   # Process configuration
│
├── services/                  # Backend services
│   ├── ml_service.py         # ML model management
│   └── feature_extractor.py  # Feature extraction from URLs/emails
│
├── chatbot/                   # Chatbot engine
│   ├── conversation.py       # Conversation management
│   └── risk_analyzer.py      # Risk assessment logic
│
├── templates/                 # Web interface
│   └── index.html            # Chat interface UI
│
├── models/                    # Trained ML models
│   └── baseline/
│       ├── random_forest.pkl  # Best performing model (98.4%)
│       ├── logistic_regression.pkl
│       └── svm_rbf.pkl
│
├── scripts/                   # Data processing & training scripts
│   ├── data_preprocessing.py
│   ├── models/
│   │   ├── baseline_models.py
│   │   └── ensemble_model.py
│   └── run_phase3_pipeline.py
│
├── data/                      # Datasets (not included in repo)
│   └── raw/
│
├── docs/                      # Research paper and documentation
│   ├── main.tex
│   ├── main.pdf
│   └── chapters/
│
└── tests/                     # Test suites
    ├── test_api.py
    ├── test_comprehensive.py
    └── test_deployment.py
```

## 🔧 API Endpoints

### Health Check
```http
GET /api/health
```
Returns service health status and model information.

### Analyze Content
```http
POST /api/analyze
Content-Type: application/json

{
  "content": "http://suspicious-link.com/verify-account",
  "type": "url"
}
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "is_phishing": true,
    "confidence": 0.92,
    "risk_level": "high",
    "risk_score": 92,
    "recommendation": "⚠️ HIGH RISK: Strong phishing indicators detected. Do not click this link..."
  }
}
```

### Chat Interface
```http
POST /api/chat
Content-Type: application/json

{
  "message": "Is this link safe: bit.ly/account-verify"
}
```

### Submit Feedback
```http
POST /api/feedback
Content-Type: application/json

{
  "content": "http://example.com",
  "predicted": "phishing",
  "actual": "legitimate",
  "comment": "False positive"
}
```

## 🎓 Research Background

This project is part of academic research on AI-powered phishing detection systems. The implementation includes:

- **Machine Learning Models**: Random Forest, SVM, Logistic Regression, and ensemble methods
- **Feature Engineering**: 48 features extracted from URLs and email content
- **Dataset**: Trained on 11,000+ phishing and legitimate samples
- **Performance**: 98.4% accuracy with Random Forest classifier

For detailed methodology and results, see the [research paper](docs/main.pdf).

## 🛠️ Technology Stack

- **Backend**: Python 3.9+, Flask 3.0.0
- **ML Framework**: scikit-learn 1.3.0
- **Data Processing**: pandas, numpy
- **Web Server**: gunicorn (production), Flask dev server (local)
- **Deployment**: Render.com with automatic CI/CD
- **Frontend**: HTML5, CSS3, JavaScript (vanilla)

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 98.4% | 98.1% | 98.7% | 98.4% |
| SVM (RBF) | 97.2% | 96.8% | 97.6% | 97.2% |
| Logistic Regression | 95.8% | 95.3% | 96.2% | 95.7% |
| Voting Ensemble | 98.2% | 97.9% | 98.5% | 98.2% |

## 🚀 Deployment

### Deploy to Render.com

1. **Fork this repository** to your GitHub account

2. **Sign up at [Render.com](https://render.com)**

3. **Create new Web Service**:
   - Connect your GitHub repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn --config gunicorn.conf.py app:app`

4. **Set environment variables**:
   ```
   FLASK_ENV=production
   FLASK_DEBUG=false
   LOG_LEVEL=info
   CORS_ORIGINS=*
   SECRET_KEY=[auto-generated]
   ```

5. **Deploy** - Render will automatically build and deploy your application

For detailed deployment instructions, see [DEPLOYMENT_INSTRUCTIONS.md](docs/DEPLOYMENT_INSTRUCTIONS.md).

## 🧪 Testing

### Run all tests
```bash
python test_comprehensive.py
```

### Test API endpoints
```bash
python test_api.py
```

### Test deployment configuration
```bash
python test_deployment.py
```

## 📝 Configuration

### Environment Variables

Create a `.env` file for local development:

```env
FLASK_ENV=development
FLASK_DEBUG=True
PORT=5001
LOG_LEVEL=debug
MAX_CONTENT_LENGTH=16777216
CORS_ORIGINS=*
SECRET_KEY=your-secret-key-here
```

For production, use `.env.production` with appropriate values.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Nouf Almosaied** - *Research & Development*
- **City, University of London** - *Academic Institution*

## 🙏 Acknowledgments

- Phishing dataset providers
- scikit-learn community
- Flask framework maintainers
- Render.com for hosting services

## 📧 Contact

For questions or collaboration opportunities:
- GitHub: [@NoufAlmosaied](https://github.com/NoufAlmosaied)
- Repository: [ai-chatbot-paper-python](https://github.com/NoufAlmosaied/ai-chatbot-paper-python)

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{almosaied2024phishguard,
  title={PhishGuard AI: An Intelligent Chatbot for Real-Time Phishing Detection},
  author={Almosaied, Nouf},
  year={2024},
  institution={City, University of London}
}
```

---

**⚠️ Disclaimer**: This tool is for educational and research purposes. Always verify suspicious links through multiple sources and follow cybersecurity best practices.