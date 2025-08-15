"""
Risk Analyzer
Converts ML predictions into user-friendly risk assessments
"""

from typing import Dict
import logging

logger = logging.getLogger(__name__)

class RiskAnalyzer:
    """Analyzes risk levels based on ML predictions."""
    
    def __init__(self):
        """Initialize the risk analyzer."""
        self.risk_thresholds = {
            'low': 0.3,      # 0-30% = Low risk
            'medium': 0.7,   # 30-70% = Medium risk
            'high': 1.0      # 70-100% = High risk
        }
        
        self.recommendations = {
            'low': "This appears safe, but always be cautious with unexpected links.",
            'medium': "Exercise caution. Verify the source before taking any action.",
            'high': "Avoid this link. This is likely a phishing attempt."
        }
        
        self.risk_descriptions = {
            'low': "The content shows minimal signs of being a phishing attempt.",
            'medium': "The content has some suspicious characteristics that warrant attention.",
            'high': "The content exhibits strong indicators of being a phishing attempt."
        }
    
    def assess(self, probability: float) -> Dict:
        """
        Assess risk level based on phishing probability.
        
        Args:
            probability: Probability of being phishing (0.0 to 1.0)
            
        Returns:
            Risk assessment dictionary
        """
        # Determine risk level
        if probability <= self.risk_thresholds['low']:
            level = 'low'
        elif probability <= self.risk_thresholds['medium']:
            level = 'medium'
        else:
            level = 'high'
        
        # Calculate risk score (0-100)
        risk_score = min(100, int(probability * 100))
        
        # Generate detailed assessment
        assessment = {
            'level': level,
            'score': risk_score,
            'probability': probability,
            'recommendation': self.recommendations[level],
            'description': self.risk_descriptions[level],
            'severity_indicators': self.get_severity_indicators(probability),
            'action_items': self.get_action_items(level)
        }
        
        return assessment
    
    def get_severity_indicators(self, probability: float) -> Dict:
        """Get visual indicators for risk severity."""
        indicators = {
            'color': self.get_risk_color(probability),
            'icon': self.get_risk_icon(probability),
            'urgency': self.get_urgency_level(probability),
            'confidence': self.get_confidence_level(probability)
        }
        
        return indicators
    
    def get_risk_color(self, probability: float) -> str:
        """Get color code for risk level."""
        if probability <= 0.3:
            return '#28a745'  # Green
        elif probability <= 0.7:
            return '#ffc107'  # Yellow/Orange
        else:
            return '#dc3545'  # Red
    
    def get_risk_icon(self, probability: float) -> str:
        """Get icon for risk level."""
        if probability <= 0.3:
            return '✅'  # Check mark
        elif probability <= 0.7:
            return '⚠️'  # Warning
        else:
            return '🚨'  # Emergency
    
    def get_urgency_level(self, probability: float) -> str:
        """Get urgency level description."""
        if probability <= 0.3:
            return 'low'
        elif probability <= 0.7:
            return 'moderate'
        else:
            return 'high'
    
    def get_confidence_level(self, probability: float) -> str:
        """Get confidence level in the assessment."""
        # Higher probabilities (closer to 0 or 1) indicate higher confidence
        confidence_score = max(probability, 1 - probability)
        
        if confidence_score >= 0.8:
            return 'high'
        elif confidence_score >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def get_action_items(self, risk_level: str) -> list:
        """Get specific action items based on risk level."""
        actions = {
            'low': [
                "Proceed with normal caution",
                "Verify sender if email is unexpected",
                "Check URL in address bar after clicking"
            ],
            'medium': [
                "Do not enter personal information immediately",
                "Verify the source through independent means",
                "Check for spelling and grammar errors",
                "Look for secure connection indicators (HTTPS)",
                "Hover over links to see actual destination"
            ],
            'high': [
                "DO NOT click any links",
                "DO NOT provide any personal information",
                "DO NOT download any attachments",
                "Report to IT security if in workplace",
                "Delete the message",
                "Block the sender",
                "Run antivirus scan if already clicked"
            ]
        }
        
        return actions.get(risk_level, [])
    
    def get_detailed_analysis(self, probability: float, features: list = None) -> Dict:
        """Get detailed risk analysis with explanations."""
        base_assessment = self.assess(probability)
        
        detailed = {
            'risk_breakdown': {
                'technical_score': probability,
                'user_friendly_score': f"{int(probability * 100)}/100",
                'risk_category': base_assessment['level'].title(),
                'threat_level': self.get_threat_level(probability)
            },
            'explanation': {
                'why_risky': self.explain_risk_factors(probability),
                'what_to_look_for': self.get_warning_signs(base_assessment['level']),
                'how_phishers_work': self.get_phishing_tactics(base_assessment['level'])
            },
            'next_steps': {
                'immediate': self.get_immediate_actions(base_assessment['level']),
                'prevention': self.get_prevention_tips(),
                'resources': self.get_helpful_resources()
            }
        }
        
        return {**base_assessment, **detailed}
    
    def get_threat_level(self, probability: float) -> str:
        """Get descriptive threat level."""
        if probability <= 0.2:
            return "Minimal Threat"
        elif probability <= 0.4:
            return "Low Threat"
        elif probability <= 0.6:
            return "Moderate Threat"
        elif probability <= 0.8:
            return "High Threat"
        else:
            return "Critical Threat"
    
    def explain_risk_factors(self, probability: float) -> str:
        """Explain why the content is considered risky."""
        if probability <= 0.3:
            return "The content shows few characteristics typical of phishing attempts."
        elif probability <= 0.7:
            return "The content exhibits some patterns commonly found in phishing attempts."
        else:
            return "The content strongly matches known phishing attack patterns."
    
    def get_warning_signs(self, risk_level: str) -> list:
        """Get warning signs to look for."""
        signs = {
            'low': [
                "Unexpected emails from unknown senders",
                "Urgent language or time pressure",
                "Generic greetings like 'Dear Customer'"
            ],
            'medium': [
                "Mismatched URLs (hover to check destination)",
                "Poor spelling and grammar",
                "Requests for personal information",
                "Suspicious attachments"
            ],
            'high': [
                "Threats of account closure or suspension",
                "Immediate action required language",
                "Links to IP addresses instead of domains",
                "Fake login pages",
                "Unexpected financial requests"
            ]
        }
        
        return signs.get(risk_level, [])
    
    def get_phishing_tactics(self, risk_level: str) -> list:
        """Explain common phishing tactics."""
        if risk_level == 'high':
            return [
                "Creating fake websites that look like legitimate services",
                "Using urgent language to pressure quick action",
                "Spoofing trusted brands and organizations",
                "Hiding malicious links behind legitimate-looking text",
                "Harvesting credentials through fake login forms"
            ]
        else:
            return [
                "Phishers often impersonate trusted organizations",
                "They create urgency to bypass critical thinking",
                "Legitimate companies rarely ask for passwords via email"
            ]
    
    def get_immediate_actions(self, risk_level: str) -> list:
        """Get immediate action recommendations."""
        if risk_level == 'high':
            return [
                "Close browser/email immediately",
                "Do not enter any information",
                "Scan device for malware",
                "Change passwords if already entered",
                "Contact IT support if workplace-related"
            ]
        elif risk_level == 'medium':
            return [
                "Stop and verify independently",
                "Contact the organization directly",
                "Do not click any links yet"
            ]
        else:
            return [
                "Proceed with normal caution",
                "Stay alert for suspicious behavior"
            ]
    
    def get_prevention_tips(self) -> list:
        """Get general prevention tips."""
        return [
            "Keep software and browsers updated",
            "Use two-factor authentication when available",
            "Verify suspicious emails through independent channels",
            "Never provide passwords or sensitive info via email",
            "Be skeptical of urgent or threatening messages",
            "Check URLs carefully before clicking",
            "Use reputable antivirus software"
        ]
    
    def get_helpful_resources(self) -> Dict:
        """Get helpful resources for users."""
        return {
            "phishing_education": "Learn more about phishing at consumer.ftc.gov",
            "reporting": "Report phishing to the Anti-Phishing Working Group",
            "verification": "Always verify suspicious communications independently",
            "tools": "Use URL checkers and safe browsing tools"
        }