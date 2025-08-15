"""
Chatbot Conversation Engine
Handles natural language interactions for phishing detection
"""

import re
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ChatbotEngine:
    """Manages chatbot conversations and responses."""
    
    def __init__(self):
        """Initialize the chatbot engine."""
        self.greetings = [
            'hello', 'hi', 'hey', 'greetings', 'good morning',
            'good afternoon', 'good evening'
        ]
        
        self.help_keywords = [
            'help', 'how', 'what', 'explain', 'guide', 'assist'
        ]
        
        self.conversation_context = {}
    
    def process_message(self, message: str, session_id: str = 'default') -> Dict:
        """
        Process user message and determine response.
        
        Args:
            message: User input message
            session_id: Session identifier for context
            
        Returns:
            Response dictionary
        """
        message_lower = message.lower().strip()
        
        # Check for greeting
        if any(greet in message_lower for greet in self.greetings):
            return self.greeting_response()
        
        # Check for help request
        if any(keyword in message_lower for keyword in self.help_keywords):
            return self.help_response()
        
        # Check for URL or email content
        urls = self.extract_urls(message)
        if urls:
            return {
                'requires_analysis': True,
                'extracted_content': urls[0],
                'content_type': 'url',
                'message': f"I found a URL in your message. Let me analyze it for you..."
            }
        
        # Check if message looks like email content
        if self.is_email_content(message):
            return {
                'requires_analysis': True,
                'extracted_content': message,
                'content_type': 'email',
                'message': "I'll analyze this email content for phishing indicators..."
            }
        
        # Default response for unclear input
        return {
            'requires_analysis': False,
            'message': "I can help you check if a URL or email is a phishing attempt. Please share the suspicious link or email content you'd like me to analyze."
        }
    
    def greeting_response(self) -> Dict:
        """Generate greeting response."""
        return {
            'requires_analysis': False,
            'message': "Hello! I'm your AI-powered phishing detection assistant. I can help you identify suspicious URLs and emails. Just paste any link or email content you'd like me to check."
        }
    
    def help_response(self) -> Dict:
        """Generate help response."""
        return {
            'requires_analysis': False,
            'message': """Here's how I can help you:

1. **Check URLs**: Send me any suspicious link and I'll analyze it for phishing indicators
2. **Analyze Emails**: Paste email content and I'll check for phishing attempts
3. **Risk Assessment**: I'll provide a risk level (Low/Medium/High) with explanations
4. **Recommendations**: I'll suggest what action you should take

Just paste the URL or email you want me to check!"""
        }
    
    def generate_response(self, content: str, prediction: Dict, risk_assessment: Dict) -> str:
        """
        Generate a detailed response based on analysis results.
        
        Args:
            content: Analyzed content
            prediction: ML model prediction
            risk_assessment: Risk analysis results
            
        Returns:
            Formatted response string
        """
        risk_emoji = {
            'low': '✅',
            'medium': '⚠️',
            'high': '🚨'
        }
        
        emoji = risk_emoji.get(risk_assessment['level'], '❓')
        confidence_percent = prediction['confidence'] * 100
        
        response = f"{emoji} **{risk_assessment['level'].upper()} RISK**\n\n"
        
        if prediction['is_phishing']:
            response += f"This appears to be a **phishing attempt** with {confidence_percent:.1f}% confidence.\n\n"
        else:
            response += f"This appears to be **legitimate** with {confidence_percent:.1f}% confidence.\n\n"
        
        # Add suspicious indicators if available
        if prediction.get('top_features'):
            response += "**Key Indicators:**\n"
            for feature in prediction['top_features'][:3]:
                response += f"• {self.format_feature_name(feature['name'])}\n"
            response += "\n"
        
        # Add recommendation
        response += f"**Recommendation:** {risk_assessment['recommendation']}\n"
        
        # Add tips based on risk level
        if risk_assessment['level'] == 'high':
            response += "\n**Safety Tips:**\n"
            response += "• Do not click on any links\n"
            response += "• Do not provide personal information\n"
            response += "• Report this to your IT security team\n"
            response += "• Delete the message\n"
        elif risk_assessment['level'] == 'medium':
            response += "\n**Proceed with caution:**\n"
            response += "• Verify the sender independently\n"
            response += "• Check the actual URL by hovering (don't click)\n"
            response += "• Look for spelling and grammar errors\n"
        
        return response
    
    def format_feature_name(self, feature_name: str) -> str:
        """Format feature name for user display."""
        feature_descriptions = {
            'NumDots': 'Excessive dots in URL',
            'SubdomainLevel': 'Suspicious subdomain depth',
            'UrlLength': 'Unusually long URL',
            'NoHttps': 'Missing secure connection (HTTPS)',
            'IpAddress': 'Uses IP address instead of domain',
            'NumDash': 'Multiple dashes in URL',
            'AtSymbol': 'Contains @ symbol (possible redirect)',
            'RandomString': 'Random characters in domain',
            'NumSensitiveWords': 'Contains suspicious keywords',
            'EmbeddedBrandName': 'Brand name used suspiciously'
        }
        
        return feature_descriptions.get(feature_name, feature_name)
    
    def extract_urls(self, text: str) -> list:
        """Extract URLs from text."""
        # Improved URL regex pattern
        url_pattern = r'(?:https?://)?(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
        urls = re.findall(url_pattern, text)
        return urls
    
    def is_email_content(self, text: str) -> bool:
        """Check if text appears to be email content."""
        email_indicators = [
            'dear', 'subject:', 'from:', 'to:', 'sent:',
            'verify your account', 'click here', 'urgent',
            'suspended', 'confirm your', 'update your'
        ]
        
        text_lower = text.lower()
        indicator_count = sum(1 for indicator in email_indicators if indicator in text_lower)
        
        # Consider it email content if it has multiple indicators or is long
        return indicator_count >= 2 or len(text) > 200