import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class SMSPhishingDetector:
    def __init__(self):
        print("Loading SMS phishing detector...")
        
        # Load sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load preprocessed data for similarity search
        self.load_training_data()
        
        print("Detector loaded successfully!")
    
    def load_training_data(self):
        """Load training data for similarity search"""
        try:
            # Load your preprocessed data
            self.training_data = pd.read_csv('../data/emails_preprocessed.csv')
            print(f"Loaded {len(self.training_data)} training examples")
        except:
            # Create some example data if file doesn't exist
            self.training_data = pd.DataFrame({
                'text': [
                    "Your account has been suspended. Click here to verify your identity immediately.",
                    "URGENT: Your bank account will be locked unless you update your password now!",
                    "You've won $1,000,000! Click here to claim your prize now!",
                    "Hi, just wanted to check in on the project status. Let me know if you need anything.",
                    "Meeting reminder: Team sync at 3 PM today. Please prepare your updates.",
                    "Your password will expire in 24 hours. Click here to update now.",
                    "Security alert: Unusual login detected. Verify your account immediately.",
                    "You have a new message. Click here to view it.",
                    "Your subscription will be cancelled unless you update your payment method.",
                    "Congratulations! You've been selected for a special offer."
                ],
                'label': [1, 1, 1, 0, 0, 1, 1, 0, 1, 1]  # 1=spam/phishing, 0=legitimate
            })
            print("Created example training data")
    
    def get_embeddings(self, texts):
        """Get embeddings for a list of texts"""
        return self.embedding_model.encode(texts)
    
    def analyze_text_advanced(self, text):
        """Advanced SMS phishing analysis using multiple techniques"""
        
        # 1. Rule-based analysis
        rule_score = self.rule_based_analysis(text)
        
        # 2. Similarity-based analysis
        similarity_score = self.similarity_based_analysis(text)
        
        # 3. Feature-based analysis
        feature_score = self.feature_based_analysis(text)
        
        # Combine scores
        combined_score = (rule_score * 0.4 + similarity_score * 0.4 + feature_score * 0.2)
        
        # Generate analysis text
        analysis = self.generate_analysis_text(text, rule_score, similarity_score, feature_score)
        
        return {
            'text': text,
            'risk_score': float(combined_score),  # Convert to Python float
            'rule_score': float(rule_score),      # Convert to Python float
            'similarity_score': float(similarity_score),  # Convert to Python float
            'feature_score': float(feature_score),  # Convert to Python float
            'analysis': analysis,
            'is_phishing': combined_score > 0.5,
            'confidence': 'high' if abs(combined_score - 0.5) > 0.3 else 'medium'
        }
    
    def rule_based_analysis(self, text):
        """Rule-based SMS phishing detection"""
        text_lower = text.lower()
        
        # High-risk indicators (common in SMS spam/phishing)
        high_risk_indicators = [
            'urgent', 'immediately', 'suspended', 'locked', 'expired',
            'verify your account', 'confirm your identity', 'update your password',
            'click here', 'claim your prize', 'you\'ve won', 'free money',
            'bank account', 'credit card', 'social security', 'txt', 'text',
            'claim now', 'limited time', 'act now', 'expires', 'last chance',
            'free gift', 'cash prize', 'lottery', 'winner', 'congratulations'
        ]
        
        # Medium-risk indicators
        medium_risk_indicators = [
            'password', 'account', 'security', 'login', 'update',
            'verify', 'confirm', 'suspended', 'locked', 'expired',
            'call now', 'reply', 'stop', 'unsubscribe', 'opt out'
        ]
        
        # Count indicators
        high_risk_count = sum(1 for indicator in high_risk_indicators if indicator in text_lower)
        medium_risk_count = sum(1 for indicator in medium_risk_indicators if indicator in text_lower)
        
        # Calculate score
        score = min(1.0, (high_risk_count * 0.3 + medium_risk_count * 0.1))
        
        return score
    
    def similarity_based_analysis(self, text):
        """Analyze similarity to known SMS spam/legitimate messages"""
        if len(self.training_data) == 0:
            return 0.5
        
        # Get embeddings
        all_texts = [text] + self.training_data['text'].tolist()
        embeddings = self.get_embeddings(all_texts)
        
        # Calculate similarities
        query_embedding = embeddings[0].reshape(1, -1)
        training_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(query_embedding, training_embeddings)[0]
        
        # Find most similar messages
        top_indices = np.argsort(similarities)[::-1][:3]
        
        # Calculate weighted score based on similar messages
        total_score = 0
        total_weight = 0
        
        for idx in top_indices:
            similarity = similarities[idx]
            label = self.training_data.iloc[idx]['label']
            weight = similarity
            
            total_score += label * weight
            total_weight += weight
        
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.5
    
    def feature_based_analysis(self, text):
        """Analyze SMS text features for phishing indicators"""
        text_lower = text.lower()
        
        # SMS-specific text features
        features = {
            'has_urgency': any(word in text_lower for word in ['urgent', 'immediately', 'now', 'asap', 'hurry']),
            'has_links': 'click here' in text_lower or 'http' in text_lower or 'www.' in text_lower,
            'has_money': any(word in text_lower for word in ['money', 'prize', 'won', 'cash', 'dollar', 'free']),
            'has_threats': any(word in text_lower for word in ['suspended', 'locked', 'expired', 'terminated', 'blocked']),
            'has_requests': any(word in text_lower for word in ['verify', 'confirm', 'update', 'login', 'call']),
            'excessive_caps': sum(1 for c in text if c.isupper()) / len(text) > 0.3,
            'short_length': len(text.split()) < 10,
            'has_sms_indicators': any(word in text_lower for word in ['txt', 'text', 'sms', 'message']),
            'has_numbers': any(char.isdigit() for char in text),
            'has_emojis': any(ord(char) > 127 for char in text)  # Basic emoji detection
        }
        
        # Calculate feature score
        feature_score = sum(features.values()) / len(features)
        
        return feature_score
    
    def generate_analysis_text(self, text, rule_score, similarity_score, feature_score):
        """Generate human-readable analysis for SMS"""
        analysis_parts = []
        
        if rule_score > 0.7:
            analysis_parts.append("High risk indicators detected")
        elif rule_score > 0.4:
            analysis_parts.append("Some suspicious indicators found")
        else:
            analysis_parts.append("No obvious phishing indicators")
        
        if similarity_score > 0.7:
            analysis_parts.append("Similar to known SMS spam messages")
        elif similarity_score < 0.3:
            analysis_parts.append("Similar to legitimate SMS messages")
        else:
            analysis_parts.append("Mixed similarity patterns")
        
        if feature_score > 0.6:
            analysis_parts.append("Multiple suspicious SMS features detected")
        elif feature_score < 0.3:
            analysis_parts.append("Few suspicious features")
        else:
            analysis_parts.append("Some suspicious features present")
        
        return ". ".join(analysis_parts) + "."
    
    def find_similar_messages(self, query_text, top_k=5):
        """Find similar SMS messages using embeddings"""
        if len(self.training_data) == 0:
            return []
        
        # Get embeddings
        all_texts = [query_text] + self.training_data['text'].tolist()
        embeddings = self.get_embeddings(all_texts)
        
        # Calculate similarities
        query_embedding = embeddings[0].reshape(1, -1)
        message_embeddings = embeddings[1:]
        
        similarities = cosine_similarity(query_embedding, message_embeddings)[0]
        
        # Get top similar messages
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'message': self.training_data.iloc[idx]['text'],
                'similarity': float(similarities[idx]),  # Convert to Python float
                'label': 'SPAM/PHISHING' if self.training_data.iloc[idx]['label'] == 1 else 'LEGITIMATE'
            })
        
        return results 