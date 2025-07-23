import os
import openai
import joblib
from langchain.chains import SimpleSequentialChain
from langchain_community.llms import OpenAI as LangChainOpenAI
import json
import numpy as np

# Set your OpenAI API key (recommended: use environment variable)
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load XGBoost model (assumes model is saved as data/best_model.pkl)
xgb_model_path = os.path.join(os.path.dirname(__file__), '../notebooks/best_model.pkl')
xgb_model = joblib.load(xgb_model_path)

# Load TF-IDF vectorizers
base_dir = os.path.dirname(__file__)
tfidf_body = joblib.load(os.path.join(base_dir, '../data/tfidf_body_vectorizer.pkl'))
tfidf_subject = joblib.load(os.path.join(base_dir, '../data/tfidf_subject_vectorizer.pkl'))

def extract_features(body, sender, subject=None):
    # Handle subject being None or blank
    subject = subject if subject is not None else ""
    # Manual features
    body_length = len(body)
    num_digits = sum(c.isdigit() for c in body)
    subject_length = len(subject)
    is_subject_empty = int(subject.strip() == "")
    num_exclamation_marks = body.count('!')
    import re
    url_pattern = r'(https?://[^\s]+)'
    num_urls = len(re.findall(url_pattern, body))
    subject_has_re = int(subject.lower().startswith('re:'))
    match = re.search(r'@([\\w.-]+)', sender)
    sender_domain_length = len(match.group(1)) if match else 0
    num_uppercase_words = len([w for w in body.split() if w.isupper() and len(w) > 1])
    manual_features = [
        body_length, num_digits, subject_length, is_subject_empty,
        num_exclamation_marks, num_urls, subject_has_re, sender_domain_length, num_uppercase_words
    ]
    # TF-IDF features
    body_tfidf = tfidf_body.transform([body]).toarray()[0]
    subject_tfidf = tfidf_subject.transform([subject]).toarray()[0]
    # Combine all features (order must match training)
    features = np.concatenate([manual_features, body_tfidf, subject_tfidf]).reshape(1, -1)
    return features

# LLM wrapper for LangChain
def llm_explanation(body, sender, subject=None):
    subject = subject if subject is not None else "(no subject)"
    prompt = (
        "You are an expert in email security. Analyze the following email and assess whether it is likely to be phishing or spam. "
        "Provide a risk score from 0 (very safe) to 1 (very risky), and a brief explanation for your assessment.\n"
        f"Sender: {sender}\n"
        f"Subject: {subject}\n"
        f"Body:\n{body}\n"
        "Respond in JSON with keys: risk_score (float), explanation (string)."
    )
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.2
    )
    content = response.choices[0].message.content
    return content

# LangChain pipeline function
def analyze_email_with_chain(body, sender, subject=None):
    features = extract_features(body, sender, subject)
    xgb_score = float(xgb_model.predict_proba(features)[0][1])
    llm_response = llm_explanation(body, sender, subject)
    result = {
        "xgboost_risk_score": xgb_score,
        "llm_response": llm_response
    }
    return result

# Example usage (for testing)
if __name__ == "__main__":
    test_body = """Dear Customer,

Your account has been temporarily suspended due to suspicious activity. Please verify your identity to restore access:

https://bank-secure.com/verify

If you do not respond within 24 hours, your account will be permanently closed.

Sincerely,  
Bank Support Team
"""
    test_sender = "support@bank-secure.com"
    test_subject = "Account Suspended - Action Required"

    result = analyze_email_with_chain(test_body, test_sender, test_subject)

    llm_dict = json.loads(result['llm_response'])
    print("XGBoost Risk Score:", result['xgboost_risk_score'])
    print("LLM Risk Score:", llm_dict['risk_score'])
    print("LLM Explanation:", llm_dict['explanation'])
