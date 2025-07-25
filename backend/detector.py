import os
import openai
import joblib
from langchain.chains import SimpleSequentialChain
from langchain_community.llms import OpenAI as LangChainOpenAI
import json
import numpy as np
import re
from urllib.parse import urlparse
import tldextract

def extract_features(body, sender, subject=None):
    subject = subject if subject is not None else ""
    body = body if isinstance(body, str) else ""
    subject = subject if isinstance(subject, str) else ""
    # Manual features
    body_length = len(body)
    num_digits = sum(c.isdigit() for c in body)
    subject_length = len(subject)
    is_subject_empty = int(subject.strip() == "")
    num_exclamation_marks = body.count('!')
    url_pattern = r'(https?://[^\s]+)'
    urls = re.findall(url_pattern, body)
    num_urls = len(urls)
    subject_has_re = int(subject.lower().startswith('re:'))
    match = re.search(r'@([\w.-]+)', sender)
    sender_domain_length = len(match.group(1)) if match else 0
    num_uppercase_words = len([w for w in body.split() if w.isupper() and len(w) > 1])
    sender_domain = tldextract.extract(sender).top_domain_under_public_suffix if sender else ""
    sender_matches_url = 0
    uses_https = 0
    url_length = 0
    num_subdomains = 0
    has_ip_address = 0
    for url in urls:
        try:
            parsed = urlparse(url)
            ext = tldextract.extract(url)
            url_domain = ext.top_domain_under_public_suffix
            if sender_domain and url_domain and sender_domain == url_domain:
                sender_matches_url = 1
            if parsed.scheme == "https":
                uses_https = 1
            url_length = max(url_length, len(url))
            num_subdomains = max(num_subdomains, len([s for s in ext.subdomain.split('.') if s]))
            if parsed.hostname and re.match(r'^\d{1,3}(\.\d{1,3}){3}$', parsed.hostname):
                has_ip_address = 1
        except Exception:
            continue
    manual_features = [
        body_length, num_digits, subject_length, is_subject_empty,
        num_exclamation_marks, num_urls, subject_has_re, sender_domain_length, num_uppercase_words,
        sender_matches_url, uses_https, url_length, num_subdomains, has_ip_address
    ]
    # Load TF-IDF vectorizers
    base_dir = os.path.dirname(__file__)
    tfidf_body = joblib.load(os.path.join(base_dir, '../data/tfidf_body_vectorizer.pkl'))
    tfidf_subject = joblib.load(os.path.join(base_dir, '../data/tfidf_subject_vectorizer.pkl'))
    body_tfidf = tfidf_body.transform([body]).toarray()[0]
    subject_tfidf = tfidf_subject.transform([subject]).toarray()[0]
    features = np.concatenate([manual_features, body_tfidf, subject_tfidf]).reshape(1, -1)
    return features

# Set your OpenAI API key (recommended: use environment variable)
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load XGBoost model (assumes model is saved as data/best_model.pkl)
xgb_model_path = os.path.join(os.path.dirname(__file__), '../notebooks/best_model.pkl')
xgb_model = joblib.load(xgb_model_path)

def llm_explanation(body, sender, subject=None, xgb_score=None):
    subject = subject if subject is not None else "(no subject)"
    xgb_score_str = f"The automated model assigned this email a risk score of {xgb_score:.4f} (0 = very safe, 1 = very risky).\n" if xgb_score is not None else ""
    prompt = (
        "You are an expert in email security. Analyze the following email and provide your own independent, evidence-based risk assessment. "
        "Do NOT simply repeat or mirror the automated model's score. Use your own reasoning and judgment based on the email content, sender, and context. "
        "The automated model's score is provided as one input, but your risk_score should reflect your own analysis. "
        "Provide a risk_score from 0 (very safe) to 1 (very risky), rounded to 4 decimal places, and a brief explanation for your assessment. "
        "If the email appears legitimate, assign a low risk score. If there are clear signs of phishing, assign a higher risk score.\n"
        f"Sender: {sender}\n"
        f"Subject: {subject}\n"
        f"Body:\n{body}\n"
        f"{xgb_score_str}"
        "Please consider the automated model's score as one factor, but make your own determination based on the evidence.\n"
        "Respond in JSON with keys: risk_score (float, rounded to 4 decimal places), explanation (string)."
    )
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.2
    )
    content = response.choices[0].message.content
    return content

def analyze_email_with_chain(body, sender, subject=None):
    features = extract_features(body, sender, subject)
    xgb_score = float(xgb_model.predict_proba(features)[0][1])
    llm_response = llm_explanation(body, sender, subject, xgb_score=xgb_score)
    llm_dict = json.loads(llm_response)
    llm_score = float(llm_dict['risk_score'])
    # Hybrid risk score (weighted: 0.35 XGBoost, 0.65 LLM)
    hybrid_score = (0.35 * xgb_score) + (0.65 * llm_score)
    risk_percent = int(round(hybrid_score * 100))
    # Risk indicator
    def risk_indicator(score_percent):
        if score_percent < 20:
            return "Very Low"
        elif score_percent < 40:
            return "Low"
        elif score_percent < 60:
            return "Medium"
        elif score_percent < 80:
            return "High"
        else:
            return "Very High"
    indicator = risk_indicator(risk_percent)
    result = {
        "xgboost_risk_score": xgb_score,
        "llm_risk_score": llm_score,
        "hybrid_risk_percent": risk_percent,
        "risk_indicator": indicator,
        "llm_explanation": llm_dict['explanation']
    }
    return result
