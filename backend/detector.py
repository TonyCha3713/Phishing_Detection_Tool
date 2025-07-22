import os
import openai

# Set your OpenAI API key (recommended: use environment variable)
openai.api_key = os.getenv('OPENAI_API_KEY')

def gpt35_analyze_email(email_body):
    """
    Analyze an email body using OpenAI GPT-3.5-turbo and return a risk assessment and explanation.
    """
    prompt = (
        "You are an expert in email security. Analyze the following email body and assess whether it is likely to be phishing or spam. "
        "Provide a risk score from 0 (very safe) to 1 (very risky), and a brief explanation for your assessment.\n"
        f"Email body:\n{email_body}\n"
        "Respond in JSON with keys: risk_score (float), explanation (string)."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.2
    )
    # Extract the model's response
    content = response['choices'][0]['message']['content']
    return content
