import pandas as pd
import re
import email
from email import policy
import glob
import os
from urllib.parse import urlparse
import tldextract
import difflib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRUSTED_DOMAINS = {
    'amazon.com', 'apple.com', 'google.com', 'microsoft.com', 'paypal.com',
    'chase.com', 'wellsfargo.com', 'bankofamerica.com', 'uber.com', 'linkedin.com',
    'netflix.com', 'comcast.com', 'southwest.com', 'delta.com', 'pge.com',
    'usps.com', 'fedex.com', 'ups.com', 'airbnb.com', 'expedia.com', 'booking.com'
}

def parse_eml(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        # Use compat32 policy for raw string headers
        msg = email.message_from_file(f, policy=policy.compat32)
    subject = msg.get('subject', '') or ""
    sender = msg.get('from', '') or ""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain' and not part.is_multipart():
                try:
                    body += part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='ignore')
                except Exception:
                    continue
    else:
        try:
            body = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='ignore')
        except Exception:
            body = ""
    return sender, subject, body

def extract_features(body, sender, subject=None):
    body = body if isinstance(body, str) else ""
    subject = subject if isinstance(subject, str) else ""
    subject_length = len(subject)
    is_subject_empty = int(subject.strip() == "")
    num_exclamation_marks = body.count('!')
    url_pattern = r'(https?://[^\s]+)'
    urls = re.findall(url_pattern, body)
    num_urls = len(urls)
    body_length = len(body)
    num_digits = sum(c.isdigit() for c in body)
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
    return [manual_features]

def parse_enron_message(raw_message):
    msg = email.message_from_string(raw_message, policy=policy.compat32)
    subject = msg.get('subject', '') or ""
    sender = msg.get('from', '') or ""
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain' and not part.is_multipart():
                try:
                    body += part.get_payload(decode=True).decode(part.get_content_charset() or 'utf-8', errors='ignore')
                except Exception:
                    continue
    else:
        try:
            body = msg.get_payload(decode=True).decode(msg.get_content_charset() or 'utf-8', errors='ignore')
        except Exception:
            body = ""
    return sender, subject, body

# --- 1. Process EPVME phishing emails ---
epvme_dir = os.path.join(BASE_DIR, 'raw_emails/')
epvme_files = glob.glob(os.path.join(epvme_dir, '*.eml'))
epvme_rows = []
for file_path in epvme_files:
    sender, subject, body = parse_eml(file_path)
    features = extract_features(body, sender, subject)[0]
    features.append(body)
    features.append(subject)
    features.append(1)  # 1 = phishing
    epvme_rows.append(features)

# --- 2. Process Enron legitimate emails ---
enron_csv = os.path.join(BASE_DIR, 'Enron.csv')
enron_df = pd.read_csv(enron_csv)
enron_rows = []
for idx, row in enron_df.iterrows():
    sender, subject, body = parse_enron_message(row['message'])
    features = extract_features(body, sender, subject)[0]
    features.append(body)
    features.append(subject)
    features.append(0)  # 0 = legitimate
    enron_rows.append(features)

# --- 3. Process CEAS_08 emails ---
ceas_csv = os.path.join(BASE_DIR, 'CEAS_08.csv')
ceas_df = pd.read_csv(ceas_csv)
ceas_rows = []
for idx, row in ceas_df.iterrows():
    sender = row['sender']
    subject = row['subject']
    body = row['body']
    label = int(row['label'])
    features = extract_features(body, sender, subject)[0]
    features.append(body)
    features.append(subject)
    features.append(label)
    ceas_rows.append(features)

# --- 4. Combine and save ---
feature_cols = [
    'body_length', 'num_digits', 'subject_length', 'is_subject_empty',
    'num_exclamation_marks', 'num_urls', 'subject_has_re', 'sender_domain_length', 'num_uppercase_words',
    'sender_matches_url', 'uses_https', 'url_length', 'num_subdomains', 'has_ip_address',
    'body', 'subject', 'label'
]
all_rows = epvme_rows + enron_rows + ceas_rows
df_all = pd.DataFrame(all_rows, columns=feature_cols)
output_csv = os.path.join(BASE_DIR, 'combined_features.csv')
df_all.to_csv(output_csv, index=False)
print("Combined feature set saved to combined_features.csv")