import pandas as pd
import re
import email
from email import policy
import glob
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    subject = subject if subject is not None else ""
    body_length = len(body)
    num_digits = sum(c.isdigit() for c in body)
    subject_length = len(subject)
    is_subject_empty = int(subject.strip() == "")
    num_exclamation_marks = body.count('!')
    url_pattern = r'(https?://[^\s]+)'
    num_urls = len(re.findall(url_pattern, body))
    subject_has_re = int(subject.lower().startswith('re:'))
    match = re.search(r'@([\w.-]+)', sender)
    sender_domain_length = len(match.group(1)) if match else 0
    num_uppercase_words = len([w for w in body.split() if w.isupper() and len(w) > 1])
    return [[
        body_length, num_digits, subject_length, is_subject_empty,
        num_exclamation_marks, num_urls, subject_has_re, sender_domain_length, num_uppercase_words
    ]]

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
    features.append(1)  # 1 = phishing
    epvme_rows.append(features)

# --- 2. Process Enron legitimate emails ---
enron_csv = os.path.join(BASE_DIR, 'emails.csv')
enron_df = pd.read_csv(enron_csv)
enron_rows = []
for idx, row in enron_df.iterrows():
    sender, subject, body = parse_enron_message(row['message'])
    features = extract_features(body, sender, subject)[0]
    features.append(0)  # 0 = legitimate
    enron_rows.append(features)

# --- 3. Combine and save ---
feature_cols = [
    'body_length', 'num_digits', 'subject_length', 'is_subject_empty',
    'num_exclamation_marks', 'num_urls', 'subject_has_re', 'sender_domain_length', 'num_uppercase_words', 'label'
]
all_rows = epvme_rows + enron_rows
df_all = pd.DataFrame(all_rows, columns=feature_cols)
output_csv = os.path.join(BASE_DIR, '../data/combined_features.csv')
df_all.to_csv(output_csv, index=False)
print("Combined feature set saved to data/combined_features.csv")
