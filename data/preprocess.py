import pandas as pd
import re

# File paths
spamassassin_path = 'data/SpamAssasin.csv'
ceas_path = 'data/CEAS_08.csv'
nazario_path = 'data/Nazario.csv'
output_path = 'data/combined_emails.csv'
features_path = 'data/features.csv'

# List of free email domains
FREE_EMAIL_DOMAINS = [
    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com', 'icloud.com',
    'mail.com', 'zoho.com', 'protonmail.com', 'gmx.com', 'yandex.com', 'msn.com',
    'live.com', 'ymail.com', 'rocketmail.com', 'inbox.com', 'me.com', 'fastmail.com'
]

# List of common URL shorteners
SHORTENERS = [
    'bit.ly', 'goo.gl', 'tinyurl.com', 'ow.ly', 't.co', 'is.gd', 'buff.ly', 'adf.ly',
    'bit.do', 'mcaf.ee', 'rebrand.ly', 'su.pr', 'cli.gs', 'tr.im', 'tiny.cc', 'shorte.st',
    'cutt.ly', 'rb.gy', 'soo.gd', 's.id', 'v.gd', 'qr.ae', 'lnkd.in', 'db.tt', 'qr.net',
    'po.st', 'bc.vc', 'twitthis.com', 'u.to', 'j.mp', 'b.link', 'yourls.org', 'prettylinkpro.com'
]

# Load datasets
df_spam = pd.read_csv(spamassassin_path)
df_ceas = pd.read_csv(ceas_path)
df_nazario = pd.read_csv(nazario_path)

# Inspect datasets
print('SpamAssassin shape:', df_spam.shape)
print('SpamAssassin columns:', df_spam.columns.tolist())
print('SpamAssassin label distribution:')
print(df_spam['label'].value_counts())
print('\n')

print('CEAS_08 shape:', df_ceas.shape)
print('CEAS_08 columns:', df_ceas.columns.tolist())
print('CEAS_08 label distribution:')
print(df_ceas['label'].value_counts())
print('\n')

print('Nazario shape:', df_nazario.shape)
print('Nazario columns:', df_nazario.columns.tolist())
print('Nazario label distribution:')
print(df_nazario['label'].value_counts())
print('\n')

# Always keep these columns if present
keep_cols = ['sender', 'receiver', 'date', 'subject', 'body', 'label', 'urls']
df_spam = df_spam[[col for col in keep_cols if col in df_spam.columns]]
df_ceas = df_ceas[[col for col in keep_cols if col in df_ceas.columns]]
df_nazario = df_nazario[[col for col in keep_cols if col in df_nazario.columns]]
# Concatenate datasets
df_combined = pd.concat([df_spam, df_ceas, df_nazario], ignore_index=True)

# Drop rows with missing labels or body
required_cols = ['label', 'body']
df_combined = df_combined.dropna(subset=required_cols)

# Feature engineering
# 1. body_length
df_combined['body_length'] = df_combined['body'].astype(str).apply(len)
# 2. urls (already present as binary, but will replace with num_urls)
# 3. subject_length
if 'subject' in df_combined.columns:
    df_combined['subject_length'] = df_combined['subject'].astype(str).apply(len)
    df_combined['is_subject_empty'] = df_combined['subject'].astype(str).str.strip().eq('').astype(int)
    # subject_has_re: 1 if subject starts with 'Re:' (case-insensitive)
    df_combined['subject_has_re'] = df_combined['subject'].astype(str).str.lower().str.startswith('re:').astype(int)
else:
    df_combined['subject_length'] = 0
    df_combined['is_subject_empty'] = 1
    df_combined['subject_has_re'] = 0

# 4. num_digits
def count_digits(text):
    return len(re.findall(r'\d', str(text)))
df_combined['num_digits'] = df_combined['body'].apply(count_digits)

# 5. num_exclamation_marks
def count_exclamations(text):
    return str(text).count('!')
df_combined['num_exclamation_marks'] = df_combined['body'].apply(count_exclamations)

# 6. num_urls (extract actual URLs from body)
url_pattern = r'(https?://[^\s]+)'
def extract_urls(text):
    return re.findall(url_pattern, str(text))
df_combined['num_urls'] = df_combined['body'].apply(lambda x: len(extract_urls(x)))

# 7. sender_domain_length
def get_domain(sender):
    if pd.isnull(sender):
        return ''
    match = re.search(r'@([\w.-]+)', str(sender))
    return match.group(1).lower() if match else ''
df_combined['sender_domain_length'] = df_combined['sender'].apply(lambda x: len(get_domain(x)))

# 8. num_uppercase_words
def count_uppercase_words(text):
    return len([w for w in str(text).split() if w.isupper() and len(w) > 1])
df_combined['num_uppercase_words'] = df_combined['body'].apply(count_uppercase_words)

# Save features for modeling
feature_cols = [
    'label', 'body_length', 'num_digits', 'subject_length', 'is_subject_empty',
    'num_exclamation_marks', 'num_urls', 'subject_has_re', 'sender_domain_length', 'num_uppercase_words'
]
df_combined[feature_cols].to_csv(features_path, index=False)
print(f'Feature set saved to {features_path}')
