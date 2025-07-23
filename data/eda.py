import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, 'features.csv')
# Load data
df = pd.read_csv(data_path)

# Basic info
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('First 5 rows:')
print(df.head())

# Class distribution
print('\nClass distribution:')
print(df['label'].value_counts())

# Plot class distribution
plt.figure(figsize=(5,4))
sns.countplot(x='label', data=df)
plt.title('Class Distribution (0=Ham, 1=Spam/Phishing)')
plt.xlabel('Label')
plt.ylabel('Count')
plt.savefig(os.path.join(BASE_DIR, 'class_distribution.png'))
plt.close()

# Check for missing values
print('\nMissing values:')
print(df.isnull().sum())

# Analyze body length
print('\nBody length stats:')
print(df['body_length'].describe())

plt.figure(figsize=(7,4))
sns.histplot(df['body_length'], bins=50, kde=True)
plt.title('Distribution of Email Body Length')
plt.xlabel('Body Length (characters)')
plt.ylabel('Frequency')
plt.savefig(os.path.join(BASE_DIR, 'body_length_distribution.png'))
plt.close()

# Emails with URLs (urls column is binary: 0 = no URLs, 1 = at least one URL)
df['has_url'] = (df['num_urls'] > 0).astype(int)
num_with_url = (df['num_urls'] > 0).sum()
print(f'\nNumber of emails with at least one URL: {num_with_url} ({num_with_url/len(df)*100:.2f}%)')

plt.figure(figsize=(5,4))
sns.countplot(x='has_url', data=df)
plt.title('Emails with URLs (0 = No URLs, 1 = At least 1 URL)')
plt.xlabel('Contains URL')
plt.ylabel('Count')
plt.xticks([0,1], ['No', 'Yes'])
plt.savefig(os.path.join(BASE_DIR, 'emails_with_urls_binary.png'))
plt.close()

# Distribution of URL presence by label
plt.figure(figsize=(7,4))
sns.countplot(x='has_url', hue='label', data=df)
plt.title('URL Presence by Label (0 = No URLs, 1 = At least 1 URL)')
plt.xlabel('Contains URL')
plt.ylabel('Count')
plt.xticks([0,1], ['No', 'Yes'])
plt.legend(title='Label', labels=['Ham', 'Spam/Phishing'])
plt.savefig(os.path.join(BASE_DIR, 'url_presence_by_label.png'))
plt.close()

# Save a summary CSV
# (urls is already binary, so no need for contains_url)
df[['label','body_length','has_url']].to_csv(os.path.join(BASE_DIR, 'eda_summary_with_url.csv'), index=False)

print('\nEDA complete. Plots and summary saved to data/.') 