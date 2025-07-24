import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(BASE_DIR, 'combined_features.csv')
df = pd.read_csv(data_path)

# Basic info
print('Shape:', df.shape)
print('Columns:', df.columns.tolist())
print('First 5 rows:')
print(df.head())

# Class distribution
print('\nClass distribution:')
print(df['label'].value_counts())

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

# --- INSIGHTFUL EDA ---
# 1. Feature distributions by label (split histograms)
feature_list = [
    'body_length', 'num_digits', 'num_exclamation_marks', 'num_urls',
    'sender_domain_length', 'num_uppercase_words', 'url_length', 'num_subdomains'
]
for feat in feature_list:
    if feat in df.columns:
        plt.figure(figsize=(8,4))
        sns.histplot(data=df, x=feat, hue='label', bins=30, kde=True, stat='density', common_norm=False, palette='Set1')
        plt.title(f'Distribution of {feat} by Label')
        plt.xlabel(feat)
        plt.ylabel('Density')
        plt.legend(title='Label', labels=['Ham', 'Phishing'])
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, f'{feat}_by_label_hist.png'))
        plt.close()

# 2. Boxplots for numeric features by label
for feat in feature_list:
    if feat in df.columns:
        plt.figure(figsize=(7,4))
        sns.boxplot(x='label', y=feat, data=df, palette='Set2')
        plt.title(f'{feat} by Label (Boxplot)')
        plt.xlabel('Label (0=Ham, 1=Phishing)')
        plt.ylabel(feat)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, f'{feat}_by_label_boxplot.png'))
        plt.close()

# 3. Correlation heatmap for numeric features
numeric_feats = [f for f in feature_list if f in df.columns]
if len(numeric_feats) > 1:
    plt.figure(figsize=(10,8))
    corr = df[numeric_feats + ['label']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap (Numeric Features + Label)')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'correlation_heatmap.png'))
    plt.close()

# 4. Pairplot for top features (if not too many rows)
if len(df) < 2000:
    top_feats = numeric_feats[:4] + ['label']
    sns.pairplot(df[top_feats], hue='label', palette='Set1', plot_kws={'alpha':0.5})
    plt.savefig(os.path.join(BASE_DIR, 'pairplot_top_features.png'))
    plt.close()

# 5. Barplot of mean feature values by label
mean_df = df.groupby('label')[numeric_feats].mean().T
plt.figure(figsize=(10,6))
mean_df.plot(kind='bar', width=0.8)
plt.title('Mean Feature Values by Label')
plt.ylabel('Mean Value')
plt.xlabel('Feature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'mean_feature_values_by_label.png'))
plt.close()

# 6. Categorical/binary features by label
for feat in ['sender_matches_url', 'uses_https', 'has_ip_address']:
    if feat in df.columns:
        plt.figure(figsize=(5,4))
        sns.countplot(x=feat, hue='label', data=df)
        plt.title(f'{feat.replace("_", " ").title()} by Label')
        plt.xlabel(feat.replace('_', ' ').title())
        plt.ylabel('Count')
        plt.legend(title='Label', labels=['Ham', 'Phishing'])
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, f'{feat}_by_label.png'))
        plt.close()

print('\nEDA complete. Insightful plots and summary saved to data/.') 