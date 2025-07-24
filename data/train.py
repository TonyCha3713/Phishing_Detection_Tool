import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load features
features_csv = os.path.join(BASE_DIR, 'combined_features.csv')
df = pd.read_csv(features_csv)
# Fill NaNs
df['body'] = df['body'].fillna('')
df['subject'] = df['subject'].fillna('')

# TF-IDF for body
tfidf_body = TfidfVectorizer(max_features=100)
body_tfidf = tfidf_body.fit_transform(df['body'])
body_tfidf_df = pd.DataFrame(body_tfidf.toarray(), columns=[f'body_tfidf_{i}' for i in range(body_tfidf.shape[1])])

# TF-IDF for subject
tfidf_subject = TfidfVectorizer(max_features=50)
subject_tfidf = tfidf_subject.fit_transform(df['subject'])
subject_tfidf_df = pd.DataFrame(subject_tfidf.toarray(), columns=[f'subject_tfidf_{i}' for i in range(subject_tfidf.shape[1])])

# Drop raw text columns
df = df.drop(['body', 'subject', 'text'], axis=1, errors='ignore')

# Combine all features
X = pd.concat([df.drop('label', axis=1).reset_index(drop=True), body_tfidf_df, subject_tfidf_df], axis=1)
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Use best XGBoost parameters from tuning
tuned_xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    subsample=1.0,
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.1,
)
tuned_xgb.fit(X_train, y_train)
y_pred_xgb = tuned_xgb.predict(X_test)

# Evaluation
print('\nXGBoost Results:')
print(classification_report(y_test, y_pred_xgb))
# Save the model
model_path = os.path.join(BASE_DIR, '../notebooks/best_model.pkl')
joblib.dump(tuned_xgb, model_path)
print(f'\nTuned XGBoost model saved to {model_path}')

# Get feature importances and feature names
importances = tuned_xgb.feature_importances_
feature_names = X.columns

# Get indices of top 20 features
top_indices = np.argsort(importances)[::-1][:20]

# For body
body_vocab = tfidf_body.get_feature_names_out()
# For subject
subject_vocab = tfidf_subject.get_feature_names_out()

# Save TF-IDF index-to-word mappings to files
body_vocab_path = os.path.join(BASE_DIR, 'body_tfidf_mapping.csv')
subject_vocab_path = os.path.join(BASE_DIR, 'subject_tfidf_mapping.csv')

pd.Series(body_vocab).to_csv(body_vocab_path, index_label='index', header=['word'])
pd.Series(subject_vocab).to_csv(subject_vocab_path, index_label='index', header=['word'])

print(f"Body TF-IDF vocabulary saved to {body_vocab_path}")
print(f"Subject TF-IDF vocabulary saved to {subject_vocab_path}")

# Plot top 20 feature importances
plt.figure(figsize=(10, 6))
top_features = [feature_names[idx] for idx in top_indices]
top_importances = importances[top_indices]
plt.barh(top_features[::-1], top_importances[::-1])  # reverse for descending order
plt.xlabel('Importance')
plt.title('Top 20 XGBoost Feature Importances')
plt.tight_layout()
feature_importance_path = os.path.join(BASE_DIR, 'xgboost_feature_importance.png')
plt.savefig(feature_importance_path)
plt.close()
print(f'Feature importance plot saved to {feature_importance_path}')

joblib.dump(tfidf_body, os.path.join(BASE_DIR, 'tfidf_body_vectorizer.pkl'))
joblib.dump(tfidf_subject, os.path.join(BASE_DIR, 'tfidf_subject_vectorizer.pkl'))
print("TF-IDF vectorizers saved.") 
