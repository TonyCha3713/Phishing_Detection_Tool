import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt

# Load features
df = pd.read_csv('data/features.csv')
X = df.drop('label', axis=1)
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Use best XGBoost parameters from tuning
tuned_xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    subsample=1.0,
    n_estimators=300,
    max_depth=8,
    learning_rate=0.2,
    colsample_bytree=0.9
)
tuned_xgb.fit(X_train, y_train)
y_pred_xgb = tuned_xgb.predict(X_test)

# Evaluation
print('\nXGBoost Results:')
print(classification_report(y_test, y_pred_xgb))
# Save the model
model_path = os.path.join('notebooks', 'best_model.pkl')
joblib.dump(tuned_xgb, model_path)
print(f'\nTuned XGBoost model saved to {model_path}')

# Plot feature importances
plt.figure(figsize=(8, 5))
plot_importance(tuned_xgb, ax=plt.gca(), importance_type='weight', show_values=False)
plt.title('XGBoost Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join('data', 'xgboost_feature_importance.png'))
plt.close()
print('Feature importance plot saved to data/xgboost_feature_importance.png') 