import pandas as pd
import os

# Load SMS spam dataset
data_path = os.path.join(os.path.dirname(__file__), 'spam.csv')
df = pd.read_csv(data_path, encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']

# Map labels: spam=1, ham=0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Shuffle and save
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
output_path = os.path.join(os.path.dirname(__file__), 'emails_preprocessed.csv')
df.to_csv(output_path, index=False)
print(f"Preprocessed data saved to {output_path}")
