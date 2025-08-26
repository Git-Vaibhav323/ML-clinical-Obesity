import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import pickle

# Load data
df = pd.read_csv('final_dataset.csv')

# Drop rows with NaN label or features needed for modeling
df = df.dropna(subset=['clinical_obesity_label', 'BMI', 'adl_limitation_flag', 'organ_dysfunction_flag'])

# Features and target
y = df['clinical_obesity_label'].astype(int)
X = df[['BMI', 'adl_limitation_flag', 'organ_dysfunction_flag']]

# Optionally add more features if available
for col in ['age', 'sex', 'region']:
    if col in df.columns:
        X[col] = df[col]

# Encode categorical features if present (ensure X is a DataFrame)
if 'sex' in X.columns and not isinstance(X, np.ndarray):
    X.loc[:, 'sex'] = X['sex'].map({'M': 0, 'F': 1}).astype('Int64').fillna(-1).astype(int)
if 'region' in X.columns and not isinstance(X, np.ndarray):
    X.loc[:, 'region'] = X['region'].astype('category').cat.codes.astype('Int64').fillna(-1).astype(int)

# Ensure all columns are numeric for XGBoost
X = X.apply(pd.to_numeric, errors='ignore')

# Train/test split (use 0.5 for small dataset)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=1),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
}

results = []
best_f1 = -1
best_model = None
best_model_name = ''

print('Training and evaluating models...')
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division='warn')
    rec = recall_score(y_test, y_pred, average='weighted', zero_division='warn')
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division='warn')
    results.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-score': f1})
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

results_df = pd.DataFrame(results)
print('\nModel Comparison:')
print(results_df)

# Save comparison table for Streamlit app
results_df.to_csv('model_comparison.csv', index=False)

# Save best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f'Best model ({best_model_name}) saved as best_model.pkl') 