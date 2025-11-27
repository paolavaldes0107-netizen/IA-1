import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# 1. Cargar dataset
df = pd.read_csv(r'IA\datasets\pima_indian_diabetes_dataset\cleaned_dataset.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Entrenar RF
rf = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    random_state=42
)
rf.fit(X_train, y_train)

# 3. Importancia de características
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 10 características más importantes:")
for i in range(8):
    idx = indices[i]
    print(f"{i+1}. {feature_names[idx]} -> {importances[idx]:.4f}")
