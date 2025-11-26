from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


df = pd.read_csv(r'IA\datasets\pima_indian_diabetes_dataset\cleaned_dataset.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

rf = RandomForestClassifier(
    n_estimators=200,
    oob_score=True,           # activar OOB
    bootstrap=True,           # necesario para OOB
    random_state=42
)
rf.fit(X_train, y_train)

# OOB
print("OOB score:", rf.oob_score_)

# Comparar con test
y_pred_test = rf.predict(X_test)
print("Test acc:", accuracy_score(y_test, y_pred_test))

import matplotlib.pyplot as plt

n_estimators_range = range(10, 201, 10)
oob_errors = []

rf_ws = RandomForestClassifier(warm_start=True, oob_score=True, bootstrap=True, random_state=42, n_jobs=-1)

for n in n_estimators_range:
    rf_ws.set_params(n_estimators=n)
    rf_ws.fit(X_train, y_train)
    oob_errors.append(1 - rf_ws.oob_score_)

plt.figure(figsize=(8, 5))
plt.plot(list(n_estimators_range), oob_errors, marker='o', linestyle='-')
plt.xlabel('Number of trees (n_estimators)')
plt.ylabel('OOB error (1 - oob_score)')
plt.title('OOB error vs number of trees')
plt.grid(True)
plt.show()