"""
Script: Análisis de Overfitting en Árboles de Decisión
======================================================

Propósito:
    Demuestra el fenómeno de overfitting al variar la profundidad del árbol
    y evalúa múltiples métricas para encontrar la profundidad óptima.

Funcionalidad:
    - Entrena árboles con diferentes profundidades (2-14)
    - Evalúa múltiples métricas: AUC, Accuracy, Precision, Recall, F1
    - Compara rendimiento en entrenamiento vs. prueba
    - Visualiza curvas de validación para detectar overfitting

Conceptos clave:
    - Overfitting: Memorización excesiva de datos de entrenamiento
    - Bias-Variance Tradeoff: Balance entre sesgo y varianza
    - Curvas de validación: Herramienta para detectar overfitting
    - Profundidad óptima: Punto de mejor generalización

Señales de Overfitting:
    - Gran diferencia entre métricas de entrenamiento y prueba
    - Métricas de entrenamiento muy altas, prueba estancadas/decrecientes

Dataset: cleaned_dataset.csv (diabetes)
Target: Predicción de diabetes (Outcome)
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)

# Cargar dataset limpio de diabetes
df = pd.read_csv('../../../datasets/pima_indian_diabetes_dataset/full_dataset.csv')

# Separar características y variable objetivo
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# División estratificada en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=44
)

# Almacenar resultados para cada profundidad
results = []

# Evaluar árboles con diferentes profundidades (2 a 14)
print("=== ANÁLISIS DE OVERFITTING POR PROFUNDIDAD ===")
print("Evaluando profundidades de 2 a 14...")
print()

for depth in range(2, 15):
    # Entrenar árbol con profundidad específica
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)

    # Generar predicciones para ambos conjuntos
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    y_test_proba = clf.predict_proba(X_test)[:, 1]

    # Calcular múltiples métricas de evaluación
    results.append({
        "Profundidad": depth,
        "AUC_Entrenamiento": roc_auc_score(y_train, y_train_proba),
        "AUC_Prueba": roc_auc_score(y_test, y_test_proba),
        "Accuracy_Entrenamiento": accuracy_score(y_train, y_train_pred),
        "Accuracy_Prueba": accuracy_score(y_test, y_test_pred),
        "Precision_Entrenamiento": precision_score(y_train, y_train_pred),
        "Precision_Prueba": precision_score(y_test, y_test_pred),
        "Recall_Entrenamiento": recall_score(y_train, y_train_pred),
        "Recall_Prueba": recall_score(y_test, y_test_pred),
        "F1_Entrenamiento": f1_score(y_train, y_train_pred),
        "F1_Prueba": f1_score(y_test, y_test_pred),
    })

# Convertir resultados a DataFrame para análisis
results_df = pd.DataFrame(results)

# Mostrar tabla completa de resultados
print("TABLA DE RESULTADOS:")
print(results_df.round(4))
print()

# Identificar profundidad óptima (máximo AUC en prueba)
optimal_depth = results_df.loc[results_df["AUC_Prueba"].idxmax(), "Profundidad"]
optimal_auc = results_df.loc[results_df["AUC_Prueba"].idxmax(), "AUC_Prueba"]
print(f"Profundidad óptima: {optimal_depth} (AUC Prueba: {optimal_auc:.4f})")

# Visualizar curva de validación para AUC
plt.figure(figsize=(10, 6))
plt.plot(results_df["Profundidad"], results_df["AUC_Entrenamiento"], 
         label="AUC Entrenamiento", marker='o', color='blue')
plt.plot(results_df["Profundidad"], results_df["AUC_Prueba"], 
         label="AUC Prueba", marker='o', color='red')

# Marcar profundidad óptima
plt.axvline(x=optimal_depth, color='green', linestyle='--', alpha=0.7,
            label=f'Óptimo (Profundidad {optimal_depth})')

plt.title("Curva de Validación: AUC vs. Profundidad del Árbol")
plt.xlabel("Profundidad del Árbol")
plt.ylabel("AUC Score")
plt.xticks(results_df["Profundidad"])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Análisis de overfitting:
# - Si AUC_Entrenamiento >> AUC_Prueba: Overfitting
# - Si ambas curvas están muy juntas: Buen balance
# - Profundidad óptima: Máximo AUC en prueba antes de divergencia
