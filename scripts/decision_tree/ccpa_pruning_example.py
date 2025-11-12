"""
Script: Poda por Complejidad de Costo (Cost Complexity Pruning)
===============================================================

Propósito:
    Demuestra la técnica de poda por complejidad de costo para optimizar
    árboles de decisión y evitar el overfitting.

Funcionalidad:
    - Implementa poda post-entrenamiento usando el parámetro ccp_alpha
    - Evalúa múltiples valores de alpha para encontrar el óptimo
    - Compara rendimiento en entrenamiento vs. prueba usando AUC
    - Visualiza la curva de validación para selección de hiperparámetros

Conceptos clave:
    - Cost Complexity Pruning: Técnica para reducir overfitting
    - ccp_alpha: Parámetro de regularización para poda
    - Curva de validación: Herramienta para selección de hiperparámetros
    - AUC (Area Under Curve): Métrica para clasificación binaria

Dataset: pima_indian_diabetes_dataset
Target: Predicción de diabetes (Outcome)
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Cargar dataset de diabetes de indios Pima
df = pd.read_csv('../../../datasets/pima_indian_diabetes_dataset/full_dataset.csv')

# Separar características y variable objetivo
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Obtener la ruta de poda (pruning path) con diferentes valores de alpha
clf = DecisionTreeClassifier(random_state=42, max_depth=5)  # Limitar profundidad inicial
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# Entrenar un árbol para cada valor de alpha
train_auc = []  # AUC en entrenamiento
test_auc = []   # AUC en prueba
trees = []      # Almacenar modelos entrenados

for alpha in ccp_alphas:
    # Crear modelo con valor específico de ccp_alpha
    model = DecisionTreeClassifier(random_state=42, ccp_alpha=alpha, max_depth=10)
    model.fit(X_train, y_train)
    trees.append(model)

    # Obtener probabilidades de predicción para calcular AUC
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular AUC para entrenamiento y prueba
    train_auc.append(roc_auc_score(y_train, y_train_proba))
    test_auc.append(roc_auc_score(y_test, y_test_proba))

# Visualizar AUC vs. ccp_alpha para identificar el valor óptimo
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_auc, marker='o', label='AUC Entrenamiento')
plt.plot(ccp_alphas, test_auc, marker='o', label='AUC Prueba')
plt.xlabel("ccp_alpha (Parámetro de Poda)")
plt.ylabel("AUC Score")
plt.title("Curva de Validación: AUC vs. ccp_alpha (max_depth=10)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Nota: El valor óptimo de ccp_alpha es donde el AUC de prueba es máximo
# sin una gran diferencia con el AUC de entrenamiento
