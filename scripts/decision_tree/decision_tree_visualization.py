"""
Script: Visualización Básica de Árbol de Decisión
=================================================

Propósito:
    Crea y visualiza un árbol de decisión básico para el dataset de diabetes,
    exportando la visualización en formato SVG para análisis detallado.

Funcionalidad:
    - Carga dataset limpio de diabetes
    - Entrena árbol de decisión con profundidad controlada
    - Genera visualización completa del árbol
    - Exporta en formato SVG para zoom sin pérdida de calidad

Conceptos clave:
    - Visualización de árboles de decisión
    - Control de profundidad para legibilidad
    - Exportación en formatos vectoriales

Dataset: cleaned_dataset.csv (diabetes)
Target: Predicción de diabetes (Outcome)
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Cargar dataset limpio de diabetes
df = pd.read_csv('../../../datasets/pima_indian_diabetes_dataset/cleaned_dataset.csv')

# Separar características y variable objetivo
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# División en conjunto de entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar árbol de decisión con profundidad limitada para visualización clara
clf = DecisionTreeClassifier(max_depth=6, random_state=42)
clf.fit(X_train, y_train)

# Crear visualización del árbol con formato SVG para zoom sin pérdida
fig, ax = plt.subplots(figsize=(40, 20))  # Figura grande para visualización detallada
plot_tree(clf, 
          filled=True,                           # Nodos coloreados por clase mayoritaria
          feature_names=X.columns,               # Nombres de las características
          class_names=['Sin Diabetes', 'Diabetes'], # Etiquetas de las clases
          rounded=True,                          # Esquinas redondeadas en nodos
          fontsize=10,                           # Tamaño de fuente legible
          ax=ax)
plt.tight_layout()
plt.savefig("decision_tree.svg", format="svg")  # Guardar como SVG para zoom infinito
plt.show()

# Nota: La visualización SVG permite hacer zoom para examinar
# nodos específicos y reglas de decisión en detalle
