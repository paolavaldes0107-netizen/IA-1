"""
Script: Análisis de Importancia de Características
==================================================

Propósito:
    Calcula y muestra la importancia relativa de cada característica
    en la predicción del árbol de decisión.

Funcionalidad:
    - Entrena árbol de decisión con profundidad limitada
    - Extrae importancias de características del modelo
    - Ordena características por importancia descendente
    - Muestra ranking de características más influyentes

Conceptos clave:
    - Feature Importance: Medida de relevancia de cada variable
    - Gini Importance: Reducción promedio de impureza por característica
    - Selección de características basada en importancia

Interpretación:
    - Valores más altos = características más importantes
    - Suma total de importancias = 1.0
    - Útil para reducción de dimensionalidad

Dataset: cleaned_dataset.csv (diabetes)
Target: Predicción de diabetes (Outcome)
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Cargar dataset limpio de diabetes
df = pd.read_csv('../../../datasets/pima_indian_diabetes_dataset/cleaned_dataset.csv')

# Separar características y variable objetivo
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo con profundidad limitada para interpretabilidad
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Extraer importancias de características y crear ranking
importance_df = pd.DataFrame({
    "Característica": X.columns,
    "Importancia": clf.feature_importances_
}).sort_values(by="Importancia", ascending=False)

print("=== RANKING DE IMPORTANCIA DE CARACTERÍSTICAS ===")
print("(Basado en reducción de impureza Gini)")
print()
print(importance_df.to_string(index=False))
print()
print(f"Suma total de importancias: {clf.feature_importances_.sum():.3f}")

# Interpretación de resultados:
# - Características con mayor importancia tienen más influencia en las decisiones
# - Valores cercanos a 0 indican características poco relevantes
# - Útil para selección de características y reducción de dimensionalidad
