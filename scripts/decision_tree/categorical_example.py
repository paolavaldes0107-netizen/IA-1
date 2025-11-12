"""
Script: Árbol de Decisión con Variables Categóricas
==================================================

Propósito:
    Demuestra cómo manejar variables categóricas en árboles de decisión usando
    codificación de etiquetas (Label Encoding) y visualizar el árbol resultante.

Funcionalidad:
    - Carga el dataset de enfermedades cardíacas
    - Identifica y codifica variables categóricas
    - Entrena un árbol de decisión con profundidad limitada
    - Genera visualización del árbol en formato SVG para mejor zoom

Conceptos clave:
    - Label Encoding: Convierte categorías de texto a números
    - Visualización de árboles de decisión
    - Manejo de variables categóricas en ML

Dataset: heart_disease.csv
Target: Predicción de enfermedad cardíaca (HeartDisease)
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Cargar el dataset de enfermedades cardíacas
df = pd.read_csv('../../../datasets/heart_disease_dataset/heart_disease.csv')

# Identificar columnas categóricas (tipo object)
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print(f"Columnas categóricas encontradas: {categorical_columns}")

# Aplicar codificación de etiquetas a las variables categóricas
# Label Encoding convierte categorías de texto a valores numéricos
label_encoders = {}
df_encoded = df.copy()

for column in categorical_columns:
    if column != 'HeartDisease':  # No codificar aún la variable objetivo
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df[column])
        label_encoders[column] = le
        print(f"Codificación de {column}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Separar características (features) y variable objetivo (target)
X = df_encoded.drop(columns=['HeartDisease'])
y = df_encoded['HeartDisease']

# Codificar la variable objetivo si es categórica
if y.dtype == 'object':
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    class_names = target_encoder.classes_
else:
    class_names = ['Sin Enfermedad', 'Con Enfermedad']

# División en conjunto de entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar árbol de decisión con profundidad limitada para evitar overfitting
clf = DecisionTreeClassifier(max_depth=6, random_state=42)
clf.fit(X_train, y_train)

# Exportar el árbol como SVG para permitir zoom y mejor visualización
fig, ax = plt.subplots(figsize=(40, 20))  # Figura grande para visualización detallada
plot_tree(clf, 
          filled=True,              # Nodos coloreados según la clase
          feature_names=X.columns,  # Nombres de las características
          class_names=class_names,  # Nombres de las clases
          rounded=True,             # Nodos con esquinas redondeadas
          fontsize=10,
          ax=ax)
plt.tight_layout()
plt.savefig("decision_tree_categorical.svg", format="svg")  # Guardar como SVG para zoom
plt.show()
