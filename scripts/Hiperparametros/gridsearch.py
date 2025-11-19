import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# Cargar el conjunto de datos
df = pd.read_csv('../../datasets/pima_indian_diabetes_dataset/cleaned_dataset.csv')

# Dividir características y objetivo
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Definir el modelo Decision Tree
clf = DecisionTreeClassifier(random_state=42)

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    'max_depth': [3, 5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8]
}

# Configurar GridSearchCV para buscar los mejores hiperparámetros con salida detallada
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='roc_auc', verbose=3)

# Ajustar el modelo con los datos
grid_search.fit(X, y)

# Obtener los mejores parámetros y el mejor puntaje AUC
best_params = grid_search.best_params_
best_auc = grid_search.best_score_

print("Mejores hiperparámetros:")
print(best_params)
print(f"Mejor AUC obtenido: {best_auc:.4f}")