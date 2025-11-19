import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import time

# Cargar el conjunto de datos
df = pd.read_csv('../../datasets/pima_indian_diabetes_dataset/cleaned_dataset.csv')

# Dividir características y objetivo
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Definir el modelo Random Forest
clf = RandomForestClassifier(random_state=42)

# Definir el espacio de búsqueda de hiperparámetros
param_grid = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': [3, 5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Configurar GridSearchCV para buscar los mejores hiperparámetros con salida detallada
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='roc_auc', verbose=3)

# Medir el tiempo de ajuste del modelo
start_time = time.time()
grid_search.fit(X, y)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Tiempo de cálculo: {elapsed_time:.2f} segundos")

# Obtener los mejores parámetros y el mejor puntaje AUC
best_params = grid_search.best_params_
best_auc = grid_search.best_score_

print("Mejores hiperparámetros:")
print(best_params)
print(f"Mejor AUC obtenido: {best_auc:.4f}")