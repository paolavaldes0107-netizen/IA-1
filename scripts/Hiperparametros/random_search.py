import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Cargar el conjunto de datos
df = pd.read_csv('../../datasets/pima_indian_diabetes_dataset/cleaned_dataset.csv')

# Dividir características y objetivo
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Definir el modelo Random Forest
clf = RandomForestClassifier(random_state=42)

# Definir el espacio de búsqueda de hiperparámetros
param_dist = {
    'n_estimators': [10, 50, 100, 200, 500],
    'max_depth': [3, 5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

# Configurar RandomizedSearchCV para buscar los mejores hiperparámetros con salida detallada
random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, n_iter=25, cv=5, scoring='roc_auc', verbose=3)

# Ajustar el modelo con los datos
random_search.fit(X, y)

# Obtener los mejores parámetros y el mejor puntaje AUC
best_params = random_search.best_params_
best_auc = random_search.best_score_

print("Mejores hiperparámetros:")
print(best_params)
print(f"Mejor AUC obtenido: {best_auc:.4f}")