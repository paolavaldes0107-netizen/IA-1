import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from skopt import BayesSearchCV
import time

# Cargar el conjunto de datos
df = pd.read_csv('../../datasets/pima_indian_diabetes_dataset/cleaned_dataset.csv')

# Dividir características y objetivo
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Definir el modelo Random Forest
clf = RandomForestClassifier(random_state=42)

# Definir el espacio de búsqueda de hiperparámetros
search_space = {
    'n_estimators': (10, 500),
    'max_depth': (3, 50),  # Replaced None with 50 as an upper bound
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 8),
    'max_features': ['sqrt', 'log2'],  # Removed 'auto' as it is not valid
    'bootstrap': [True, False]
}

# Configurar BayesSearchCV para buscar los mejores hiperparámetros con salida detallada
bayes_search = BayesSearchCV(estimator=clf, search_spaces=search_space, cv=5, scoring='roc_auc', verbose=3, n_iter=50, random_state=42)

# Medir el tiempo de ajuste del modelo
start_time = time.time()
bayes_search.fit(X, y)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Tiempo de cálculo: {elapsed_time:.2f} segundos")

# Obtener los mejores parámetros y el mejor puntaje AUC
best_params = bayes_search.best_params_
best_auc = bayes_search.best_score_

print("Mejores hiperparámetros:")
print(best_params)
print(f"Mejor AUC obtenido: {best_auc:.4f}")
