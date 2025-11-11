import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from pathlib import Path

# Cargar el conjunto de datos
# Cargar el conjunto de datos (ruta relativa desde este archivo)
csv_path = Path(__file__).resolve().parents[2] / 'datasets' / 'pima_indian_diabetes_dataset' / 'cleaned_dataset.csv'
df = pd.read_csv(csv_path)

# Dividir características y objetivo
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Inicializar lista para almacenar valores de AUC
auc_values = []

# Ejecutar 20 iteraciones con diferentes divisiones aleatorias
for i in range(100):
    print(f"Ejecución {i + 1}:")
    # División 80/20 de entrenamiento-prueba con estado aleatorio variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i, stratify=y)

    # Entrenar el modelo Random Forest
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predecir probabilidades para calcular AUC
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Calcular AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    auc_values.append(auc)
    print(f"AUC: {auc:.4f}\n")

# Calcular y mostrar el mínimo, máximo y promedio de AUC
min_auc = min(auc_values)
max_auc = max(auc_values)
avg_auc = sum(auc_values) / len(auc_values)
print(f"AUC mínimo: {min_auc:.4f}")
print(f"AUC máximo: {max_auc:.4f}")
print(f"AUC promedio: {avg_auc:.4f}")