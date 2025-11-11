import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from pathlib import Path

# Cargar el conjunto de datos
csv_path = Path(__file__).resolve().parents[2] / 'datasets' / 'pima_indian_diabetes_dataset' / 'cleaned_dataset.csv'
df = pd.read_csv(csv_path)

# Dividir características y objetivo
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Inicializar lista para almacenar valores de AUC
auc_values = []

# Configurar K-Fold con 5 divisiones
kf = KFold(n_splits=5, shuffle=True)

# Iterar sobre cada división
fold = 1
for train_index, test_index in kf.split(X):
    print("-----------------------------")
    print(f"Fold {fold}:")
    fold += 1
    # Dividir datos en entrenamiento y prueba
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Entrenar el modelo Random Forest
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # Predecir probabilidades para calcular AUC
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Calcular AUC
    auc = roc_auc_score(y_test, y_pred_proba)
    auc_values.append(auc)
    print(f"AUC: {auc:.4f}\n")

    # Imprimir balance de clases en el conjunto de entrenamiento y prueba
    train_class_balance = y_train.value_counts(normalize=True)
    test_class_balance = y_test.value_counts(normalize=True)
    print("-----------------------------")
    print("Balance de clases en entrenamiento:")
    print(train_class_balance)
    print("Balance de clases en prueba:")
    print(test_class_balance)
    print("-----------------------------")

# Calcular y mostrar el mínimo, máximo y promedio de AUC
min_auc = min(auc_values)
max_auc = max(auc_values)
avg_auc = sum(auc_values) / len(auc_values)
print(f"AUC mínimo: {min_auc:.4f}")
print(f"AUC máximo: {max_auc:.4f}")
print(f"AUC promedio: {avg_auc:.4f}")
