import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

# Cargar el conjunto de datos
df = pd.read_csv('IA\datasets\pima_indian_diabetes_dataset\cleaned_dataset.csv')

# Dividir el conjunto de datos en características y objetivo
X = df.drop(columns=['Outcome'])  # Reemplazar 'target' con el nombre real de la columna objetivo
y = df['Outcome']  # Reemplazar 'target' con el nombre real de la columna objetivo

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el clasificador Random Forest
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Predecir probabilidades para el conjunto de prueba
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

# Calcular el puntaje ROC AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {roc_auc}")

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'Curva ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Estimación Aleatoria')
plt.xlabel('1 - Especificidad')  # La tasa de falsos positivos corresponde a 1 - Especificidad
plt.ylabel('Sensibilidad')       # La tasa de verdaderos positivos corresponde a Sensibilidad
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

