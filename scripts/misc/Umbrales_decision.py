import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)

# -----------------------------
# Cargar el conjunto de datos
# -----------------------------
# Evita el warning de secuencias de escape en Windows usando raw string (r'...') o '/'.
df = pd.read_csv(r'IA\datasets\pima_indian_diabetes_dataset\cleaned_dataset.csv')

# Dividir en características y objetivo
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modelo
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# Probabilidades para clase positiva
y_pred_proba = rf_classifier.predict_proba(X_test)[:, 1]

# (Opcional) AUC/ROC si lo quieres usar o mostrar luego
fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
auc_val = roc_auc_score(y_test, y_pred_proba)

# ---------------------------------------------------------
# Precalcular curvas de métricas vs umbral (para las líneas)
# ---------------------------------------------------------
thresholds = np.round(np.linspace(0.00, 1.00, 21), 2)  # 0.00, 0.05, ..., 1.00

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for thr in thresholds:
    y_pred_thr = (y_pred_proba >= thr).astype(int)
    accuracy_scores.append(accuracy_score(y_test, y_pred_thr))
    precision_scores.append(precision_score(y_test, y_pred_thr, zero_division=0))
    recall_scores.append(recall_score(y_test, y_pred_thr, zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred_thr, zero_division=0))

# -----------------------------
# Gráfico y deslizador
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)  # espacio para el slider

# Líneas de métricas
(line_accuracy,) = ax.plot(thresholds, accuracy_scores, label='Exactitud')
(line_precision,) = ax.plot(thresholds, precision_scores, label='Precisión')
(line_recall,) = ax.plot(thresholds, recall_scores, label='Sensibilidad')
(line_f1,) = ax.plot(thresholds, f1_scores, label='Puntaje F1')

# Línea vertical que se moverá con el slider
init_thr = 0.50
vertical_line = ax.axvline(x=init_thr, color='gray', linestyle='--', alpha=0.8)

# Etiquetas y estilo
ax.set_xlabel('Umbral')
ax.set_ylabel('Valor de la Métrica')
ax.set_title('Evolución de las Métricas en Función del Umbral')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.25)
legend = ax.legend(loc='lower right', frameon=True)

# Recuadro de texto para mostrar métricas actuales en el umbral
metrics_text = ax.text(
    0.02, 0.98,
    "",
    transform=ax.transAxes,
    va='top', ha='left',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)

# Slider
slider_ax = plt.axes([0.15, 0.10, 0.70, 0.04])  # [left, bottom, width, height]
threshold_slider = Slider(
    ax=slider_ax,
    label='Umbral',
    valmin=0.0,
    valmax=1.0,
    valinit=init_thr,
    valstep=0.01  # paso fino
)

# Función para actualizar recuadro de métricas
def compute_and_format_metrics(thr):
    y_pred_thr = (y_pred_proba >= thr).astype(int)
    acc = accuracy_score(y_test, y_pred_thr)
    prec = precision_score(y_test, y_pred_thr, zero_division=0)
    rec = recall_score(y_test, y_pred_thr, zero_division=0)
    f1 = f1_score(y_test, y_pred_thr, zero_division=0)
    return acc, prec, rec, f1

def update(val):
    threshold = float(val)

    # Mover línea vertical (necesita secuencia)
    vertical_line.set_xdata([threshold, threshold])

    # Recalcular y mostrar métricas en el recuadro
    acc, prec, rec, f1 = compute_and_format_metrics(threshold)
    metrics_text.set_text(
        f"AUC ROC: {auc_val:.3f}\n"
        f"Umbral actual: {threshold:.2f}\n"
        f"Exactitud:  {acc:.3f}\n"
        f"Precisión:  {prec:.3f}\n"
        f"Sensibilidad: {rec:.3f}\n"
        f"Puntaje F1: {f1:.3f}"
    )

    fig.canvas.draw_idle()

# Inicializa el recuadro con el umbral inicial
acc0, prec0, rec0, f10 = compute_and_format_metrics(init_thr)
metrics_text.set_text(
    f"AUC ROC: {auc_val:.3f}\n"
    f"Umbral actual: {init_thr:.2f}\n"
    f"Exactitud:  {acc0:.3f}\n"
    f"Precisión:  {prec0:.3f}\n"
    f"Sensibilidad: {rec0:.3f}\n"
    f"Puntaje F1: {f10:.3f}"
)

threshold_slider.on_changed(update)

plt.show()
