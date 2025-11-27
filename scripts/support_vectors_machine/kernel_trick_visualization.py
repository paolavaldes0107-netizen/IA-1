import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC

# Cargar y transformar datos
df = pd.read_csv(r"IA\datasets\mnist\mnist_test.csv")
# df = pd.read_csv(r"IA\scripts\support_vectors_machine\SVG_Shape_Features.csv")
df['cx'] = df['cx'] - 50
df['cy'] = 980 - df['cy'] + 50

# Preparar datos
rects = df[df['label'] == 'rectangle']
circles = df[df['label'] == 'circle']
samples = pd.concat([rects, circles])
X = samples[['cx', 'cy']].values
y = samples['label'].map({'rectangle': -1, 'circle': 1}).values
labels = samples['label'].values

# Usaremos una proyección no lineal simple para ilustrar el kernel trick
gamma = 0.00005
z = np.exp(-gamma * np.sum((X - np.mean(X, axis=0))**2, axis=1))  # Proyección tipo RBF

# Entrenar SVM (en 2D, pero para simular separación)
clf = SVC(kernel='linear')
clf.fit(np.c_[X, z], y)  # Entrenar en 3D con proyección manual

# Graficar en 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Dibujar puntos
for i in range(len(X)):
    if labels[i] == 'rectangle':
        ax.scatter(X[i, 0], X[i, 1], z[i], c='blue', s=40)
    else:
        ax.scatter(X[i, 0], X[i, 1], z[i], c='red', s=40)

# Hiperplano (estimado) en 3D
xx, yy = np.meshgrid(np.linspace(0, 1820, 50),
                     np.linspace(0, 980, 50))
zz = (-clf.intercept_[0] - clf.coef_[0][0]*xx - clf.coef_[0][1]*yy) / clf.coef_[0][2]
ax.plot_surface(xx, yy, zz, alpha=0.3, color='purple')

# Etiquetas
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z (proyección no lineal)')
ax.set_title('Representación 3D del kernel trick (RBF simulado)')

plt.show()
