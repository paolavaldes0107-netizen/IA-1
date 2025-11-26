import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from sklearn.svm import SVC

# Enable interactive mode
plt.ion()
fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')  # White background
plt.pause(2)

def train_and_plot_svm(df, n_samples, title_suffix):
    ax.clear()
    ax.set_facecolor('white')

    # Transform coordinates
    features = df[['cx', 'cy', 'label']].copy()
    features['cx'] = features['cx'] - 50
    features['cy'] = 980 - features['cy'] + 50

    # Select samples
    samples = features.groupby('label').head(n_samples)
    X_train = samples[['cx', 'cy']].values
    y_train = samples['label'].map({'rectangle': 0, 'circle': 1}).values
    labels = samples['label'].values

    # Train linear SVM
    clf = SVC(kernel='linear', C=1)
    clf.fit(X_train, y_train)

    # Decision boundary and margins
    xx, yy = np.meshgrid(np.linspace(0, 1820, 500),
                         np.linspace(0, 980, 500))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision boundary and margins
    ax.contour(xx, yy, Z, levels=[0], colors='purple', linewidths=2)  # Decision boundary
    ax.contour(xx, yy, Z, levels=[-1, 1], linestyles='dashed', colors='gray', linewidths=1)  # Margins

    # Draw shapes
    for i, (x, y) in enumerate(X_train):
        if labels[i] == 'rectangle':
            rect = patches.Rectangle((x - 20, y - 20), 40, 40,
                                     linewidth=1, edgecolor='blue',
                                     facecolor='blue', alpha=0.7)
            ax.add_patch(rect)
        else:
            circ = patches.Circle((x, y), 20,
                                  linewidth=1, edgecolor='red',
                                  facecolor='red', alpha=0.7)
            ax.add_patch(circ)

    # Draw support vectors
    sv = clf.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=100,
               facecolors='none', edgecolors='black',
               linewidths=1.5, label='Support Vectors')

    # Axis settings
    ax.set_xlim(0, 1820)
    ax.set_ylim(0, 980)
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_title(f'SVM Decision Boundary - Samples per Class ({title_suffix})')
    ax.grid(True)

    # Legend
    legend_elements = [
        Line2D([0], [0], color='purple', lw=2, label='Decision Boundary'),
        Line2D([0], [0], color='gray', lw=1, linestyle='--', label='Margins'),
        Line2D([0], [0], marker='o', color='w', label='Support Vectors',
               markerfacecolor='none', markeredgecolor='black', markersize=10)
    ]
    ax.legend(handles=legend_elements)

    plt.draw()
    plt.pause(0.05)

# Load data
df = pd.read_csv(r"IA\scripts\support_vectors_machine\SVG_Shape_Features.csv")

# Determine minimum samples per class
min_class_count = df['label'].value_counts().min()

# Iterate over increasing sample sizes
for n in range(2, min_class_count + 1):
    train_and_plot_svm(df, n_samples=n, title_suffix=f"Round {n - 1}")

# Keep final plot open
plt.ioff()
plt.show()
