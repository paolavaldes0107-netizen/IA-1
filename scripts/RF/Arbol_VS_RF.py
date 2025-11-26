from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import tree as sktree

# 1. Dataset sencillo 2D
df = pd.read_csv(r'IA\datasets\pima_indian_diabetes_dataset\cleaned_dataset.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Árbol de decisión "libre"
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# 3. Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42
)
rf.fit(X_train, y_train)

# 4. Accuracy
for model, name in [(tree, "Árbol"), (rf, "Random Forest")]:
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print(f"{name}:")
    print("  Train acc:", accuracy_score(y_train, y_pred_train))
    print("  Test  acc:", accuracy_score(y_test, y_pred_test))

    # colectar resultados y plotear cuando estén ambos modelos
    try:
        _results
    except NameError:
        _results = {'names': [], 'train': [], 'test': []}

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    _results['names'].append(name)
    _results['train'].append(train_acc)
    _results['test'].append(test_acc)

    if len(_results['names']) == 2:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1) Barras comparando Train vs Test
        x = range(len(_results['names']))
        axes[0].bar([i - 0.15 for i in x], _results['train'], width=0.3, label='Train', color='C0')
        axes[0].bar([i + 0.15 for i in x], _results['test'],  width=0.3, label='Test',  color='C1')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(_results['names'])
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].set_title('Train vs Test Accuracy')

        # 2) Visualizar el árbol de decisión entrenado (se limita la profundidad para legibilidad)
        sktree.plot_tree(tree, feature_names=X.columns, class_names=[str(c) for c in sorted(y.unique())],
                         filled=True, ax=axes[1], max_depth=3)
        axes[1].set_title('Árbol (vista parcial, max_depth=3)')

        # 3) Visualizar un árbol ejemplo del Random Forest
        sktree.plot_tree(rf.estimators_[0], feature_names=X.columns, class_names=[str(c) for c in sorted(y.unique())],
                         filled=True, ax=axes[2], max_depth=3)
        axes[2].set_title('Ejemplo de árbol en Random Forest (vista parcial, max_depth=3)')

        plt.tight_layout()
        plt.show()
