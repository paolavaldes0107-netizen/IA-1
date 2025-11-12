# Árboles de Decisión - Scripts de Análisis

Este directorio contiene una colección de scripts en Python para el análisis y entrenamiento de árboles de decisión, abarcando desde conceptos básicos hasta técnicas avanzadas de optimización.

## Estructura de Archivos

```
decision_tree/
├── categorical_example.py          # Manejo de variables categóricas
├── ccpa_pruning_example.py        # Poda por complejidad de costo
├── decision_tree_visualization.py # Visualización básica
├── feature_importance.py          # Análisis de importancia
├── overfitting_example.py         # Análisis de overfitting
└── readme.md                      # Este archivo
```

## Descripción de Scripts

### 1. `categorical_example.py`
**Propósito**: Demuestra el manejo de variables categóricas en árboles de decisión.

**Características principales**:
- Codificación de variables categóricas usando Label Encoding
- Entrenamiento de árbol con dataset de enfermedades cardíacas
- Visualización en formato SVG para análisis detallado
- Control de profundidad para evitar overfitting

**Dataset**: `heart_disease.csv`
**Target**: Predicción de enfermedad cardíaca

**Conceptos cubiertos**:
- Label Encoding para variables categóricas
- Visualización de árboles de decisión
- Exportación en formatos vectoriales

### 2. `ccpa_pruning_example.py`
**Propósito**: Implementa técnicas de poda post-entrenamiento para optimizar árboles.

**Características principales**:
- Poda por complejidad de costo (Cost Complexity Pruning)
- Evaluación de múltiples valores de `ccp_alpha`
- Curvas de validación para selección de hiperparámetros
- Comparación de rendimiento entrenamiento vs. prueba

**Dataset**: `pima_indian_diabetes_dataset`
**Target**: Predicción de diabetes

**Conceptos cubiertos**:
- Cost Complexity Pruning
- Parámetro de regularización `ccp_alpha`
- Curvas de validación
- Prevención de overfitting

### 3. `decision_tree_visualization.py`
**Propósito**: Visualización básica y exportación de árboles de decisión.

**Características principales**:
- Entrenamiento de árbol básico
- Visualización completa del árbol
- Exportación en formato SVG
- Control de profundidad para legibilidad

**Dataset**: `cleaned_dataset.csv` (diabetes)
**Target**: Predicción de diabetes

**Conceptos cubiertos**:
- Visualización de árboles
- Exportación vectorial
- Interpretabilidad de modelos

### 4. `feature_importance.py`
**Propósito**: Análisis de la importancia relativa de características.

**Características principales**:
- Cálculo de importancia basada en Gini
- Ranking de características más influyentes
- Identificación de variables relevantes
- Soporte para selección de características

**Dataset**: `cleaned_dataset.csv` (diabetes)
**Target**: Predicción de diabetes

**Conceptos cubiertos**:
- Feature Importance
- Gini Importance
- Selección de características
- Reducción de dimensionalidad

### 5. `overfitting_example.py`
**Propósito**: Análisis exhaustivo del fenómeno de overfitting.

**Características principales**:
- Evaluación de múltiples profundidades (2-14)
- Múltiples métricas: AUC, Accuracy, Precision, Recall, F1
- Comparación entrenamiento vs. prueba
- Identificación de profundidad óptima
- Visualización de curvas de validación

**Dataset**: `cleaned_dataset.csv` (diabetes)
**Target**: Predicción de diabetes

**Conceptos cubiertos**:
- Overfitting y underfitting
- Bias-Variance Tradeoff
- Curvas de validación
- Selección de hiperparámetros

## Requisitos

```python
# Librerías necesarias
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
numpy>=1.20.0
```

## Datasets Utilizados

### 1. Heart Disease Dataset
- **Archivo**: `heart_disease.csv`
- **Descripción**: Dataset de enfermedades cardíacas con variables categóricas y numéricas
- **Target**: Presencia/ausencia de enfermedad cardíaca
- **Uso**: Demostración de manejo de variables categóricas

### 2. Pima Indian Diabetes Dataset
- **Archivos**: `full_dataset.csv`, `cleaned_dataset.csv`
- **Descripción**: Dataset clásico de diabetes con variables numéricas
- **Target**: Presencia/ausencia de diabetes
- **Uso**: Análisis de overfitting, importancia de características, poda

## Instrucciones de Uso

### Ejecución Individual
```bash
# Ejecutar script específico
python categorical_example.py
python overfitting_example.py
# ... etc
```

### Orden Recomendado de Ejecución
1. **`decision_tree_visualization.py`** - Conceptos básicos
2. **`categorical_example.py`** - Variables categóricas
3. **`feature_importance.py`** - Importancia de características
4. **`overfitting_example.py`** - Análisis de overfitting
5. **`ccpa_pruning_example.py`** - Técnicas de optimización

## Salidas Generadas

### Archivos de Visualización
- `decision_tree.svg` - Árbol básico de diabetes
- `decision_tree_categorical.svg` - Árbol con variables categóricas

### Métricas Evaluadas
- **AUC (Area Under Curve)**: Métrica principal para clasificación binaria
- **Accuracy**: Porcentaje de predicciones correctas
- **Precision**: Precisión de predicciones positivas
- **Recall**: Sensibilidad del modelo
- **F1-Score**: Media armónica de precision y recall

## Conceptos de Machine Learning Cubiertos

### Fundamentos
- Algoritmo de árboles de decisión
- División de nodos basada en impureza
- Criterios de división (Gini, Entropy)

### Preprocesamiento
- Label Encoding para variables categóricas
- División entrenamiento/prueba
- Manejo de datasets reales

### Optimización
- Control de profundidad (`max_depth`)
- Poda por complejidad de costo (`ccp_alpha`)
- Prevención de overfitting

### Evaluación
- Múltiples métricas de clasificación
- Curvas de validación
- Análisis de bias-variance

### Interpretabilidad
- Visualización de árboles
- Importancia de características
- Análisis de reglas de decisión

## Interpretación de Resultados

### Detección de Overfitting
- **Señal de alarma**: Gran diferencia entre métricas de entrenamiento y prueba
- **Métrica objetivo**: AUC en conjunto de prueba
- **Profundidad óptima**: Punto de máximo AUC en prueba

### Selección de Características
- **Alta importancia**: Variables con mayor influencia en decisiones
- **Baja importancia**: Candidatas para eliminación
- **Suma total**: Siempre igual a 1.0

### Poda de Árboles
- **ccp_alpha bajo**: Árboles más complejos
- **ccp_alpha alto**: Árboles más simples
- **Valor óptimo**: Máximo AUC en prueba


