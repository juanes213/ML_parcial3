# Guía de Uso - ML Parcial 3

## Descripción General

Este proyecto implementa una solución completa para el Parcial 3 de Machine Learning, demostrando competencias en análisis de datos, preprocesamiento, modelado, evaluación y visualización.

## Estructura de Archivos

```
ML_parcial3/
├── README.md                    # Documentación principal
├── requirements.txt             # Dependencias de Python
├── .gitignore                  # Archivos a ignorar en Git
├── USAGE.md                    # Esta guía de uso
├── parcial3_analysis.ipynb     # Notebook Jupyter interactivo
├── main.py                     # Script principal - ejecuta pipeline completo
├── data_analysis.py            # Módulo de análisis exploratorio
├── preprocessing.py            # Módulo de preprocesamiento
├── models.py                   # Módulo de entrenamiento de modelos
├── evaluation.py               # Módulo de evaluación
├── visualizations.py           # Módulo de visualizaciones
└── results/                    # Carpeta de resultados (generada automáticamente)
    └── run_YYYYMMDD_HHMMSS/   # Resultados por ejecución
        ├── *.png              # Gráficos generados
        └── reporte_ml_parcial3.txt  # Reporte final
```

## Formas de Usar el Proyecto

### 1. Ejecución Completa (Recomendado)

Ejecuta el pipeline completo con datos de muestra:

```bash
python main.py
```

Esto generará:
- Análisis exploratorio completo
- Preprocesamiento de datos
- Entrenamiento de 7 modelos diferentes
- Evaluación comprehensiva
- Visualizaciones
- Reporte final

### 2. Opciones de Línea de Comandos

```bash
# Ejecutar sin visualizaciones (más rápido)
python main.py --no-visualize

# Optimizar hiperparámetros (más lento pero mejores resultados)
python main.py --optimize

# Usar dataset personalizado
python main.py --dataset mi_dataset.csv --target columna_objetivo

# Combinaciones
python main.py --dataset datos.csv --target resultado --optimize --no-visualize
```

### 3. Uso de Módulos Individuales

#### Análisis de Datos
```python
from data_analysis import DataAnalyzer
import pandas as pd

# Cargar datos
data = pd.read_csv('mi_dataset.csv')

# Crear analizador
analyzer = DataAnalyzer(data)

# Ejecutar análisis completo
results = analyzer.run_complete_analysis()
```

#### Preprocesamiento
```python
from preprocessing import DataPreprocessor

# Crear preprocesador
preprocessor = DataPreprocessor(data, target_column='mi_objetivo')

# Aplicar preprocesamiento paso a paso
preprocessor.handle_missing_values() \
           .remove_outliers() \
           .encode_categorical_variables() \
           .scale_features()

# Dividir datos
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()
```

#### Entrenamiento de Modelos
```python
from models import MLModelTrainer

# Crear entrenador
trainer = MLModelTrainer(problem_type='classification')

# Entrenar modelos específicos
trainer.train_single_model('Random Forest', X_train, y_train)
trainer.train_single_model('SVM', X_train, y_train, optimize_hyperparams=True)

# O entrenar todos los modelos
trainer.train_all_models(X_train, y_train)
```

#### Evaluación
```python
from evaluation import ModelEvaluator

# Generar predicciones
predictions = trainer.predict_with_all_models(X_test)
probabilities = trainer.predict_proba_with_all_models(X_test)

# Evaluar modelos
evaluator = ModelEvaluator(problem_type='classification')
comparison_df = evaluator.evaluate_multiple_models(y_test, predictions, probabilities)
```

#### Visualizaciones
```python
from visualizations import MLVisualizer

# Crear visualizador
visualizer = MLVisualizer(save_plots=True, output_dir='mis_graficos')

# Generar visualizaciones específicas
visualizer.plot_data_distribution(data)
visualizer.plot_model_comparison(comparison_df)
visualizer.plot_roc_curves(y_test, probabilities)
```

### 4. Análisis Interactivo con Jupyter

Abre el notebook para análisis interactivo:

```bash
jupyter notebook parcial3_analysis.ipynb
```

El notebook incluye:
- Celdas paso a paso para cada etapa del análisis
- Visualizaciones interactivas
- Explicaciones detalladas
- Posibilidad de experimentar con parámetros

## Configuración Personalizada

Para personalizar el comportamiento del pipeline, modifica la configuración en `main.py`:

```python
config = {
    'preprocessing': {
        'handle_missing': True,
        'missing_strategy': 'mean',  # 'mean', 'median', 'constant'
        'remove_outliers': True,
        'outlier_method': 'iqr',     # 'iqr', 'zscore'
        'outlier_threshold': 2.0,
        'encode_categorical': True,
        'encoding_method': 'onehot',  # 'onehot', 'label'
        'scale_features': True,
        'scaling_method': 'standard', # 'standard', 'minmax', 'robust'
        'create_interactions': True,
        'max_interactions': 5
    },
    'modeling': {
        'optimize_hyperparams': False,
        'cv_folds': 5,
        'test_size': 0.2,
        'validation_size': 0.1,
        'random_state': 42
    },
    'evaluation': {
        'detailed_analysis': True,
        'threshold_analysis': True,
        'learning_curves': True
    },
    'visualization': {
        'create_plots': True,
        'save_plots': True
    }
}
```

## Tipos de Problemas Soportados

### Clasificación (Por Defecto)
- Clasificación binaria
- Clasificación multiclase
- Métricas: accuracy, precision, recall, F1-score, AUC-ROC
- Visualizaciones: matrices de confusión, curvas ROC, curvas precision-recall

### Regresión (Experimental)
- Métricas: MSE, RMSE, MAE, R²
- Modelos: Linear Regression, Random Forest Regressor, etc.

## Modelos Incluidos

### Clasificación
1. **Logistic Regression** - Modelo lineal, rápido y interpretable
2. **Random Forest** - Ensemble de árboles, robusto
3. **Gradient Boosting** - Boosting secuencial, alta precisión
4. **SVM** - Support Vector Machine, efectivo en espacios de alta dimensión
5. **K-Nearest Neighbors** - Clasificación basada en proximidad
6. **Naive Bayes** - Probabilístico, funciona bien con pocas muestras
7. **Decision Tree** - Interpretable, propenso a sobreajuste

## Métricas de Evaluación

### Clasificación
- **Accuracy**: Proporción de predicciones correctas
- **Precision**: Proporción de verdaderos positivos entre predicciones positivas
- **Recall**: Proporción de verdaderos positivos entre casos positivos reales
- **F1-Score**: Media armónica entre precision y recall
- **AUC-ROC**: Área bajo la curva ROC
- **Log-Loss**: Pérdida logarítmica

### Visualizaciones Generadas
- Distribución de variables
- Matriz de correlación
- Comparación de modelos
- Matrices de confusión
- Curvas ROC
- Curvas Precision-Recall
- Importancia de características
- Curvas de aprendizaje (opcional)

## Solución de Problemas

### Error de Dependencias
```bash
pip install -r requirements.txt
```

### Error de Memoria en Datasets Grandes
- Reduce el tamaño de muestra
- Deshabilita la optimización de hiperparámetros
- Usa `--no-visualize` para ahorrar memoria

### Error en Visualizaciones
- Verifica que matplotlib y seaborn estén instalados
- Usa `--no-visualize` si no necesitas gráficos

### Dataset Personalizado no Reconocido
- Verifica que el archivo exista
- Asegúrate de especificar la columna objetivo correcta
- Usa formatos CSV o Excel (.xlsx)

## Interpretación de Resultados

### Tabla de Comparación de Modelos
- **Accuracy**: Métrica principal para clasificación balanceada
- **F1-Score**: Mejor para datasets desbalanceados
- **AUC-ROC**: Independiente del umbral de decisión

### Matrices de Confusión
- **Diagonal principal**: Predicciones correctas
- **Fuera de diagonal**: Errores de clasificación

### Curvas ROC
- **Línea diagonal**: Clasificador aleatorio
- **Área bajo la curva**: Mayor es mejor

### Importancia de Características
- Identifica variables más relevantes para el modelo
- Útil para selección de características y interpretabilidad

## Extensiones Futuras

Este proyecto puede extenderse con:
- Más algoritmos de ML (XGBoost, LightGBM, Neural Networks)
- Técnicas de feature selection automática
- Validación cruzada estratificada
- Detección automática de outliers
- Pipeline de AutoML
- Despliegue de modelos en producción

## Contacto y Soporte

Para preguntas sobre el proyecto:
1. Revisa la documentación en README.md
2. Consulta los comentarios en el código
3. Experimenta con el notebook interactivo
4. Modifica parámetros en main.py para diferentes configuraciones