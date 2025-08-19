# ML Parcial 3 - Machine Learning Project

## Descripción del Proyecto

Este proyecto presenta una solución completa para el Parcial 3 de Machine Learning, incluyendo análisis de datos, implementación de modelos, evaluación y visualizaciones.

## Objetivos

- Demostrar el preprocesamiento y análisis exploratorio de datos
- Implementar y comparar diferentes algoritmos de machine learning
- Evaluar modelos usando métricas apropiadas
- Generar visualizaciones informativas
- Proporcionar conclusiones y recomendaciones basadas en los resultados

## Estructura del Proyecto

```
ML_parcial3/
├── README.md                 # Documentación del proyecto
├── requirements.txt          # Dependencias de Python
├── data_analysis.py          # Análisis exploratorio de datos
├── preprocessing.py          # Preprocesamiento de datos
├── models.py                # Implementación de modelos ML
├── evaluation.py            # Evaluación y métricas
├── visualizations.py        # Gráficos y visualizaciones
├── main.py                  # Script principal
└── results/                 # Carpeta para resultados y gráficos
```

## Instalación y Configuración

1. Clona el repositorio:
```bash
git clone https://github.com/juanes213/ML_parcial3.git
cd ML_parcial3
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecuta el proyecto completo:
```bash
python main.py
```

## Metodología

### 1. Análisis Exploratorio de Datos
- Carga y exploración inicial del dataset
- Análisis estadístico descriptivo
- Identificación de valores faltantes y outliers
- Análisis de correlaciones

### 2. Preprocesamiento
- Limpieza de datos
- Manejo de valores faltantes
- Codificación de variables categóricas
- Normalización y escalado
- División en conjuntos de entrenamiento y prueba

### 3. Modelado
- Implementación de múltiples algoritmos:
  - Regresión Logística
  - Random Forest
  - Support Vector Machine
  - Gradient Boosting
- Optimización de hiperparámetros
- Validación cruzada

### 4. Evaluación
- Métricas de clasificación (Accuracy, Precision, Recall, F1-Score)
- Matriz de confusión
- Curvas ROC y AUC
- Análisis de importancia de features

### 5. Visualizaciones
- Distribuciones de variables
- Gráficos de correlación
- Comparación de modelos
- Curvas de aprendizaje

## Resultados

Los resultados detallados, incluyendo métricas de evaluación y visualizaciones, se generan automáticamente en la carpeta `results/` al ejecutar el proyecto.

## Conclusiones

[Las conclusiones se generarán automáticamente basadas en los resultados del análisis]

## Tecnologías Utilizadas

- Python 3.8+
- Pandas para manipulación de datos
- NumPy para computación numérica
- Scikit-learn para machine learning
- Matplotlib y Seaborn para visualizaciones
- Jupyter para análisis interactivo

## Autor

Desarrollado para el Parcial 3 de Machine Learning

## Licencia

Este proyecto es para fines académicos.