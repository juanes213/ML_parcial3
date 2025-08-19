"""
Módulo de Visualizaciones
Parcial 3 - Machine Learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('default')
sns.set_palette("husl")

class MLVisualizer:
    """Clase para crear visualizaciones de ML"""
    
    def __init__(self, figsize=(12, 8), save_plots=True, output_dir='results'):
        """
        Inicializa el visualizador
        
        Args:
            figsize (tuple): Tamaño por defecto de las figuras
            save_plots (bool): Si guardar gráficos
            output_dir (str): Directorio para guardar gráficos
        """
        self.figsize = figsize
        self.save_plots = save_plots
        self.output_dir = output_dir
        
        # Crear directorio de resultados si no existe
        import os
        if self.save_plots and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def _save_plot(self, filename):
        """Guarda el gráfico actual"""
        if self.save_plots:
            filepath = f"{self.output_dir}/{filename}"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado: {filepath}")
    
    def plot_data_distribution(self, data, columns=None, title="Distribución de Variables"):
        """
        Visualiza distribución de variables numéricas
        
        Args:
            data (pd.DataFrame): Dataset
            columns (list): Columnas a visualizar
            title (str): Título del gráfico
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = min(4, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                axes[i].hist(data[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribución de {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frecuencia')
                axes[i].grid(True, alpha=0.3)
        
        # Ocultar subplots vacíos
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_plot("data_distribution.png")
        plt.show()
    
    def plot_correlation_matrix(self, data, title="Matriz de Correlación"):
        """
        Visualiza matriz de correlación
        
        Args:
            data (pd.DataFrame): Dataset
            title (str): Título del gráfico
        """
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            print("Insuficientes variables numéricas para matriz de correlación")
            return
        
        correlation_matrix = numeric_data.corr()
        
        plt.figure(figsize=self.figsize)
        
        # Crear máscara para triángulo superior
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        # Crear heatmap
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_plot("correlation_matrix.png")
        plt.show()
    
    def plot_categorical_distribution(self, data, columns=None, title="Distribución de Variables Categóricas"):
        """
        Visualiza distribución de variables categóricas
        
        Args:
            data (pd.DataFrame): Dataset
            columns (list): Columnas categóricas a visualizar
            title (str): Título del gráfico
        """
        if columns is None:
            columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not columns:
            print("No hay variables categóricas para visualizar")
            return
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                value_counts = data[col].value_counts()
                
                # Solo mostrar top 10 categorías si hay muchas
                if len(value_counts) > 10:
                    value_counts = value_counts.head(10)
                
                value_counts.plot(kind='bar', ax=axes[i], color='skyblue', edgecolor='black')
                axes[i].set_title(f'Distribución de {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frecuencia')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        # Ocultar subplots vacíos
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_plot("categorical_distribution.png")
        plt.show()
    
    def plot_target_distribution(self, data, target_column, title="Distribución de Variable Objetivo"):
        """
        Visualiza distribución de la variable objetivo
        
        Args:
            data (pd.DataFrame): Dataset
            target_column (str): Nombre de la columna objetivo
            title (str): Título del gráfico
        """
        plt.figure(figsize=(10, 6))
        
        if data[target_column].dtype in ['object', 'category'] or data[target_column].nunique() <= 10:
            # Variable categórica o discreta
            value_counts = data[target_column].value_counts()
            
            plt.subplot(1, 2, 1)
            value_counts.plot(kind='bar', color='lightcoral', edgecolor='black')
            plt.title('Distribución por Conteo')
            plt.xlabel(target_column)
            plt.ylabel('Frecuencia')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            plt.title('Distribución Porcentual')
        
        else:
            # Variable continua
            plt.subplot(1, 2, 1)
            plt.hist(data[target_column].dropna(), bins=30, color='lightgreen', 
                    edgecolor='black', alpha=0.7)
            plt.title('Histograma')
            plt.xlabel(target_column)
            plt.ylabel('Frecuencia')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            plt.boxplot(data[target_column].dropna())
            plt.title('Box Plot')
            plt.ylabel(target_column)
            plt.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_plot("target_distribution.png")
        plt.show()
    
    def plot_model_comparison(self, comparison_df, title="Comparación de Modelos"):
        """
        Visualiza comparación de métricas entre modelos
        
        Args:
            comparison_df (pd.DataFrame): DataFrame con métricas por modelo
            title (str): Título del gráfico
        """
        # Excluir columna 'Model' para gráficos
        numeric_columns = comparison_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) == 0:
            print("No hay métricas numéricas para visualizar")
            return
        
        n_metrics = len(numeric_columns)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(numeric_columns):
            if i < len(axes):
                # Gráfico de barras
                bars = axes[i].bar(comparison_df['Model'], comparison_df[metric], 
                                 color='lightblue', edgecolor='black')
                
                # Añadir valores sobre las barras
                for bar in bars:
                    height = bar.get_height()
                    axes[i].annotate(f'{height:.3f}',
                                   xy=(bar.get_x() + bar.get_width() / 2, height),
                                   xytext=(0, 3),  # 3 points vertical offset
                                   textcoords="offset points",
                                   ha='center', va='bottom')
                
                axes[i].set_title(f'{metric}')
                axes[i].set_xlabel('Modelo')
                axes[i].set_ylabel(metric)
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].grid(True, alpha=0.3)
        
        # Ocultar subplots vacíos
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_plot("model_comparison.png")
        plt.show()
    
    def plot_confusion_matrices(self, y_true, predictions_dict, title="Matrices de Confusión"):
        """
        Visualiza matrices de confusión para múltiples modelos
        
        Args:
            y_true: Etiquetas verdaderas
            predictions_dict (dict): Predicciones por modelo
            title (str): Título del gráfico
        """
        n_models = len(predictions_dict)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (model_name, y_pred) in enumerate(predictions_dict.items()):
            if i < len(axes):
                cm = confusion_matrix(y_true, y_pred)
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(f'{model_name}')
                axes[i].set_xlabel('Predicho')
                axes[i].set_ylabel('Real')
        
        # Ocultar subplots vacíos
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_plot("confusion_matrices.png")
        plt.show()
    
    def plot_roc_curves(self, y_true, probabilities_dict, title="Curvas ROC"):
        """
        Visualiza curvas ROC para múltiples modelos
        
        Args:
            y_true: Etiquetas verdaderas
            probabilities_dict (dict): Probabilidades por modelo
            title (str): Título del gráfico
        """
        plt.figure(figsize=self.figsize)
        
        for model_name, y_pred_proba in probabilities_dict.items():
            if y_pred_proba.shape[1] == 2:  # Solo clasificación binaria
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                auc = np.trapz(tpr, fpr)
                
                plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        self._save_plot("roc_curves.png")
        plt.show()
    
    def plot_precision_recall_curves(self, y_true, probabilities_dict, title="Curvas Precision-Recall"):
        """
        Visualiza curvas Precision-Recall para múltiples modelos
        
        Args:
            y_true: Etiquetas verdaderas
            probabilities_dict (dict): Probabilidades por modelo
            title (str): Título del gráfico
        """
        plt.figure(figsize=self.figsize)
        
        for model_name, y_pred_proba in probabilities_dict.items():
            if y_pred_proba.shape[1] == 2:  # Solo clasificación binaria
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
                avg_precision = np.trapz(precision, recall)
                
                plt.plot(recall, precision, linewidth=2, 
                        label=f'{model_name} (AP = {avg_precision:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        self._save_plot("precision_recall_curves.png")
        plt.show()
    
    def plot_feature_importance(self, importance_dict, top_n=10, title="Importancia de Características"):
        """
        Visualiza importancia de características para múltiples modelos
        
        Args:
            importance_dict (dict): Importancia por modelo
            top_n (int): Número de características top a mostrar
            title (str): Título del gráfico
        """
        n_models = len(importance_dict)
        n_cols = min(2, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*8, n_rows*6))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (model_name, importance_df) in enumerate(importance_dict.items()):
            if i < len(axes):
                # Tomar top N características
                top_features = importance_df.head(top_n)
                
                # Gráfico horizontal de barras
                y_pos = np.arange(len(top_features))
                axes[i].barh(y_pos, top_features['importance'], color='lightgreen', edgecolor='black')
                axes[i].set_yticks(y_pos)
                axes[i].set_yticklabels(top_features['feature'])
                axes[i].invert_yaxis()
                axes[i].set_xlabel('Importancia')
                axes[i].set_title(f'{model_name}')
                axes[i].grid(True, alpha=0.3)
        
        # Ocultar subplots vacíos
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save_plot("feature_importance.png")
        plt.show()
    
    def plot_learning_curves(self, learning_curves_data, title="Curvas de Aprendizaje"):
        """
        Visualiza curvas de aprendizaje
        
        Args:
            learning_curves_data (dict): Datos de curvas de aprendizaje
            title (str): Título del gráfico
        """
        plt.figure(figsize=self.figsize)
        
        train_sizes = learning_curves_data['train_sizes']
        train_scores_mean = learning_curves_data['train_scores_mean']
        train_scores_std = learning_curves_data['train_scores_std']
        val_scores_mean = learning_curves_data['val_scores_mean']
        val_scores_std = learning_curves_data['val_scores_std']
        
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, alpha=0.1, color="g")
        
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, val_scores_mean, 'o-', color="g", label="Cross-validation score")
        
        plt.xlabel('Tamaño del conjunto de entrenamiento')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        self._save_plot("learning_curves.png")
        plt.show()
    
    def plot_validation_curve(self, validation_curve_data, param_name, title="Curva de Validación"):
        """
        Visualiza curva de validación
        
        Args:
            validation_curve_data (dict): Datos de curva de validación
            param_name (str): Nombre del parámetro
            title (str): Título del gráfico
        """
        plt.figure(figsize=self.figsize)
        
        param_range = validation_curve_data['param_range']
        train_scores_mean = validation_curve_data['train_scores_mean']
        train_scores_std = validation_curve_data['train_scores_std']
        val_scores_mean = validation_curve_data['val_scores_mean']
        val_scores_std = validation_curve_data['val_scores_std']
        
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(param_range, val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, alpha=0.1, color="g")
        
        plt.plot(param_range, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(param_range, val_scores_mean, 'o-', color="g", label="Validation score")
        
        plt.xlabel(param_name)
        plt.ylabel('Score')
        plt.title(f'{title} - {param_name}')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        self._save_plot(f"validation_curve_{param_name}.png")
        plt.show()
    
    def create_comprehensive_report(self, data, evaluator, trainer, target_column):
        """
        Crea un reporte visual completo
        
        Args:
            data (pd.DataFrame): Dataset original
            evaluator: Objeto evaluador con resultados
            trainer: Objeto entrenador con modelos
            target_column (str): Columna objetivo
        """
        print("Generando reporte visual completo...")
        print("="*50)
        
        # 1. Análisis exploratorio
        print("1. Generando visualizaciones de análisis exploratorio...")
        self.plot_data_distribution(data, title="Distribución de Variables Numéricas")
        self.plot_correlation_matrix(data, title="Matriz de Correlación")
        self.plot_categorical_distribution(data, title="Distribución de Variables Categóricas")
        self.plot_target_distribution(data, target_column, title="Distribución de Variable Objetivo")
        
        # 2. Comparación de modelos
        print("2. Generando visualizaciones de comparación de modelos...")
        if hasattr(evaluator, 'comparative_results') and evaluator.comparative_results is not None:
            self.plot_model_comparison(evaluator.comparative_results, title="Comparación de Modelos")
        
        print("Reporte visual completo generado exitosamente!")

def demo_visualizations():
    """Función de demostración de visualizaciones"""
    print("Demostrando visualizaciones ML...")
    
    # Importar evaluación completa
    from evaluation import demo_evaluation
    
    evaluator, comparison_df = demo_evaluation()
    
    # Crear visualizador
    visualizer = MLVisualizer(save_plots=True, output_dir='results')
    
    # Importar datos adicionales para visualizaciones completas
    from data_analysis import DataAnalyzer
    from models import demo_models
    
    analyzer = DataAnalyzer(pd.DataFrame())
    sample_data = analyzer.generate_sample_data(1000)
    
    trainer, (X_train, X_val, X_test, y_train, y_val, y_test), predictions = demo_models()
    probabilities = trainer.predict_proba_with_all_models(X_test)
    
    # Generar visualizaciones
    print("\n--- Generando visualizaciones ---")
    
    # Distribuciones de datos
    visualizer.plot_data_distribution(sample_data)
    visualizer.plot_correlation_matrix(sample_data)
    visualizer.plot_target_distribution(sample_data, 'purchase')
    
    # Comparación de modelos
    visualizer.plot_model_comparison(comparison_df)
    
    # Matrices de confusión
    visualizer.plot_confusion_matrices(y_test, predictions)
    
    # Curvas ROC y Precision-Recall (solo para clasificación binaria)
    if any(prob.shape[1] == 2 for prob in probabilities.values()):
        visualizer.plot_roc_curves(y_test, probabilities)
        visualizer.plot_precision_recall_curves(y_test, probabilities)
    
    # Importancia de características
    feature_names = X_train.columns.tolist()
    importance = trainer.get_feature_importance(feature_names)
    if importance:
        visualizer.plot_feature_importance(importance)
    
    print("Todas las visualizaciones han sido generadas exitosamente!")
    
    return visualizer

if __name__ == "__main__":
    demo_visualizations()