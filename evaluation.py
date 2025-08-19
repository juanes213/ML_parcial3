"""
Módulo de Evaluación de Modelos
Parcial 3 - Machine Learning
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, log_loss
)
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Clase para evaluación completa de modelos de ML"""
    
    def __init__(self, problem_type='classification'):
        """
        Inicializa el evaluador
        
        Args:
            problem_type (str): Tipo de problema ('classification', 'regression')
        """
        self.problem_type = problem_type
        self.evaluation_results = {}
        self.comparative_results = {}
    
    def evaluate_single_model(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
        """
        Evalúa un modelo individual
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones del modelo
            y_pred_proba: Probabilidades predichas (opcional)
            model_name (str): Nombre del modelo
        
        Returns:
            dict: Métricas de evaluación
        """
        if self.problem_type == 'classification':
            return self._evaluate_classification(y_true, y_pred, y_pred_proba, model_name)
        else:
            return self._evaluate_regression(y_true, y_pred, model_name)
    
    def _evaluate_classification(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
        """Evalúa modelo de clasificación"""
        print(f"Evaluando modelo de clasificación: {model_name}")
        
        # Métricas básicas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        
        # Reporte de clasificación
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': class_report
        }
        
        # Métricas adicionales si hay probabilidades
        if y_pred_proba is not None:
            try:
                if y_pred_proba.shape[1] == 2:  # Clasificación binaria
                    auc_roc = roc_auc_score(y_true, y_pred_proba[:, 1])
                    avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
                    logloss = log_loss(y_true, y_pred_proba)
                    
                    results.update({
                        'auc_roc': auc_roc,
                        'average_precision': avg_precision,
                        'log_loss': logloss
                    })
                else:  # Clasificación multiclase
                    auc_roc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
                    logloss = log_loss(y_true, y_pred_proba)
                    
                    results.update({
                        'auc_roc': auc_roc,
                        'log_loss': logloss
                    })
            except Exception as e:
                print(f"Error calculando métricas con probabilidades: {str(e)}")
        
        # Guardar resultados
        self.evaluation_results[model_name] = results
        
        # Imprimir resumen
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        if 'auc_roc' in results:
            print(f"  AUC-ROC: {results['auc_roc']:.4f}")
        
        return results
    
    def _evaluate_regression(self, y_true, y_pred, model_name="Model"):
        """Evalúa modelo de regresión"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        print(f"Evaluando modelo de regresión: {model_name}")
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        }
        
        self.evaluation_results[model_name] = results
        
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        return results
    
    def evaluate_multiple_models(self, y_true, predictions_dict, probabilities_dict=None):
        """
        Evalúa múltiples modelos
        
        Args:
            y_true: Etiquetas verdaderas
            predictions_dict (dict): Diccionario con predicciones por modelo
            probabilities_dict (dict): Diccionario con probabilidades por modelo
        
        Returns:
            pd.DataFrame: Comparación de métricas
        """
        print("Evaluando múltiples modelos...")
        print("="*50)
        
        for model_name, y_pred in predictions_dict.items():
            y_pred_proba = probabilities_dict.get(model_name, None) if probabilities_dict else None
            self.evaluate_single_model(y_true, y_pred, y_pred_proba, model_name)
            print()
        
        return self.create_comparison_table()
    
    def create_comparison_table(self):
        """Crea tabla comparativa de modelos"""
        if not self.evaluation_results:
            print("No hay resultados de evaluación disponibles")
            return None
        
        comparison_data = []
        
        for model_name, results in self.evaluation_results.items():
            row = {'Model': model_name}
            
            if self.problem_type == 'classification':
                row.update({
                    'Accuracy': results['accuracy'],
                    'Precision': results['precision'],
                    'Recall': results['recall'],
                    'F1-Score': results['f1_score']
                })
                
                if 'auc_roc' in results:
                    row['AUC-ROC'] = results['auc_roc']
                if 'log_loss' in results:
                    row['Log-Loss'] = results['log_loss']
            
            else:  # regression
                row.update({
                    'MSE': results['mse'],
                    'RMSE': results['rmse'],
                    'MAE': results['mae'],
                    'R²': results['r2_score']
                })
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Ordenar por métrica principal
        if self.problem_type == 'classification':
            comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        else:
            comparison_df = comparison_df.sort_values('R²', ascending=False)
        
        self.comparative_results = comparison_df
        
        print("COMPARACIÓN DE MODELOS")
        print("="*50)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        return comparison_df
    
    def detailed_classification_report(self, model_name):
        """Reporte detallado para un modelo de clasificación"""
        if model_name not in self.evaluation_results:
            print(f"Modelo {model_name} no encontrado en los resultados")
            return None
        
        results = self.evaluation_results[model_name]
        
        print(f"REPORTE DETALLADO: {model_name}")
        print("="*50)
        
        # Matriz de confusión
        print("MATRIZ DE CONFUSIÓN:")
        cm = results['confusion_matrix']
        cm_df = pd.DataFrame(cm, 
                           columns=[f'Pred_{i}' for i in range(cm.shape[1])],
                           index=[f'True_{i}' for i in range(cm.shape[0])])
        print(cm_df)
        
        print("\nREPORTE DE CLASIFICACIÓN:")
        class_report = results['classification_report']
        
        # Convertir a DataFrame para mejor visualización
        report_df = pd.DataFrame(class_report).transpose()
        print(report_df.round(4))
        
        return {
            'confusion_matrix': cm_df,
            'classification_report': report_df
        }
    
    def get_roc_curve_data(self, y_true, y_pred_proba, model_name):
        """Obtiene datos para curva ROC"""
        if y_pred_proba.shape[1] != 2:
            print("Curva ROC solo disponible para clasificación binaria")
            return None
        
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        return {
            'model_name': model_name,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': auc
        }
    
    def get_precision_recall_curve_data(self, y_true, y_pred_proba, model_name):
        """Obtiene datos para curva Precision-Recall"""
        if y_pred_proba.shape[1] != 2:
            print("Curva Precision-Recall solo disponible para clasificación binaria")
            return None
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba[:, 1])
        avg_precision = average_precision_score(y_true, y_pred_proba[:, 1])
        
        return {
            'model_name': model_name,
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'average_precision': avg_precision
        }
    
    def threshold_analysis(self, y_true, y_pred_proba, model_name, thresholds=None):
        """
        Analiza métricas en diferentes umbrales de decisión
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred_proba: Probabilidades predichas
            model_name (str): Nombre del modelo
            thresholds (list): Lista de umbrales a evaluar
        
        Returns:
            pd.DataFrame: Métricas por umbral
        """
        if y_pred_proba.shape[1] != 2:
            print("Análisis de umbral solo disponible para clasificación binaria")
            return None
        
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)
        
        results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba[:, 1] >= threshold).astype(int)
            
            accuracy = accuracy_score(y_true, y_pred_thresh)
            precision = precision_score(y_true, y_pred_thresh, zero_division=0)
            recall = recall_score(y_true, y_pred_thresh, zero_division=0)
            f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
        
        threshold_df = pd.DataFrame(results)
        
        print(f"ANÁLISIS DE UMBRALES: {model_name}")
        print("="*40)
        print(threshold_df.round(4))
        
        return threshold_df
    
    def cross_validation_analysis(self, model, X, y, cv_folds=5, scoring_metrics=None):
        """
        Análisis detallado de validación cruzada
        
        Args:
            model: Modelo entrenado
            X: Características
            y: Variable objetivo
            cv_folds (int): Número de folds
            scoring_metrics (list): Lista de métricas a evaluar
        
        Returns:
            dict: Resultados de CV detallados
        """
        from sklearn.model_selection import cross_validate
        
        if scoring_metrics is None:
            if self.problem_type == 'classification':
                scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            else:
                scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        cv_results = cross_validate(
            model, X, y, cv=cv_folds, 
            scoring=scoring_metrics, return_train_score=True
        )
        
        # Procesar resultados
        results_summary = {}
        
        for metric in scoring_metrics:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results_summary[metric] = {
                'test_mean': test_scores.mean(),
                'test_std': test_scores.std(),
                'train_mean': train_scores.mean(),
                'train_std': train_scores.std(),
                'test_scores': test_scores,
                'train_scores': train_scores
            }
        
        print("ANÁLISIS DE VALIDACIÓN CRUZADA")
        print("="*40)
        
        for metric, scores in results_summary.items():
            print(f"\n{metric.upper()}:")
            print(f"  Test:  {scores['test_mean']:.4f} (+/- {scores['test_std']*2:.4f})")
            print(f"  Train: {scores['train_mean']:.4f} (+/- {scores['train_std']*2:.4f})")
        
        return results_summary
    
    def model_interpretability_analysis(self, model, feature_names, X_sample=None, n_samples=100):
        """
        Análisis de interpretabilidad del modelo
        
        Args:
            model: Modelo entrenado
            feature_names (list): Nombres de características
            X_sample: Muestra de datos para análisis
            n_samples (int): Número de muestras para análisis
        
        Returns:
            dict: Resultados de interpretabilidad
        """
        results = {}
        
        # Importancia de características
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            results['feature_importance'] = importance_df
            
            print("IMPORTANCIA DE CARACTERÍSTICAS (Top 10):")
            print(importance_df.head(10).to_string(index=False, float_format='%.4f'))
        
        elif hasattr(model, 'coef_'):
            # Para modelos lineales
            coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
            
            coef_df = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coef,
                'abs_coefficient': np.abs(coef)
            }).sort_values('abs_coefficient', ascending=False)
            
            results['coefficients'] = coef_df
            
            print("COEFICIENTES DEL MODELO (Top 10):")
            print(coef_df.head(10).to_string(index=False, float_format='%.4f'))
        
        return results
    
    def export_results(self, filename=None):
        """Exporta resultados de evaluación"""
        if filename is None:
            filename = f"evaluation_results_{self.problem_type}.json"
        
        export_data = {
            'problem_type': self.problem_type,
            'evaluation_results': {},
            'comparative_results': self.comparative_results.to_dict() if hasattr(self.comparative_results, 'to_dict') else None
        }
        
        # Convertir resultados para serialización
        for model_name, results in self.evaluation_results.items():
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                elif isinstance(value, dict):
                    serializable_results[key] = value
                else:
                    serializable_results[key] = value
            
            export_data['evaluation_results'][model_name] = serializable_results
        
        import json
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Resultados exportados a {filename}")

def demo_evaluation():
    """Función de demostración de evaluación"""
    print("Demostrando evaluación de modelos...")
    
    # Importar modelos entrenados
    from models import demo_models
    
    trainer, (X_train, X_val, X_test, y_train, y_val, y_test), predictions = demo_models()
    
    # Crear evaluador
    evaluator = ModelEvaluator(problem_type='classification')
    
    # Obtener probabilidades
    probabilities = trainer.predict_proba_with_all_models(X_test)
    
    # Evaluar todos los modelos
    comparison_df = evaluator.evaluate_multiple_models(y_test, predictions, probabilities)
    
    # Análisis detallado del mejor modelo
    best_model_name = comparison_df.iloc[0]['Model']
    print(f"\n--- Análisis detallado del mejor modelo: {best_model_name} ---")
    
    evaluator.detailed_classification_report(best_model_name)
    
    # Análisis de umbral si hay probabilidades
    if best_model_name in probabilities:
        threshold_analysis = evaluator.threshold_analysis(
            y_test, probabilities[best_model_name], best_model_name
        )
    
    return evaluator, comparison_df

if __name__ == "__main__":
    demo_evaluation()