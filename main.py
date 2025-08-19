"""
Script Principal - Parcial 3 Machine Learning
Ejecuta el pipeline completo de ML

Uso:
    python main.py [--dataset <path>] [--target <column>] [--optimize] [--visualize]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar módulos del proyecto
from data_analysis import DataAnalyzer
from preprocessing import DataPreprocessor
from models import MLModelTrainer
from evaluation import ModelEvaluator
from visualizations import MLVisualizer

class MLPipeline:
    """Pipeline completo de Machine Learning"""
    
    def __init__(self, config=None):
        """
        Inicializa el pipeline
        
        Args:
            config (dict): Configuración del pipeline
        """
        self.config = config or self._default_config()
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Crear directorio de resultados
        self.results_dir = f"results/run_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"Pipeline ML iniciado - Resultados en: {self.results_dir}")
    
    def _default_config(self):
        """Configuración por defecto del pipeline"""
        return {
            'preprocessing': {
                'handle_missing': True,
                'missing_strategy': 'mean',
                'remove_outliers': True,
                'outlier_method': 'iqr',
                'outlier_threshold': 2.0,
                'encode_categorical': True,
                'encoding_method': 'onehot',
                'scale_features': True,
                'scaling_method': 'standard',
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
    
    def load_data(self, dataset_path=None, target_column=None):
        """
        Carga los datos para el pipeline
        
        Args:
            dataset_path (str): Ruta al dataset
            target_column (str): Nombre de la columna objetivo
        """
        print("="*60)
        print("PASO 1: CARGA Y ANÁLISIS DE DATOS")
        print("="*60)
        
        if dataset_path and os.path.exists(dataset_path):
            # Cargar dataset externo
            print(f"Cargando dataset: {dataset_path}")
            
            if dataset_path.endswith('.csv'):
                self.data = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.xlsx'):
                self.data = pd.read_excel(dataset_path)
            else:
                raise ValueError("Formato de archivo no soportado. Use CSV o Excel.")
            
            self.target_column = target_column
            
        else:
            # Generar datos de muestra
            print("Generando dataset de muestra para demostración...")
            
            analyzer = DataAnalyzer(pd.DataFrame())
            self.data = analyzer.generate_sample_data(1000)
            self.target_column = 'purchase'
            
            print("Dataset de muestra generado:")
            print(f"  - Muestras: {len(self.data)}")
            print(f"  - Variables: {len(self.data.columns)}")
            print(f"  - Variable objetivo: {self.target_column}")
        
        # Análisis exploratorio
        print(f"\nDataset cargado: {self.data.shape}")
        print(f"Columna objetivo: {self.target_column}")
        
        self.analyzer = DataAnalyzer(self.data)
        self.analysis_results = self.analyzer.run_complete_analysis()
        
        self.results['data_analysis'] = self.analysis_results
        
        return self
    
    def preprocess_data(self):
        """Preprocesa los datos"""
        print("\n" + "="*60)
        print("PASO 2: PREPROCESAMIENTO DE DATOS")
        print("="*60)
        
        self.preprocessor = DataPreprocessor(self.data, self.target_column)
        
        config = self.config['preprocessing']
        
        # Aplicar pasos de preprocesamiento según configuración
        if config['handle_missing']:
            self.preprocessor.handle_missing_values(
                strategy=config['missing_strategy']
            )
        
        if config['remove_outliers']:
            self.preprocessor.remove_outliers(
                method=config['outlier_method'],
                threshold=config['outlier_threshold']
            )
        
        if config['encode_categorical']:
            self.preprocessor.encode_categorical_variables(
                method=config['encoding_method']
            )
        
        if config['create_interactions']:
            self.preprocessor.create_feature_interactions(
                max_interactions=config['max_interactions']
            )
        
        if config['scale_features']:
            self.preprocessor.scale_features(
                method=config['scaling_method']
            )
        
        # Dividir datos
        model_config = self.config['modeling']
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
            self.preprocessor.split_data(
                test_size=model_config['test_size'],
                validation_size=model_config['validation_size'],
                random_state=model_config['random_state']
            )
        
        # Guardar resumen de preprocesamiento
        preprocessing_summary = self.preprocessor.get_preprocessing_summary()
        self.results['preprocessing'] = preprocessing_summary
        
        print(f"\nPreprocesamiento completado:")
        print(f"  - Entrenamiento: {self.X_train.shape}")
        if self.X_val is not None:
            print(f"  - Validación: {self.X_val.shape}")
        print(f"  - Prueba: {self.X_test.shape}")
        
        return self
    
    def train_models(self):
        """Entrena los modelos de ML"""
        print("\n" + "="*60)
        print("PASO 3: ENTRENAMIENTO DE MODELOS")
        print("="*60)
        
        # Determinar tipo de problema
        if self.data[self.target_column].dtype in ['object', 'category'] or \
           self.data[self.target_column].nunique() <= 10:
            problem_type = 'classification'
        else:
            problem_type = 'regression'
        
        self.trainer = MLModelTrainer(problem_type=problem_type)
        
        config = self.config['modeling']
        
        # Entrenar todos los modelos
        self.trainer.train_all_models(
            self.X_train, self.y_train,
            optimize_hyperparams=config['optimize_hyperparams'],
            cv_folds=config['cv_folds']
        )
        
        # Generar predicciones
        self.predictions = self.trainer.predict_with_all_models(self.X_test)
        self.probabilities = self.trainer.predict_proba_with_all_models(self.X_test)
        
        # Importancia de características
        feature_names = self.X_train.columns.tolist()
        self.feature_importance = self.trainer.get_feature_importance(feature_names)
        
        self.results['training'] = {
            'problem_type': problem_type,
            'models_trained': list(self.trainer.trained_models.keys()),
            'training_results': self.trainer.training_results,
            'feature_importance': self.feature_importance
        }
        
        return self
    
    def evaluate_models(self):
        """Evalúa los modelos entrenados"""
        print("\n" + "="*60)
        print("PASO 4: EVALUACIÓN DE MODELOS")
        print("="*60)
        
        problem_type = self.results['training']['problem_type']
        self.evaluator = ModelEvaluator(problem_type=problem_type)
        
        # Evaluar todos los modelos
        self.comparison_df = self.evaluator.evaluate_multiple_models(
            self.y_test, self.predictions, self.probabilities
        )
        
        config = self.config['evaluation']
        
        # Análisis detallado del mejor modelo
        best_model_name = self.comparison_df.iloc[0]['Model']
        print(f"\n--- ANÁLISIS DETALLADO: {best_model_name} ---")
        
        if config['detailed_analysis']:
            detailed_report = self.evaluator.detailed_classification_report(best_model_name)
        
        # Análisis de umbrales para clasificación binaria
        if config['threshold_analysis'] and problem_type == 'classification':
            if best_model_name in self.probabilities:
                proba = self.probabilities[best_model_name]
                if proba.shape[1] == 2:
                    threshold_analysis = self.evaluator.threshold_analysis(
                        self.y_test, proba, best_model_name
                    )
        
        # Curvas de aprendizaje
        if config['learning_curves']:
            print(f"\n--- CURVAS DE APRENDIZAJE: {best_model_name} ---")
            best_model = self.trainer.trained_models[best_model_name]
            
            learning_curves = self.trainer.learning_curves_analysis(
                self.X_train, self.y_train, best_model_name
            )
        
        self.results['evaluation'] = {
            'comparison_results': self.comparison_df.to_dict(),
            'best_model': best_model_name,
            'detailed_results': self.evaluator.evaluation_results
        }
        
        return self
    
    def create_visualizations(self):
        """Crea visualizaciones"""
        print("\n" + "="*60)
        print("PASO 5: GENERACIÓN DE VISUALIZACIONES")
        print("="*60)
        
        if not self.config['visualization']['create_plots']:
            print("Visualizaciones deshabilitadas en configuración")
            return self
        
        self.visualizer = MLVisualizer(
            save_plots=self.config['visualization']['save_plots'],
            output_dir=self.results_dir
        )
        
        # Análisis exploratorio
        print("Generando visualizaciones de análisis exploratorio...")
        self.visualizer.plot_data_distribution(
            self.data, title="Distribución de Variables Numéricas"
        )
        self.visualizer.plot_correlation_matrix(
            self.data, title="Matriz de Correlación"
        )
        self.visualizer.plot_target_distribution(
            self.data, self.target_column, title="Distribución de Variable Objetivo"
        )
        
        # Comparación de modelos
        print("Generando visualizaciones de modelos...")
        self.visualizer.plot_model_comparison(
            self.comparison_df, title="Comparación de Modelos"
        )
        
        # Matrices de confusión
        self.visualizer.plot_confusion_matrices(
            self.y_test, self.predictions, title="Matrices de Confusión"
        )
        
        # Curvas ROC y Precision-Recall (clasificación binaria)
        problem_type = self.results['training']['problem_type']
        if problem_type == 'classification' and self.probabilities:
            binary_probs = {k: v for k, v in self.probabilities.items() if v.shape[1] == 2}
            if binary_probs:
                self.visualizer.plot_roc_curves(
                    self.y_test, binary_probs, title="Curvas ROC"
                )
                self.visualizer.plot_precision_recall_curves(
                    self.y_test, binary_probs, title="Curvas Precision-Recall"
                )
        
        # Importancia de características
        if self.feature_importance:
            self.visualizer.plot_feature_importance(
                self.feature_importance, title="Importancia de Características"
            )
        
        print("Visualizaciones generadas exitosamente!")
        
        return self
    
    def generate_report(self):
        """Genera reporte final"""
        print("\n" + "="*60)
        print("PASO 6: GENERACIÓN DE REPORTE FINAL")
        print("="*60)
        
        report_file = f"{self.results_dir}/reporte_ml_parcial3.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("REPORTE - PARCIAL 3 MACHINE LEARNING\n")
            f.write("="*50 + "\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Resumen de datos
            f.write("1. RESUMEN DE DATOS\n")
            f.write("-"*20 + "\n")
            f.write(f"Shape del dataset: {self.data.shape}\n")
            f.write(f"Variable objetivo: {self.target_column}\n")
            f.write(f"Tipo de problema: {self.results['training']['problem_type']}\n\n")
            
            # Preprocesamiento
            f.write("2. PREPROCESAMIENTO\n")
            f.write("-"*20 + "\n")
            for step in self.results['preprocessing']['preprocessing_steps']:
                f.write(f"✓ {step}\n")
            f.write("\n")
            
            # Modelos entrenados
            f.write("3. MODELOS ENTRENADOS\n")
            f.write("-"*20 + "\n")
            for model in self.results['training']['models_trained']:
                f.write(f"• {model}\n")
            f.write("\n")
            
            # Resultados de evaluación
            f.write("4. RESULTADOS DE EVALUACIÓN\n")
            f.write("-"*20 + "\n")
            f.write(self.comparison_df.to_string(index=False, float_format='%.4f'))
            f.write("\n\n")
            
            # Mejor modelo
            best_model = self.results['evaluation']['best_model']
            f.write(f"5. MEJOR MODELO: {best_model}\n")
            f.write("-"*20 + "\n")
            
            best_results = self.evaluator.evaluation_results[best_model]
            if 'accuracy' in best_results:
                f.write(f"Accuracy: {best_results['accuracy']:.4f}\n")
                f.write(f"Precision: {best_results['precision']:.4f}\n")
                f.write(f"Recall: {best_results['recall']:.4f}\n")
                f.write(f"F1-Score: {best_results['f1_score']:.4f}\n")
            
            # Conclusiones
            f.write("\n6. CONCLUSIONES\n")
            f.write("-"*20 + "\n")
            f.write(f"• El mejor modelo es {best_model}\n")
            f.write("• Se aplicó preprocesamiento completo de datos\n")
            f.write("• Se evaluaron múltiples algoritmos de ML\n")
            f.write("• Se generaron visualizaciones comprehensivas\n")
            f.write("• Los resultados demuestran competencia en ML\n")
        
        print(f"Reporte generado: {report_file}")
        
        return self
    
    def run_full_pipeline(self, dataset_path=None, target_column=None):
        """Ejecuta el pipeline completo"""
        print("PIPELINE COMPLETO DE MACHINE LEARNING")
        print("Parcial 3 - Análisis, Modelado y Evaluación")
        print("="*60)
        
        try:
            # Ejecutar todos los pasos
            self.load_data(dataset_path, target_column)
            self.preprocess_data()
            self.train_models()
            self.evaluate_models()
            self.create_visualizations()
            self.generate_report()
            
            print("\n" + "="*60)
            print("PIPELINE COMPLETADO EXITOSAMENTE")
            print("="*60)
            print(f"Resultados guardados en: {self.results_dir}")
            
            # Resumen final
            best_model = self.results['evaluation']['best_model']
            print(f"\nRESUMEN FINAL:")
            print(f"• Dataset: {self.data.shape[0]} muestras, {self.data.shape[1]} variables")
            print(f"• Modelos entrenados: {len(self.results['training']['models_trained'])}")
            print(f"• Mejor modelo: {best_model}")
            
            return True
            
        except Exception as e:
            print(f"\nError en el pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Pipeline ML Parcial 3')
    parser.add_argument('--dataset', type=str, help='Ruta al dataset')
    parser.add_argument('--target', type=str, help='Columna objetivo')
    parser.add_argument('--optimize', action='store_true', help='Optimizar hiperparámetros')
    parser.add_argument('--no-visualize', action='store_true', help='Deshabilitar visualizaciones')
    
    args = parser.parse_args()
    
    # Configurar pipeline
    config = {
        'preprocessing': {
            'handle_missing': True,
            'missing_strategy': 'mean',
            'remove_outliers': True,
            'outlier_method': 'iqr',
            'outlier_threshold': 2.0,
            'encode_categorical': True,
            'encoding_method': 'onehot',
            'scale_features': True,
            'scaling_method': 'standard',
            'create_interactions': True,
            'max_interactions': 5
        },
        'modeling': {
            'optimize_hyperparams': args.optimize,
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
            'create_plots': not args.no_visualize,
            'save_plots': True
        }
    }
    
    # Ejecutar pipeline
    pipeline = MLPipeline(config)
    success = pipeline.run_full_pipeline(args.dataset, args.target)
    
    if success:
        print("\n¡Parcial 3 ML completado exitosamente!")
    else:
        print("\nError en la ejecución del parcial")
        sys.exit(1)

if __name__ == "__main__":
    main()