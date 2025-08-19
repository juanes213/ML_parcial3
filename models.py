"""
Módulo de Modelos de Machine Learning
Parcial 3 - Machine Learning
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, validation_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class MLModelTrainer:
    """Clase para entrenar y evaluar múltiples modelos de ML"""
    
    def __init__(self, problem_type='classification'):
        """
        Inicializa el entrenador de modelos
        
        Args:
            problem_type (str): Tipo de problema ('classification', 'regression')
        """
        self.problem_type = problem_type
        self.models = {}
        self.trained_models = {}
        self.best_params = {}
        self.training_results = {}
        
        # Inicializar modelos predefinidos
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicializa modelos predefinidos según el tipo de problema"""
        if self.problem_type == 'classification':
            self.models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                'Random Forest': RandomForestClassifier(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'SVM': SVC(random_state=42, probability=True),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Naive Bayes': GaussianNB(),
                'Decision Tree': DecisionTreeClassifier(random_state=42)
            }
        else:
            # Para regresión (si se implementa en el futuro)
            from sklearn.linear_model import LinearRegression
            from sklearn.ensemble import RandomForestRegressor
            
            self.models = {
                'Linear Regression': LinearRegression(),
                'Random Forest': RandomForestRegressor(random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42)
            }
    
    def get_hyperparameter_grids(self):
        """Retorna grids de hiperparámetros para optimización"""
        if self.problem_type == 'classification':
            return {
                'Logistic Regression': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                },
                'Random Forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'Gradient Boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'SVM': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'K-Nearest Neighbors': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                },
                'Decision Tree': {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'criterion': ['gini', 'entropy']
                }
            }
        return {}
    
    def train_single_model(self, model_name, X_train, y_train, optimize_hyperparams=False, cv_folds=5):
        """
        Entrena un modelo individual
        
        Args:
            model_name (str): Nombre del modelo
            X_train: Características de entrenamiento
            y_train: Variable objetivo de entrenamiento
            optimize_hyperparams (bool): Si optimizar hiperparámetros
            cv_folds (int): Número de folds para validación cruzada
        """
        print(f"Entrenando {model_name}...")
        
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no está disponible")
        
        model = self.models[model_name]
        
        if optimize_hyperparams:
            # Optimización de hiperparámetros
            param_grid = self.get_hyperparameter_grids().get(model_name, {})
            
            if param_grid:
                print(f"  Optimizando hiperparámetros con GridSearch ({cv_folds}-fold CV)...")
                
                grid_search = GridSearchCV(
                    model, param_grid, cv=cv_folds, 
                    scoring='accuracy', n_jobs=-1, verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                model = grid_search.best_estimator_
                self.best_params[model_name] = grid_search.best_params_
                
                print(f"  Mejores parámetros: {grid_search.best_params_}")
                print(f"  Mejor score CV: {grid_search.best_score_:.4f}")
            else:
                print(f"  No hay grid de hiperparámetros definido para {model_name}")
                model.fit(X_train, y_train)
        else:
            # Entrenamiento con parámetros por defecto
            model.fit(X_train, y_train)
        
        # Validación cruzada
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
        
        # Guardar modelo entrenado y resultados
        self.trained_models[model_name] = model
        self.training_results[model_name] = {
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores,
            'best_params': self.best_params.get(model_name, {})
        }
        
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return model
    
    def train_all_models(self, X_train, y_train, optimize_hyperparams=False, cv_folds=5):
        """
        Entrena todos los modelos disponibles
        
        Args:
            X_train: Características de entrenamiento
            y_train: Variable objetivo de entrenamiento
            optimize_hyperparams (bool): Si optimizar hiperparámetros
            cv_folds (int): Número de folds para validación cruzada
        """
        print("Entrenando todos los modelos...")
        print("="*50)
        
        for model_name in self.models.keys():
            try:
                self.train_single_model(
                    model_name, X_train, y_train, 
                    optimize_hyperparams, cv_folds
                )
                print()
            except Exception as e:
                print(f"Error entrenando {model_name}: {str(e)}")
                continue
        
        # Resumen de resultados
        self.print_training_summary()
    
    def print_training_summary(self):
        """Imprime resumen de resultados de entrenamiento"""
        print("RESUMEN DE ENTRENAMIENTO")
        print("="*50)
        
        results_df = pd.DataFrame({
            'Model': list(self.training_results.keys()),
            'CV_Mean': [self.training_results[model]['cv_mean'] for model in self.training_results.keys()],
            'CV_Std': [self.training_results[model]['cv_std'] for model in self.training_results.keys()]
        })
        
        results_df = results_df.sort_values('CV_Mean', ascending=False)
        
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        best_model = results_df.iloc[0]['Model']
        print(f"\nMejor modelo: {best_model} (CV Score: {results_df.iloc[0]['CV_Mean']:.4f})")
        
        return results_df
    
    def predict_with_all_models(self, X_test):
        """
        Realiza predicciones con todos los modelos entrenados
        
        Args:
            X_test: Características de prueba
        
        Returns:
            dict: Predicciones de todos los modelos
        """
        predictions = {}
        
        for model_name, model in self.trained_models.items():
            try:
                pred = model.predict(X_test)
                predictions[model_name] = pred
            except Exception as e:
                print(f"Error en predicción con {model_name}: {str(e)}")
        
        return predictions
    
    def predict_proba_with_all_models(self, X_test):
        """
        Realiza predicciones de probabilidad con todos los modelos
        
        Args:
            X_test: Características de prueba
        
        Returns:
            dict: Probabilidades de todos los modelos
        """
        probabilities = {}
        
        for model_name, model in self.trained_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)
                    probabilities[model_name] = proba
                else:
                    print(f"Modelo {model_name} no soporta predict_proba")
            except Exception as e:
                print(f"Error en probabilidades con {model_name}: {str(e)}")
        
        return probabilities
    
    def get_feature_importance(self, feature_names):
        """
        Obtiene importancia de características para modelos que la soporten
        
        Args:
            feature_names (list): Nombres de las características
        
        Returns:
            dict: Importancia de características por modelo
        """
        feature_importance = {}
        
        for model_name, model in self.trained_models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    feature_importance[model_name] = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importance
                    }).sort_values('importance', ascending=False)
                
                elif hasattr(model, 'coef_'):
                    # Para modelos lineales
                    coef = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                    feature_importance[model_name] = pd.DataFrame({
                        'feature': feature_names,
                        'importance': coef
                    }).sort_values('importance', ascending=False)
                
            except Exception as e:
                print(f"Error obteniendo importancia para {model_name}: {str(e)}")
        
        return feature_importance
    
    def learning_curves_analysis(self, X_train, y_train, model_name, cv_folds=5):
        """
        Analiza curvas de aprendizaje para un modelo específico
        
        Args:
            X_train: Características de entrenamiento
            y_train: Variable objetivo de entrenamiento
            model_name (str): Nombre del modelo
            cv_folds (int): Número de folds para CV
        
        Returns:
            dict: Resultados de las curvas de aprendizaje
        """
        from sklearn.model_selection import learning_curve
        
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no disponible")
        
        model = self.models[model_name]
        
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=cv_folds, 
            train_sizes=train_sizes, scoring='accuracy', n_jobs=-1
        )
        
        results = {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': train_scores.mean(axis=1),
            'train_scores_std': train_scores.std(axis=1),
            'val_scores_mean': val_scores.mean(axis=1),
            'val_scores_std': val_scores.std(axis=1)
        }
        
        return results
    
    def validation_curve_analysis(self, X_train, y_train, model_name, param_name, param_range, cv_folds=5):
        """
        Analiza curva de validación para un parámetro específico
        
        Args:
            X_train: Características de entrenamiento
            y_train: Variable objetivo de entrenamiento
            model_name (str): Nombre del modelo
            param_name (str): Nombre del parámetro
            param_range (list): Rango de valores del parámetro
            cv_folds (int): Número de folds para CV
        
        Returns:
            dict: Resultados de la curva de validación
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no disponible")
        
        model = self.models[model_name]
        
        train_scores, val_scores = validation_curve(
            model, X_train, y_train, param_name=param_name,
            param_range=param_range, cv=cv_folds, scoring='accuracy', n_jobs=-1
        )
        
        results = {
            'param_range': param_range,
            'train_scores_mean': train_scores.mean(axis=1),
            'train_scores_std': train_scores.std(axis=1),
            'val_scores_mean': val_scores.mean(axis=1),
            'val_scores_std': val_scores.std(axis=1)
        }
        
        return results
    
    def ensemble_prediction(self, X_test, method='voting'):
        """
        Realiza predicción ensemble con todos los modelos
        
        Args:
            X_test: Características de prueba
            method (str): Método de ensemble ('voting', 'weighted')
        
        Returns:
            array: Predicciones ensemble
        """
        predictions = self.predict_with_all_models(X_test)
        
        if not predictions:
            raise ValueError("No hay modelos entrenados para ensemble")
        
        if method == 'voting':
            # Votación mayoritaria
            pred_array = np.array(list(predictions.values())).T
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x).argmax(), axis=1, arr=pred_array
            )
        
        elif method == 'weighted':
            # Votación ponderada por CV score
            weights = np.array([
                self.training_results[model]['cv_mean'] 
                for model in predictions.keys()
            ])
            weights = weights / weights.sum()  # Normalizar
            
            pred_array = np.array(list(predictions.values())).T
            ensemble_pred = np.apply_along_axis(
                lambda x: np.bincount(x, weights=weights).argmax(), axis=1, arr=pred_array
            )
        
        return ensemble_pred

def demo_models():
    """Función de demostración de entrenamiento de modelos"""
    print("Demostrando entrenamiento de modelos ML...")
    
    # Importar datos preprocesados
    from preprocessing import demo_preprocessing
    
    preprocessor, (X_train, X_val, X_test, y_train, y_val, y_test) = demo_preprocessing()
    
    print(f"\nForma de datos de entrenamiento: {X_train.shape}")
    print(f"Distribución de clases: {pd.Series(y_train).value_counts().to_dict()}")
    
    # Crear entrenador de modelos
    trainer = MLModelTrainer(problem_type='classification')
    
    # Entrenar algunos modelos (sin optimización para demo rápida)
    models_to_train = ['Logistic Regression', 'Random Forest', 'Gradient Boosting']
    
    for model_name in models_to_train:
        trainer.train_single_model(model_name, X_train, y_train, optimize_hyperparams=False)
    
    # Predicciones en conjunto de prueba
    predictions = trainer.predict_with_all_models(X_test)
    
    # Mostrar importancia de características
    feature_names = X_train.columns.tolist()
    importance = trainer.get_feature_importance(feature_names)
    
    print("\n=== IMPORTANCIA DE CARACTERÍSTICAS (Top 5) ===")
    for model_name, imp_df in importance.items():
        print(f"\n{model_name}:")
        print(imp_df.head().to_string(index=False, float_format='%.4f'))
    
    return trainer, (X_train, X_val, X_test, y_train, y_val, y_test), predictions

if __name__ == "__main__":
    demo_models()