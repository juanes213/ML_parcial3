"""
Módulo de Preprocesamiento de Datos
Parcial 3 - Machine Learning
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Clase para preprocesamiento completo de datos"""
    
    def __init__(self, data, target_column=None):
        """
        Inicializa el preprocesador
        
        Args:
            data (pd.DataFrame): Dataset original
            target_column (str): Nombre de la columna objetivo
        """
        self.data = data.copy()
        self.target_column = target_column
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remover columna objetivo de las listas si está presente
        if target_column:
            if target_column in self.numeric_columns:
                self.numeric_columns.remove(target_column)
            if target_column in self.categorical_columns:
                self.categorical_columns.remove(target_column)
        
        # Objetos para transformaciones
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.preprocessing_steps = []
    
    def handle_missing_values(self, strategy='mean', categorical_strategy='most_frequent'):
        """
        Maneja valores faltantes
        
        Args:
            strategy (str): Estrategia para variables numéricas ('mean', 'median', 'constant')
            categorical_strategy (str): Estrategia para variables categóricas ('most_frequent', 'constant')
        """
        print(f"Manejando valores faltantes...")
        print(f"Estrategia numérica: {strategy}")
        print(f"Estrategia categórica: {categorical_strategy}")
        
        # Variables numéricas
        if self.numeric_columns:
            if strategy in ['mean', 'median']:
                imputer_num = SimpleImputer(strategy=strategy)
            else:
                imputer_num = SimpleImputer(strategy='constant', fill_value=0)
            
            self.data[self.numeric_columns] = imputer_num.fit_transform(self.data[self.numeric_columns])
            self.imputers['numeric'] = imputer_num
        
        # Variables categóricas
        if self.categorical_columns:
            if categorical_strategy == 'most_frequent':
                imputer_cat = SimpleImputer(strategy='most_frequent')
            else:
                imputer_cat = SimpleImputer(strategy='constant', fill_value='Unknown')
            
            self.data[self.categorical_columns] = imputer_cat.fit_transform(self.data[self.categorical_columns])
            self.imputers['categorical'] = imputer_cat
        
        self.preprocessing_steps.append(f"Missing values handled: num={strategy}, cat={categorical_strategy}")
        
        missing_after = self.data.isnull().sum().sum()
        print(f"Valores faltantes después del procesamiento: {missing_after}")
        
        return self
    
    def remove_outliers(self, method='iqr', threshold=1.5):
        """
        Remueve outliers de variables numéricas
        
        Args:
            method (str): Método ('iqr', 'zscore')
            threshold (float): Umbral para detección
        """
        print(f"Removiendo outliers usando método {method} con umbral {threshold}")
        
        initial_rows = len(self.data)
        outlier_indices = set()
        
        for col in self.numeric_columns:
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)].index
            
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outliers = self.data[z_scores > threshold].index
            
            outlier_indices.update(outliers)
            print(f"  {col}: {len(outliers)} outliers detectados")
        
        # Remover outliers
        self.data = self.data.drop(outlier_indices)
        self.data = self.data.reset_index(drop=True)
        
        final_rows = len(self.data)
        removed_rows = initial_rows - final_rows
        
        print(f"Filas removidas: {removed_rows} ({removed_rows/initial_rows*100:.2f}%)")
        print(f"Filas restantes: {final_rows}")
        
        self.preprocessing_steps.append(f"Outliers removed: {method}, threshold={threshold}, removed={removed_rows}")
        
        return self
    
    def encode_categorical_variables(self, method='onehot', max_categories=10):
        """
        Codifica variables categóricas
        
        Args:
            method (str): Método de codificación ('onehot', 'label')
            max_categories (int): Máximo número de categorías para one-hot encoding
        """
        print(f"Codificando variables categóricas usando {method}")
        
        for col in self.categorical_columns:
            unique_values = self.data[col].nunique()
            print(f"  {col}: {unique_values} categorías únicas")
            
            if method == 'onehot' and unique_values <= max_categories:
                # One-hot encoding
                encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
                encoded = encoder.fit_transform(self.data[[col]])
                
                # Crear nombres de columnas
                feature_names = [f"{col}_{category}" for category in encoder.categories_[0][1:]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names)
                
                # Reemplazar columna original
                self.data = pd.concat([self.data.drop(col, axis=1), encoded_df], axis=1)
                self.encoders[col] = encoder
                
                print(f"    One-hot encoding aplicado: {len(feature_names)} nuevas columnas")
            
            else:
                # Label encoding
                encoder = LabelEncoder()
                self.data[col] = encoder.fit_transform(self.data[col])
                self.encoders[col] = encoder
                
                print(f"    Label encoding aplicado")
        
        # Actualizar lista de columnas numéricas
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column and self.target_column in self.numeric_columns:
            self.numeric_columns.remove(self.target_column)
        
        self.categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.preprocessing_steps.append(f"Categorical encoding: {method}, max_categories={max_categories}")
        
        return self
    
    def scale_features(self, method='standard'):
        """
        Escala variables numéricas
        
        Args:
            method (str): Método de escalado ('standard', 'minmax', 'robust')
        """
        print(f"Escalando features usando {method} scaling")
        
        if not self.numeric_columns:
            print("No hay columnas numéricas para escalar")
            return self
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Método de escalado no soportado: {method}")
        
        # Aplicar escalado
        self.data[self.numeric_columns] = scaler.fit_transform(self.data[self.numeric_columns])
        self.scalers['features'] = scaler
        
        print(f"Escalado aplicado a {len(self.numeric_columns)} columnas")
        
        self.preprocessing_steps.append(f"Feature scaling: {method}")
        
        return self
    
    def create_feature_interactions(self, max_interactions=5):
        """
        Crea interacciones entre features numéricas
        
        Args:
            max_interactions (int): Número máximo de interacciones a crear
        """
        print(f"Creando interacciones entre features (máximo {max_interactions})")
        
        if len(self.numeric_columns) < 2:
            print("Insuficientes columnas numéricas para crear interacciones")
            return self
        
        interactions_created = 0
        
        for i, col1 in enumerate(self.numeric_columns):
            if interactions_created >= max_interactions:
                break
            
            for j, col2 in enumerate(self.numeric_columns[i+1:], i+1):
                if interactions_created >= max_interactions:
                    break
                
                # Crear interacción multiplicativa
                interaction_name = f"{col1}_x_{col2}"
                self.data[interaction_name] = self.data[col1] * self.data[col2]
                interactions_created += 1
                
                print(f"  Creada interacción: {interaction_name}")
        
        # Actualizar lista de columnas numéricas
        self.numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column and self.target_column in self.numeric_columns:
            self.numeric_columns.remove(self.target_column)
        
        self.preprocessing_steps.append(f"Feature interactions created: {interactions_created}")
        
        return self
    
    def split_data(self, test_size=0.2, validation_size=0.1, random_state=42, stratify=True):
        """
        Divide los datos en conjuntos de entrenamiento, validación y prueba
        
        Args:
            test_size (float): Proporción del conjunto de prueba
            validation_size (float): Proporción del conjunto de validación
            random_state (int): Semilla para reproducibilidad
            stratify (bool): Si aplicar estratificación basada en la variable objetivo
        """
        print(f"Dividiendo datos: train={1-test_size-validation_size:.1f}, val={validation_size:.1f}, test={test_size:.1f}")
        
        if not self.target_column:
            raise ValueError("Debe especificar target_column para dividir los datos")
        
        # Preparar características y objetivo
        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]
        
        # Estratificación solo para variables categóricas o con pocos valores únicos
        stratify_param = None
        if stratify and (y.dtype == 'object' or y.nunique() <= 10):
            stratify_param = y
        
        # Primera división: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        # Segunda división: train vs val
        if validation_size > 0:
            val_size_adjusted = validation_size / (1 - test_size)
            stratify_temp = None
            if stratify and (y_temp.dtype == 'object' or y_temp.nunique() <= 10):
                stratify_temp = y_temp
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, 
                random_state=random_state, stratify=stratify_temp
            )
        else:
            X_train, X_val, y_train, y_val = X_temp, None, y_temp, None
        
        print(f"Tamaños de conjuntos:")
        print(f"  Entrenamiento: {X_train.shape[0]} muestras")
        if X_val is not None:
            print(f"  Validación: {X_val.shape[0]} muestras")
        print(f"  Prueba: {X_test.shape[0]} muestras")
        
        self.preprocessing_steps.append(f"Data split: test={test_size}, val={validation_size}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_preprocessing_summary(self):
        """Retorna resumen del preprocesamiento realizado"""
        summary = {
            'original_shape': self.data.shape,
            'numeric_columns': len(self.numeric_columns),
            'categorical_columns': len(self.categorical_columns),
            'preprocessing_steps': self.preprocessing_steps,
            'transformers': {
                'scalers': list(self.scalers.keys()),
                'encoders': list(self.encoders.keys()),
                'imputers': list(self.imputers.keys())
            }
        }
        
        return summary
    
    def fit_transform(self, data):
        """Aplica todas las transformaciones ajustadas a nuevos datos"""
        processed_data = data.copy()
        
        # Aplicar imputación
        for key, imputer in self.imputers.items():
            if key == 'numeric' and self.numeric_columns:
                processed_data[self.numeric_columns] = imputer.transform(processed_data[self.numeric_columns])
            elif key == 'categorical' and self.categorical_columns:
                processed_data[self.categorical_columns] = imputer.transform(processed_data[self.categorical_columns])
        
        # Aplicar codificación
        for col, encoder in self.encoders.items():
            if hasattr(encoder, 'transform'):
                processed_data[col] = encoder.transform(processed_data[col])
        
        # Aplicar escalado
        for key, scaler in self.scalers.items():
            if key == 'features' and self.numeric_columns:
                processed_data[self.numeric_columns] = scaler.transform(processed_data[self.numeric_columns])
        
        return processed_data

def demo_preprocessing():
    """Función de demostración del preprocesamiento"""
    print("Demostrando preprocesamiento de datos...")
    
    # Importar datos de muestra del módulo de análisis
    from data_analysis import DataAnalyzer
    
    analyzer = DataAnalyzer(pd.DataFrame())
    sample_data = analyzer.generate_sample_data(1000)
    
    # Añadir algunos valores faltantes para demostración
    np.random.seed(42)
    missing_indices = np.random.choice(sample_data.index, size=50, replace=False)
    sample_data.loc[missing_indices[:25], 'income'] = np.nan
    sample_data.loc[missing_indices[25:], 'education'] = np.nan
    
    print(f"Dataset original: {sample_data.shape}")
    print(f"Valores faltantes: {sample_data.isnull().sum().sum()}")
    
    # Crear preprocesador
    preprocessor = DataPreprocessor(sample_data, target_column='purchase')
    
    # Aplicar preprocesamiento completo
    preprocessor.handle_missing_values()
    preprocessor.remove_outliers(method='iqr', threshold=2.0)
    preprocessor.encode_categorical_variables(method='onehot')
    preprocessor.create_feature_interactions(max_interactions=3)
    preprocessor.scale_features(method='standard')
    
    # Dividir datos
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()
    
    # Mostrar resumen
    summary = preprocessor.get_preprocessing_summary()
    print("\n=== RESUMEN DEL PREPROCESAMIENTO ===")
    for step in summary['preprocessing_steps']:
        print(f"✓ {step}")
    
    return preprocessor, (X_train, X_val, X_test, y_train, y_val, y_test)

if __name__ == "__main__":
    demo_preprocessing()