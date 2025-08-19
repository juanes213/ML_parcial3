"""
Módulo de Análisis Exploratorio de Datos
Parcial 3 - Machine Learning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DataAnalyzer:
    """Clase para realizar análisis exploratorio de datos"""
    
    def __init__(self, data):
        """
        Inicializa el analizador con los datos
        
        Args:
            data (pd.DataFrame): Dataset a analizar
        """
        self.data = data.copy()
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def basic_info(self):
        """Muestra información básica del dataset"""
        print("=== INFORMACIÓN BÁSICA DEL DATASET ===")
        print(f"Forma del dataset: {self.data.shape}")
        print(f"Columnas numéricas: {len(self.numeric_columns)}")
        print(f"Columnas categóricas: {len(self.categorical_columns)}")
        print("\n--- Tipos de datos ---")
        print(self.data.dtypes)
        print("\n--- Información general ---")
        print(self.data.info())
        
        return {
            'shape': self.data.shape,
            'numeric_cols': len(self.numeric_columns),
            'categorical_cols': len(self.categorical_columns),
            'total_memory': self.data.memory_usage(deep=True).sum()
        }
    
    def missing_values_analysis(self):
        """Analiza valores faltantes"""
        print("\n=== ANÁLISIS DE VALORES FALTANTES ===")
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        
        missing_df = pd.DataFrame({
            'Columna': missing.index,
            'Valores_Faltantes': missing.values,
            'Porcentaje': missing_pct.values
        })
        
        missing_df = missing_df[missing_df['Valores_Faltantes'] > 0].sort_values('Valores_Faltantes', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df)
        else:
            print("No hay valores faltantes en el dataset.")
        
        return missing_df
    
    def descriptive_statistics(self):
        """Estadísticas descriptivas para variables numéricas"""
        print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
        desc_stats = self.data[self.numeric_columns].describe()
        print(desc_stats)
        
        # Medidas adicionales
        additional_stats = pd.DataFrame({
            'Skewness': self.data[self.numeric_columns].skew(),
            'Kurtosis': self.data[self.numeric_columns].kurtosis()
        })
        
        print("\n--- Asimetría y Curtosis ---")
        print(additional_stats)
        
        return desc_stats, additional_stats
    
    def outlier_detection(self):
        """Detecta outliers usando el método IQR"""
        print("\n=== DETECCIÓN DE OUTLIERS ===")
        outliers_info = {}
        
        for col in self.numeric_columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            outliers_count = len(outliers)
            outliers_pct = (outliers_count / len(self.data)) * 100
            
            outliers_info[col] = {
                'count': outliers_count,
                'percentage': outliers_pct,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            print(f"{col}: {outliers_count} outliers ({outliers_pct:.2f}%)")
        
        return outliers_info
    
    def correlation_analysis(self):
        """Análisis de correlaciones"""
        print("\n=== ANÁLISIS DE CORRELACIONES ===")
        
        if len(self.numeric_columns) < 2:
            print("Insuficientes variables numéricas para análisis de correlación.")
            return None
        
        correlation_matrix = self.data[self.numeric_columns].corr()
        
        # Encontrar correlaciones más altas
        corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j]
                ))
        
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        print("Top 10 correlaciones más altas:")
        for i, (var1, var2, corr) in enumerate(corr_pairs[:10]):
            print(f"{i+1}. {var1} - {var2}: {corr:.3f}")
        
        return correlation_matrix
    
    def categorical_analysis(self):
        """Análisis de variables categóricas"""
        if not self.categorical_columns:
            print("\n=== No hay variables categóricas en el dataset ===")
            return None
        
        print("\n=== ANÁLISIS DE VARIABLES CATEGÓRICAS ===")
        categorical_info = {}
        
        for col in self.categorical_columns:
            unique_values = self.data[col].nunique()
            value_counts = self.data[col].value_counts()
            
            categorical_info[col] = {
                'unique_values': unique_values,
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'value_counts': value_counts
            }
            
            print(f"\n--- {col} ---")
            print(f"Valores únicos: {unique_values}")
            print("Distribución de valores:")
            print(value_counts.head(10))
        
        return categorical_info
    
    def generate_sample_data(self, n_samples=1000):
        """
        Genera datos de muestra para demostración
        
        Args:
            n_samples (int): Número de muestras a generar
        
        Returns:
            pd.DataFrame: Dataset de muestra
        """
        np.random.seed(42)
        
        # Generar variables independientes
        age = np.random.normal(35, 10, n_samples)
        age = np.clip(age, 18, 80)
        
        income = np.random.lognormal(10, 0.5, n_samples)
        income = np.clip(income, 20000, 200000)
        
        education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                   n_samples, p=[0.3, 0.4, 0.2, 0.1])
        
        # Variable objetivo con dependencias
        education_score = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
        ed_numeric = [education_score[ed] for ed in education]
        
        # Probabilidad de compra basada en edad, ingresos y educación
        prob = (0.3 * (age - 18) / 62 + 
                0.4 * (income - 20000) / 180000 + 
                0.3 * (np.array(ed_numeric) - 1) / 3 +
                np.random.normal(0, 0.1, n_samples))
        
        prob = np.clip(prob, 0, 1)
        purchase = np.random.binomial(1, prob, n_samples)
        
        # Crear DataFrame
        sample_data = pd.DataFrame({
            'age': age,
            'income': income,
            'education': education,
            'purchase': purchase
        })
        
        return sample_data
    
    def run_complete_analysis(self):
        """Ejecuta análisis completo"""
        print("ANÁLISIS EXPLORATORIO DE DATOS COMPLETO")
        print("="*50)
        
        basic_info = self.basic_info()
        missing_info = self.missing_values_analysis()
        desc_stats, additional_stats = self.descriptive_statistics()
        outliers_info = self.outlier_detection()
        correlation_matrix = self.correlation_analysis()
        categorical_info = self.categorical_analysis()
        
        return {
            'basic_info': basic_info,
            'missing_info': missing_info,
            'descriptive_stats': desc_stats,
            'additional_stats': additional_stats,
            'outliers_info': outliers_info,
            'correlation_matrix': correlation_matrix,
            'categorical_info': categorical_info
        }

# Función de demostración
def demo_analysis():
    """Función de demostración del análisis"""
    print("Generando datos de muestra para demostración...")
    
    # Crear instancia del analizador con datos de muestra
    analyzer = DataAnalyzer(pd.DataFrame())  # Temporal
    sample_data = analyzer.generate_sample_data(1000)
    
    # Crear nuevo analizador con los datos de muestra
    analyzer = DataAnalyzer(sample_data)
    
    # Ejecutar análisis completo
    results = analyzer.run_complete_analysis()
    
    return analyzer, results

if __name__ == "__main__":
    demo_analysis()