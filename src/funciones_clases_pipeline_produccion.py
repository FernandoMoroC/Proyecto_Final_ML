import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
import sys
sys.path.append("C:/Users/Fernando/Desktop/Proyecto_Final_ML/src")
import preprocess_functions as pr
import numpy as np

# Filling NaN  

filling_na = FunctionTransformer(pr.fill_na,validate=False)

# data["Sub-product"] = filling_na.transform(data["Sub-product"])

# Codificacion con Target Encoder y sustitución de la columna Sub-issue

class FillEncodeColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column, smooth="auto", cv=5, categories = "auto", random_state=42):
        self.column = column
        self.smooth = smooth
        self.cv = cv
        self.random_state = random_state
        self.categories = categories
        self.encoder = None

    def fit(self, X, y=None):
        self.encoder= TargetEncoder(
            smooth=self.smooth,
            cv=self.cv,
            random_state=self.random_state,
            categories = self.categories 
        )
        self.encoder.fit(X[[self.column]], y)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        encoded = self.encoder.transform(X[[self.column]])
        X[self.column] = encoded
        return X

## Funciones Dates

date_converter_received = FunctionTransformer(pr.date_converter, kw_args={"column": "Date received"}, validate=False)
date_converter_sent = FunctionTransformer(pr.date_converter, kw_args={"column": "Date sent to company"}, validate=False)
day_of_week_month_received = FunctionTransformer(pr.day_of_week_and_month, kw_args={"column_date": "Date received"}, validate = False)
day_of_week_month_sent = FunctionTransformer(pr.day_of_week_and_month, kw_args={"column_date": "Date sent to company"}, validate = False)

# Al contrario que el notebook de preprocessing, he incorporado el drop
# en la función, por lo que elimina la columna original al aplicar la función


## Funciones Product

class ProductEncoder(BaseEstimator, TransformerMixin):

    def __init__(self,column, sparse_output = False, dtype=int):
        self.column = column
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.encoder = None

    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse_output=self.sparse_output,
                                     dtype=self.dtype)
        self.encoder.fit(X[[self.column]], y)
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        encoded= self.encoder.transform(X[[self.column]])
        encoded = pd.DataFrame(encoded, 
                               columns = self.encoder.get_feature_names_out([self.column]), 
                               index=X.index)
        X = pd.concat([X, encoded], axis=1)
        return X
        

## Funciones State

regions_and_divisions = FunctionTransformer(pr.region_and_division, kw_args={"column_state": "State"}, validate=False)

# Las columnas de salida se llaman regions y divisions

class RegionDivisionEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self,column, sparse_output = False, dtype=int):
        self.column = column
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.encoder = None

    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse_output=self.sparse_output,
                                     dtype=self.dtype)
        self.encoder.fit(X[[self.column]], y)
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        encoded= self.encoder.transform(X[[self.column]])
        encoded = pd.DataFrame(encoded, 
                               columns = self.encoder.get_feature_names_out([self.column]), 
                               index=X.index)
        X = pd.concat([X, encoded], axis=1)
        X = X.drop(columns=[self.column])
        return X
    
    # Recuerda que hay que hacerlo para la columna regions y divisions


## Functions Issue

# Limpieza de texto

text_cleaner = FunctionTransformer(pr.tokenize_column,kw_args={"column": "Issue"}, validate=False)



class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """
    Convierte una columna de listas de tokens en embeddings promedio usando Word2Vec.
    """
    def __init__(self, column="Issue_tokens", vector_size=25, window=5, min_count=1, workers=4):
        self.column = column
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model_ = None  # Modelo Word2Vec entrenado

    def fit(self, X, y=None):
        self.model_ = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers
        )
        return self

    def transform(self, X):
        # Función para obtener embedding promedio de una lista de tokens
        def sentence_vector(tokens):
            vectors = [self.model_.wv[word] for word in tokens if word in self.model_.wv]
            if len(vectors) == 0:
                return np.zeros(self.vector_size)
            return np.mean(vectors, axis=0)

        # Aplicar transformación
        embeddings = X[self.column].apply(sentence_vector)
        embedding_matrix = np.vstack(embeddings.values)
        
        # Crear DataFrame de embeddings
        embedding_df = pd.DataFrame(
            embedding_matrix,
            columns=[f"embedding_{i}" for i in range(self.vector_size)],
            index=X.index
        )

        # Concatenar embeddings al DataFrame original (sin la columna de tokens original)
        X_transformed = pd.concat([X.drop(columns=[self.column], errors='ignore'), embedding_df], axis=1)
        return X_transformed

# Hay que hacerlo con la columna Issue_tokens


## Function company

company_type_converter = FunctionTransformer(pr.company_type_converter, kw_args={"column": "Product"})

class Company_type_from_product(BaseEstimator, TransformerMixin):

    def __init__(self,column, sparse_output = False, dtype=int):
        self.column = column
        self.sparse_output = sparse_output
        self.dtype = dtype
        self.encoder = None

    def fit(self, X, y=None):
        self.encoder = OneHotEncoder(sparse_output=self.sparse_output,
                                     dtype=self.dtype)
        self.encoder.fit(X[[self.column]], y)
        self.n_features_in_ = X.shape[1]
        return self
    
    def transform(self, X):
        encoded= self.encoder.transform(X[[self.column]])
        encoded = pd.DataFrame(encoded, 
                               columns = self.encoder.get_feature_names_out([self.column]), 
                               index=X.index)
        X = pd.concat([X, encoded], axis=1)
        X = X.drop(columns=[self.column])
        return X
    

# Hay que hacerlo para la columna Company_type


## ELIMINADO DE COLUMNAS

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors="ignore")