
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
import numpy as np
import requests

## FUNCIONES DE PREPROCESAMIENTO

## SUbproduct filling Nan

def fill_na(x):
    return x.fillna("No Subproduct")




## DATE FUNCTION

def date_converter(df: pd.DataFrame, column:str) -> pd.DataFrame:
    ''' 
    Transforma las columnas en formato datetime
    '''
    df[column] = pd.to_datetime(df[column], errors = "coerce")
    
    return df


def day_of_week_and_month(df: pd.DataFrame, column_date: str) -> pd.DataFrame:
    ''' 
   Crea tres columnas a partir de una columna de fecha y hora. La columna 1 contiene el día del mes,
     la columna 2 el día de la semana y la columna 3 indica si el día fue fin de semana o no. 
    '''

    df[f'{column_date}_day_of_month'] = df[column_date].dt.day
    df[f'{column_date}_day_week'] = df[column_date].dt.day_of_week
    df[f'{column_date}_weekend'] = df[f'{column_date}_day_week'].apply(lambda x: 1 if (x == 5) | (x == 6) else 0)
    df = df.drop(columns=[column_date])
    return df


# STATE FUNCTION

def region_and_division(df: pd.DataFrame, column_state:str) -> pd.DataFrame:

    ''' 
    Esta función mapea los valores de las siglas de estado en una columna del
    dataFrame de entrada y crea dos nuevas columnas que corresponden a las regiones
    y las divisiones a las que pertenecen dichos estados
    
    '''
    # Diccionary state -> region
    state_to_region = {'CT': 'Northeast','ME': 'Northeast','MA': 'Northeast','NH': 'Northeast','RI': 'Northeast','VT': 'Northeast',
        'NJ': 'Northeast','NY': 'Northeast','PA': 'Northeast','IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest','OH': 'Midwest',
        'WI': 'Midwest','IA': 'Midwest','KS': 'Midwest','MN': 'Midwest','MO': 'Midwest','NE': 'Midwest','ND': 'Midwest','SD': 'Midwest',
        'DE': 'South','FL': 'South','GA': 'South','MD': 'South','NC': 'South','SC': 'South', 'VA': 'South','WV': 'South',
        'AL': 'South', 'KY': 'South','MS': 'South','TN': 'South','AR': 'South','LA': 'South','OK': 'South','TX': 'South',
        'AZ': 'West','CO': 'West','ID': 'West','MT': 'West','NV': 'West','NM': 'West','UT': 'West','WY': 'West','AK': 'West',
        'CA': 'West','HI': 'West','OR': 'West','WA': 'West','DC': 'South'}


    # Diccionary state -> division
    state_to_division = {'CT': 'New England', 'ME': 'New England','MA': 'New England','NH': 'New England','RI': 'New England',
        'VT': 'New England','NJ': 'Middle Atlantic','NY': 'Middle Atlantic','PA': 'Middle Atlantic','IL': 'East North Central',
        'IN': 'East North Central','MI': 'East North Central','OH': 'East North Central','WI': 'East North Central',
        'IA': 'West North Central','KS': 'West North Central','MN': 'West North Central','MO': 'West North Central',
        'NE': 'West North Central','ND': 'West North Central','SD': 'West North Central','DE': 'South Atlantic',
        'FL': 'South Atlantic','GA': 'South Atlantic','MD': 'South Atlantic','NC': 'South Atlantic',
        'SC': 'South Atlantic','VA': 'South Atlantic','WV': 'South Atlantic','DC': 'South Atlantic',
        'AL': 'East South Central','KY': 'East South Central','MS': 'East South Central','TN': 'East South Central',
        'AR': 'West South Central','LA': 'West South Central','OK': 'West South Central','TX': 'West South Central',
        'AZ': 'Mountain','CO': 'Mountain','ID': 'Mountain','MT': 'Mountain','NV': 'Mountain','NM': 'Mountain','UT': 'Mountain','WY': 'Mountain',
        'AK': 'Pacific','CA': 'Pacific','HI': 'Pacific','OR': 'Pacific','WA': 'Pacific'}


    # First, we transform the categories:

    df["regions"] = df[column_state].map(state_to_region).fillna("No state")
    df["divisions"] = df[column_state].map(state_to_division).fillna("No state")
    df = df.drop(columns=[column_state])
    return df




def region_and_division_creator(df_train: pd.DataFrame, df_test:pd.DataFrame, column_state:str) -> pd.DataFrame:

    ''' 
    Esta función crea dos columnas con los criterios de la Oficina del Censo de EE. UU. a partir de la abreviatura 
    de estados y territorios de EE. UU. La columna 1 corresponde a la región y la columna 2 a la división.
    
    '''
    # Diccionary state -> region
    state_to_region = {'CT': 'Northeast','ME': 'Northeast','MA': 'Northeast','NH': 'Northeast','RI': 'Northeast','VT': 'Northeast',
        'NJ': 'Northeast','NY': 'Northeast','PA': 'Northeast','IL': 'Midwest', 'IN': 'Midwest', 'MI': 'Midwest','OH': 'Midwest',
        'WI': 'Midwest','IA': 'Midwest','KS': 'Midwest','MN': 'Midwest','MO': 'Midwest','NE': 'Midwest','ND': 'Midwest','SD': 'Midwest',
        'DE': 'South','FL': 'South','GA': 'South','MD': 'South','NC': 'South','SC': 'South', 'VA': 'South','WV': 'South',
        'AL': 'South', 'KY': 'South','MS': 'South','TN': 'South','AR': 'South','LA': 'South','OK': 'South','TX': 'South',
        'AZ': 'West','CO': 'West','ID': 'West','MT': 'West','NV': 'West','NM': 'West','UT': 'West','WY': 'West','AK': 'West',
        'CA': 'West','HI': 'West','OR': 'West','WA': 'West','DC': 'South'}


    # Diccionary state -> division
    state_to_division = {'CT': 'New England', 'ME': 'New England','MA': 'New England','NH': 'New England','RI': 'New England',
        'VT': 'New England','NJ': 'Middle Atlantic','NY': 'Middle Atlantic','PA': 'Middle Atlantic','IL': 'East North Central',
        'IN': 'East North Central','MI': 'East North Central','OH': 'East North Central','WI': 'East North Central',
        'IA': 'West North Central','KS': 'West North Central','MN': 'West North Central','MO': 'West North Central',
        'NE': 'West North Central','ND': 'West North Central','SD': 'West North Central','DE': 'South Atlantic',
        'FL': 'South Atlantic','GA': 'South Atlantic','MD': 'South Atlantic','NC': 'South Atlantic',
        'SC': 'South Atlantic','VA': 'South Atlantic','WV': 'South Atlantic','DC': 'South Atlantic',
        'AL': 'East South Central','KY': 'East South Central','MS': 'East South Central','TN': 'East South Central',
        'AR': 'West South Central','LA': 'West South Central','OK': 'West South Central','TX': 'West South Central',
        'AZ': 'Mountain','CO': 'Mountain','ID': 'Mountain','MT': 'Mountain','NV': 'Mountain','NM': 'Mountain','UT': 'Mountain','WY': 'Mountain',
        'AK': 'Pacific','CA': 'Pacific','HI': 'Pacific','OR': 'Pacific','WA': 'Pacific'}


    # First, we transform the categories:

    df_train["regions"] = df_train[column_state].map(state_to_region).fillna("No state")
    df_train["divisions"] = df_train[column_state].map(state_to_division).fillna("No state")

    df_test["regions"] = df_test[column_state].map(state_to_region).fillna("No state")
    df_test["divisions"] = df_test[column_state].map(state_to_division).fillna("No state")

    # Transform in OneHotEncoder

    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(sparse_output= False, dtype=int)

    # Regions
    train_encoded_region = encoder.fit_transform(df_train[["regions"]])
    test_encoded_region = encoder.transform(df_test[["regions"]])

    df_train_ohe = pd.DataFrame(train_encoded_region, columns=encoder.get_feature_names_out(["regions"]), index=df_train.index)
    df_test_ohe = pd.DataFrame(test_encoded_region, columns=encoder.get_feature_names_out(["regions"]), index=df_test.index)

    df_train = pd.concat([df_train, df_train_ohe], axis=1)
    df_test = pd.concat([df_test, df_test_ohe], axis=1)

    # Divisions
    train_encoded_division = encoder.fit_transform(df_train[["divisions"]])
    test_encoded_division = encoder.transform(df_test[["divisions"]])

    df_train_ohe = pd.DataFrame(train_encoded_division, columns=encoder.get_feature_names_out(["divisions"]), index=df_train.index)
    df_test_ohe = pd.DataFrame(test_encoded_division, columns=encoder.get_feature_names_out(["divisions"]), index=df_test.index)

    df_train = pd.concat([df_train, df_train_ohe], axis=1)
    df_test = pd.concat([df_test, df_test_ohe], axis=1)

    return df_train, df_test


# ISSUE FUNCTION


def preprocessing_text(text):
    import pandas as pd
    import string

    from spacy.lang.en.stop_words import STOP_WORDS
    ''' 
    Esta función preprocesa el texto de entrada mediante:
    - Conversión a minúsculas
    - Reemplazo de comas por espacios
    - Mantiene los apóstrofes sin cambios
    - Eliminación de todos los demás caracteres de puntuación
    - Eliminación de palabras vacías

    Devuelve el texto limpio como una sola cadena.
    '''
    text = text.lower()
    
    cleaned_chars = []
    for char in text:
        if char == ',':
            cleaned_chars.append(' ')
        elif char == "'":
            cleaned_chars.append(char)
        elif char in string.punctuation:
            continue
        else:
            cleaned_chars.append(char)
    
    cleaned_text = ''.join(cleaned_chars)
    
    words = cleaned_text.split()
    
    return [word for word in words if word not in STOP_WORDS]


def tokenize_column(df: pd.DataFrame, column:str)-> pd.DataFrame:
    ''' 
    Esta función se utilizará para la limpieza de texto en 
    el pipeline de procesamiento
    '''
    df[f"{column}_tokens"] = df[column].apply(preprocessing_text)
    df = df.drop(columns=[column])
    return df


# Company FUNCTION

def company_type_converter(df: pd.DataFrame, column:str) -> pd.DataFrame:

    ''' 
    Esta función agrupa las consultas por tipo de operación comercial.
    '''

    product_to_group_single = {
    'Credit reporting': 'Bureau',
    'Mortgage': 'Lender',
    'Consumer loan': 'Lender',
    'Student loan': 'Lender',
    'Payday loan': 'Lender',
    'Bank account or service': 'Bank',
    'Credit card': 'Bank',
    'Prepaid card': 'Bank',
    'Debt collection': 'Collector',
    'Money transfers': 'Fintech',
    'Other financial service': 'Other'
}
    df["Company_type"] = df[column].map(product_to_group_single)
    df = df.drop(columns=[column])
    return df
    



def company_type(df_train: pd.DataFrame, df_test: pd.DataFrame, column:str) -> pd.DataFrame:

    ''' 
    Esta función agrupa las consultas por tipo de operación comercial.
    '''

    product_to_group_single = {
    'Credit reporting': 'Bureau',
    'Mortgage': 'Lender',
    'Consumer loan': 'Lender',
    'Student loan': 'Lender',
    'Payday loan': 'Lender',
    'Bank account or service': 'Bank',
    'Credit card': 'Bank',
    'Prepaid card': 'Bank',
    'Debt collection': 'Collector',
    'Money transfers': 'Fintech',
    'Other financial service': 'Other'
}
    df_train["Company_type"] = df_train[column].map(product_to_group_single)
    df_test["Company_type"] = df_test[column].map(product_to_group_single)

    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(sparse_output= False, dtype=int)

    # Regions
    train_encoded_company_type = encoder.fit_transform(df_train[["Company_type"]])
    test_encoded_company_type = encoder.transform(df_test[["Company_type"]])

    df_train_ohe = pd.DataFrame(train_encoded_company_type, columns=encoder.get_feature_names_out(["Company_type"]), index=df_train.index)
    df_test_ohe = pd.DataFrame(test_encoded_company_type, columns=encoder.get_feature_names_out(["Company_type"]), index=df_test.index)

    df_train = pd.concat([df_train.drop(columns=["Product"]), df_train_ohe], axis=1)
    df_test = pd.concat([df_test.drop(columns=["Product"]), df_test_ohe], axis=1)

    return df_train, df_test


## FUNCIONES Y CLASES DEL PIPELINE

# Filling NaN  

filling_na = FunctionTransformer(fill_na,validate=False)

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

date_converter_received = FunctionTransformer(date_converter, kw_args={"column": "Date received"}, validate=False)
date_converter_sent = FunctionTransformer(date_converter, kw_args={"column": "Date sent to company"}, validate=False)
day_of_week_month_received = FunctionTransformer(day_of_week_and_month, kw_args={"column_date": "Date received"}, validate = False)
day_of_week_month_sent = FunctionTransformer(day_of_week_and_month, kw_args={"column_date": "Date sent to company"}, validate = False)

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

regions_and_divisions = FunctionTransformer(region_and_division, kw_args={"column_state": "State"}, validate=False)

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

text_cleaner = FunctionTransformer(tokenize_column,kw_args={"column": "Issue"}, validate=False)



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

company_type_converter_pipe = FunctionTransformer(company_type_converter, kw_args={"column": "Product"})

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
    ''' 
    Elimina columnas del dataFrame
    '''

    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors="ignore")