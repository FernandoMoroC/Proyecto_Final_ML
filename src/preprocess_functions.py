##FUNCIONES USADAS EN EL PREPROCESAMIENTO DE LOS DATOS


import pandas as pd
import numpy as np
import requests

## SUbproduct filling Nan

def fill_na(x):
    ''' 
    Rellena valores faltantes con la cadena "No Subproduct"
    '''
    return x.fillna("No Subproduct")

# ZIP CODE FUNCTION

def zip_to_string(x) -> pd.DataFrame:
    '''  
    Transforma los valores de la columna de código postal en cadenas. 
    Los valores NaN no se eliminan. Cuando la longitud del código es 4, se añade un cero delante del código.
    '''
    if pd.isna(x):
        return x  # devolver tal cual si es NaN
    
    # Quitar espacios y convertir a string
    x = str(x).strip()
    
    # Si viene como 1234.0 -> "1234"
    if "." in x:
        x = x.split(".")[0]
    
    # Añadir cero inicial si es de 4 dígitos
    if len(x) == 4:
        return "0" + x
    
    return x
    
    

    df[zip_code_column] = df[zip_code_column].apply(lambda x: str(int(x)) if pd.notnull(x) else x)
    df[zip_code_column] = df[zip_code_column].apply(lambda x: "0" + x if len(x) == 4 else x)
    return df


def search_state_from_zip(df: pd.DataFrame, zip_code_column: str, state_column: str) -> pd.DataFrame:

    ''' 
    Esta función busca, basándose en los valores del código postal, el nombre del estado al que pertenecen 
    mediante la API de Zippopotam. Si no lo encuentra, mostrará un mensaje indicando que no se encontraron 
    coincidencias.
    '''

    zip_codes_to_search = df[zip_code_column][df[state_column].isna() & df[zip_code_column].notna()]

    zip_codes_list = [x for x in (list(zip_codes_to_search))]

    states_abbreviations_zip = {}
    for zip_code in zip_codes_list:
        try:
            url = f"http://api.zippopotam.us/US/{zip_code}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                state_abbreviation = data["places"][0]["state abbreviation"]
                states_abbreviations_zip[zip_code] = state_abbreviation
            else:
                print(f"ZIP code {zip_code} not found in API.")
        except Exception as e:
            print(f"Error retrieving ZIP code {zip_code}: {e}")
    
    df[state_column] = df[state_column].fillna(df[zip_code_column].astype(str).map(states_abbreviations_zip))

    return df

def search_city_from_zip(df: pd.DataFrame, zip_code_column: str) -> pd.DataFrame:
    import requests
    from time import sleep

    ''' 
    Esta función toma como entrada un DataFrame y el nombre de una columna que contiene códigos postales de 
    EE. UU. Busca el nombre de la ciudad de donde proviene el código postal y devuelve una nueva columna 
    llamada "city_column" en el mismo DataFrame con los nombres de las ciudades. Utiliza la API zippopotam.
    
    '''

    zip_codes_list = df[zip_code_column].dropna().unique()

    cities_zip = {}
    for zip_code in zip_codes_list:
        try:
            url = f"http://api.zippopotam.us/US/{zip_code}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                city_name = data["places"][0]["place name"]
                cities_zip[zip_code] = city_name
            else:
                cities_zip[zip_code] = None
        except Exception as e:
            print(f"Error retrieving ZIP code {zip_code}: {e}")
        sleep(0.2)
    print("creating city column")
    df["city_column"] = df[zip_code_column].map(cities_zip)

    return df



def get_county_from_zip(df: pd.DataFrame, zip_code_column: str) -> pd.DataFrame:

    import requests
    from time import sleep

    ''' 
    Esta función toma como entrada un DataFrame y el nombre de una columna de códigos postales de EE. UU.,
    y devuelve el nombre del condado de donde provienen en una nueva columna llamada "condado" dentro 
    del mismo DataFrame. Utiliza la API de Zippopotam y la API de la Comisión Federal de Comunicaciones. 
    '''

    zip_codes_list = df[zip_code_column].dropna().unique()

    lat_lon_zip = {}
    
    for zip_code in zip_codes_list:
    
        try:
            # get latitude y longitude from Zippopotam
            url = f"http://api.zippopotam.us/US/{zip_code}"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                lat = float(data['places'][0]['latitude'])
                lon = float(data['places'][0]['longitude'])
                lat_lon_zip[zip_code] = (lat, lon)
            else:
                print(f"ZIP code {zip_code} not found in API.")
        except Exception as e:
            print(f"Coordinates don't exist for the zip code: {zip_code}: {e}")
        
        sleep(0.2)

    county_zip = {}
    for key,value in lat_lon_zip.items():
            
        try:
            url_fcc = f"https://geo.fcc.gov/api/census/block/find?latitude={value[0]}&longitude={value[1]}&format=json"
            r_fcc = requests.get(url_fcc, timeout=5)
            if r_fcc.status_code == 200:
                data_fcc = r_fcc.json()
                county = data_fcc.get('County', {}).get('name')
                county_zip[key] = county.replace("County", "").strip() if county else None
            else:
                county_zip[key] = None

        except Exception as e:
            print(f"County name doesn't exist for the zip code: {key}: {e}")
            
    df["county"] = df[zip_code_column].map(county_zip)
    return df








## DATE FUNCTION

def date_converter(df: pd.DataFrame, column:str) -> pd.DataFrame:
    ''' 
    Transformar las columnas al formato de fecha y hora de Pandas 
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
    Esta función crea dos columnas con los criterios de la Oficina del Censo de EE. UU. a partir de 
    la abreviatura de estados y territorios de EE. UU. La columna 1 corresponde a la región y 
    la columna 2 a la división.
    
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
    This function groups queries by type of business operation.
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
