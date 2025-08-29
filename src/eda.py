import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualization_features(df: pd.DataFrame, *args: str,  bins=20):
    ''' 
    Creación de gráficos de barras para características 
    categóricas e histogramas para características numéricas
    '''
    
    try:
          for col in args:
                if df[col].dtype.name in ["object", "category"]:
                    counts = df[col].value_counts().sort_values(ascending=False)
                    ax = counts.plot(kind="bar", figsize=(6, 4), title=col)
                    ax.set_xlabel("")
                    ax.bar_label(ax.containers[0])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                    plt.tight_layout()
                    plt.show()
                elif df[col].dtype.name in ["int64", "float64"]:
                    df[col].hist(bins=bins, figsize=(6,4), title=col)
                    plt.xlabel("")
                    plt.tight_layout()
                    plt.show()
                else:
                    print(f"Columna {col} no es categórica ni numérica.")

          
    except Exception as e:
            print(f"{col} no se pudo graficar. Error: {e}")
    


def missing_values(df: pd.DataFrame, *args) -> pd.DataFrame:
    ''' 
    Proporciona un marco de datos con características del marco 
    de datos de entrada y sus valores faltantes    '''
    missing_values_dict = {}
    for col in args:
        missing_values_dict[col] = df[col].isnull().sum()
    
    result_df = pd.DataFrame(list(missing_values_dict.items()), columns=['feature', 'missing_values'])
    return result_df
