import joblib
import pandas as pd
import utils
import pandas
import os
from pydantic import BaseModel


# Creamos una clase para los valores de las variables:

class DatosReclamacion(BaseModel):
    Complaint_ID:int
    Product:str
    Sub_product: str
    Issue: str
    Sub_issue: str
    State: str
    ZIP_code: int
    Date_received: str
    Date_sent_to_company: str
    Company: str



from fastapi import FastAPI

app = FastAPI(
    title="Modelo predictivo de reclamaciones"
)

# Ruta absoluta al archivo .pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Obtiene la ruta donde se ejecuta el archivo .py
modelo_path = os.path.join(BASE_DIR, "pipeline_produccion.pkl") # Une la ruta anterior al nombre del archivo .pkl que contiene el pipeline con el modelo
modelo = joblib.load(modelo_path) # carga todo el modelo

@app.post("/predict")

def predecir_valores(reclamacion: DatosReclamacion):
    df = pd.DataFrame([{'Complaint ID': reclamacion.Complaint_ID , 'Product': reclamacion.Product, 
                        'Sub-product': reclamacion.Sub_product, 'Issue': reclamacion.Issue, 'Sub-issue': reclamacion.Sub_issue,
                        'State': reclamacion.State,'ZIP code': reclamacion.ZIP_code, 'Date received': reclamacion.Date_received,
                         'Date sent to company': reclamacion.Date_sent_to_company, 'Company': reclamacion.Company
    }])
    pred = modelo.predict(df)
    prob = modelo.predict_proba(df)

    df["Prediccion"] = pred
    df["Probabilidad"] = [prob[0, pred[0]]]
    df["Mensaje"] = df["Prediccion"].map({
        0: "La queja requiere compensación",
        1: "La queja no requiere compensación"
    })
    return df

@app.get("/")
def prediccion_reclamaciones():
    return {"model":"prediccion de reclamaciones"}

