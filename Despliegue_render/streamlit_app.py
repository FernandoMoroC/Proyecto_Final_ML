import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import utils

st.title('Modelo de predicción de compensaciones')

st.write("Sube tu csv de reclamaciones")


# Ruta absoluta al archivo .pkl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(BASE_DIR, "pipeline_produccion.pkl")

def carga_modelo(ruta_modelo):
    if not os.path.exists(ruta_modelo):
        raise FileNotFoundError(f"Model file not found: {ruta_modelo}")
    return joblib.load(ruta_modelo)

modelo = carga_modelo(modelo_path)


    

# -------------------------------
# SUBIR CSV
# -------------------------------
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Leer CSV
        df = pd.read_csv(uploaded_file)
         
        predicciones = modelo.predict(df)
        probabilidad = modelo.predict_proba(df)

        prob_predicha = []
        for i, pred in enumerate(predicciones):
            prob_predicha.append(probabilidad[i, pred])
        
        df["Prediccion"] = predicciones
        df["Probabilidad"] = prob_predicha
        df["Mensaje"] = df["Prediccion"].map({
            0: "La queja requiere compensación",
            1: "La queja no requiere compensación"
            })

        # Mostrar resultados
        st.subheader("Resultados con Predicciones")
        st.dataframe(df)

        # Descargar CSV con resultados
        csv_result = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar CSV con predicciones",
            data=csv_result,
            file_name='predicciones.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")
else:
    st.info("Por favor, sube un archivo CSV para obtener predicciones.")



