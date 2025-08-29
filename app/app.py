import utils
import joblib
import gradio as gr
import pandas as pd



pipeline = joblib.load("pipeline_produccion.pkl")


# Predicción desde CSV

def predecir_csv(ruta_csv):
    df = pd.read_csv(ruta_csv.name)  
    predicciones = pipeline.predict(df)
    probabilidad = pipeline.predict_proba(df)

    prob_predicha = []
    for i, pred in enumerate(predicciones):
        prob_predicha.append(probabilidad[i, pred])
        
    df["Prediccion"] = predicciones
    df["Probabilidad"] = prob_predicha
    df["Mensaje"] = df["Prediccion"].map({
        0: "La queja requiere compensación",
        1: "La queja no requiere compensación"
    })
    return df

# Predicción manual

def predecir_valores(Complaint_ID, Product, Sub_product, Issue, Sub_issue, State,
                     ZIP_code, Date_received, Date_sent_to_company, Company):
    
    df = pd.DataFrame([{'Complaint ID': Complaint_ID , 'Product':Product, 
                        'Sub-product': Sub_product, 'Issue':Issue, 'Sub-issue':Sub_issue,
                        'State': State,'ZIP code':ZIP_code, 'Date received':Date_received,
                         'Date sent to company':Date_sent_to_company, 'Company':Company
    }])
    
    pred = pipeline.predict(df)
    prob = pipeline.predict_proba(df)

    df["Prediccion"] = pred
    df["Probabilidad"] = [prob[0, pred[0]]]
    df["Mensaje"] = df["Prediccion"].map({
        0: "La queja requiere compensación",
        1: "La queja no requiere compensación"
    })
    return df


# Interfaz Gradio

with gr.Blocks() as demo:
    gr.Markdown("# Modelo predictivo de atención a reclamaciones")

    with gr.Tab("Subir CSV"):
        csv_input = gr.File(type="file", file_types=[".csv"], label="Sube un CSV con las reclamaciones")
        csv_output = gr.Dataframe(label="Predicciones")
        btn1 = gr.Button("Predecir desde CSV")
        btn1.click(fn=predecir_csv, inputs=csv_input, outputs=csv_output)

    with gr.Tab("Introducir valores manuales"):
        gr.Markdown("Introduce los valores de la queja (ajusta a tus variables reales).")

        # 🔹 Ajusta estos campos según las columnas que usa tu pipeline

        Complaint_ID = gr.Number(label="Reclamación ID")
        Product = gr.Dropdown(choices=['Debt collection', 'Mortgage', 'Credit reporting', 'Credit card',
                                       'Bank account or service', 'Consumer loan', 'Student loan', 
                                       'Payday loan', 'Money transfers', 'Prepaid card'], label="Producto")
        Sub_product = gr.Textbox(label="Sub-Producto")
        Issue = gr.Textbox(label="Problema")
        Sub_issue = gr.Textbox(label="Sub-Problema")
        State = gr.Dropdown(choices=['TX', 'KS', 'NY', 'CT', 'MS', 'UT', 'FL', 'AZ', 'VA', 'NV',
                                     'MD', 'CA', 'OR', 'DE', 'IL', 'LA', 'NJ', 'SC', 'OH', 'ID', 'WA',
                                     'MO', 'ME', 'PA', 'AL', 'IA', 'PR', 'CO', 'KY', 'IN', 'MN', 'NE',
                                     'GA', 'NC', 'MA', 'WI', 'MI', 'AR', 'OK', 'TN', 'NH', 'WV', 'SD',
                                     'AK', 'DC', 'NM', 'WY', 'RI', 'VT', 'HI', 'MH', 'VI', 'AP', 'MT',
                                     'AS', 'ND', 'GU', 'AE', 'PW'], label="Estado")
        ZIP_code = gr.Number(label="Código Postal")
        Date_received = gr.DatePicker(label="fecha de recepción de la reclamación")
        Date_sent_to_company = gr.DatePicker(label="fecha de envío de la reclamación a la empresa")
        Company = gr.Textbox(label="Nombre de la compañía")

        btn2 = gr.Button("Predecir desde valores")
        val_output = gr.Dataframe(label="Predicción manual")

        btn2.click(fn=predecir_valores, inputs=[Complaint_ID, Product, Sub_product, Issue, 
                                                Sub_issue, State,ZIP_code, Date_received,
                                                  Date_sent_to_company, Company], outputs=val_output)

demo.launch()
