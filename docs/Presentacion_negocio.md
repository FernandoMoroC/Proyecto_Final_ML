# INFORME SOBRE EL DESARROLLO DEL MODELO PREDICTIVO

---

## 1. DESCRIPCION DEL PROBLEMA

Un 2,5% de las reclamaciones que recibimos no son gestionadas dentro del marco reglamentario de los 15 días. Además observamos que las reclamaciones de los clientes que fueron disputadas posteriormente a la respuesta parecen mostrar cierta asociación con aquellas reclamaciones que tuvieron un mayor retraso entre que se recibió y se envió a la empresa.

![Respuestas dadas a tiempo o no](/Images/respuestas_dadas_a_tiempo.png)

![Respuestas disputadas](/Images/respuestas_disputadas.png)


Dentro de las reclamaciones recibidas, aquellas que requieren de algún tipo de compensación, monetaria o no, merecen prioridad, ya que implican un mayor número de gestiones por parte de las empresas para solventar el problema. Además, no responder a tiempo a este tipo de reclamaciones puede generar un perjuicio mayor sobre los clientes que si la reclamación no requiere compensación.

---

# 2. VALOR DEL MODELO

El modelo de inteligencia artificial desarrollado es capaz de predecir si una nueva reclamación necesitará de algún tipo de compensación o no. Esto nos permitirá priorizar aquellas que sí la requieran y mejorar la satisfacción del cliente. 

![Proporcion de respuestas](/Images/proporcion_de_los_tipos_de_respuesta.png)


![Proporcion de respuestas a destiempo](/Images/proporcion_de_los_tipos_de_respuesta_a_destiempo.png)

---

# BENEFICIOS Y APLICACIONES PRÁCTICAS

Beneficios potenciales del uso del modelo:
- Mejora en la enficiencia de tratamiento de las reclamaciones de los clientes, dando prioridad a aquellas que son más complejas.
- Las empresas solucionarán antes las reclamaciones más graves, mejorando la valoración de los clientes y disminuyendo el número de disputas post-respuesta.
- Mejora en la eficiencia del tiempo empleado por las empresas para el análisis de la queja y su clasificación.

Uso práctico e implementación:

1. Las reclamaciones son almacendas y descargadas en un archivo csv.

2. El archivo descargado es subido a la plataforma del modelo.
3. El archivo es procesado por el modelo y devuelve el mismo archivo pero con la información de como se clasifica la reclamación y la probabilidad de esa clasificación.

---

# EJEMPLO DE USO DEL MODELO

![Resumen de Funcionamiento](/Images/funcionamiento_modelo_huggingFace.png)


