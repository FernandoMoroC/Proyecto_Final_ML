# DESARROLLO TÉCNICO

---

## CONTEXTUALIZACION TECNICA

Todas las reclamaciones son importantes, sin embargo aquellas que requieren de algún tipo de acción correctiva por parte de la empresa deben ser prioritarias ya que al contrario de las otras, éstas requieren de modificaciones y/o rectificaciones que reclaman mayor tiempo de trabajo que simplemente una explicación. El 2,5% de las reclamaciones no fueron atendidas a tiempo por las empresas, y de ese porcentaje, el 12% correspondia a reclamaciones que necesitaron algún tipo de acción correctiva.

Dado este contexto se propuso construir un modelo de *machine learning* capaz de clasificar si las reclamaciones necesitarán o no de acciones correctivas con el fin de darles prioridad y reducir el porcentaje de reclamaciones solucionadas fuera de tiempo.

---

# METODOLOGIA

- Separación de un 20% de los datos para validación.

- El conjunto de datos proporcionado presentaba la mayoría de variables de tipo  objeto y algunas con una alta cardinalidad:

![Variables_nullvalue_cardinality](/Images/variables_valores_nulos_unicos.png)

Con baja asociación entre las variables:

![Asociacion_entre_variables](/Images/asociacion_variables.png)

Las respuestas de las empresas a las reclamaciones de los clientes se clasifican en: 
- *Closed with explanation* -> Cerrada con una explicación
- *Closed* -> Cerrada
- *Closed with monetary relief* -> Cerrada con compensación monetaria
- *Closed with non-monetary relief* -> Cerrada con compesación no monetaria

- La variable asociada a las respuestas dadas por las empresas se binarizó en respuestas sin compensación y respuestas con compensación

![Proporcion de respuestas](/Images/proporcion_de_los_tipos_de_respuesta.png)


![Proporcion de respuestas a destiempo](/Images/proporcion_de_los_tipos_de_respuesta_a_destiempo.png)


- Los valores faltantes de la variable *Sub-product* se imputaron con el valor "No Subproduct" y se codificó con *target encoder*.

- Los valores de las variables con las fechas de registro de la reclamación y envío a la empresa se desglosaron en día del mes, día de la semana y si era fin de semana o no.

- La variable *Product* dado su baja cardinalidad se codificó mediante la técnica de *OneHotencoding*.

- La variable *State* se recategorizó en valores de regiones y divisiones territoriales propuestas por *U.S. Census Bureau* y se codificó con *OneHotEncoding*.

- La variable *Issue* se codificó como embeddings construidos a partir del modelo Word2Vec y se codificó cada embeddingo con la técnica *OneHotEncoding*.

- La variable *Company* se volvió a utilizar la información de la columna *Product* y se reclasificó como actividad empresarial en función del producto implicado en la reclamación.

---
## MODELADO, RESULTADOS Y METRICAS DE EVALUACION

- Los datos fueron entrenados con modelos de regresión logística, árboles de decisión, máquinas de vectores y un modelo secuencial de red neuronal. Además se utilizaron diferentes ensembles como voting, randomforest, AdaBoost y XGboost. 

- Todos los modelos se entrenaron teniendo en cuenta el desbalance entre la clases y utilizando diferentes combinaciones de hiperparámetros con el fin de explorar lo máximo posible el espacio de soluciones.

- Se seleccionó el modelo con el f1-score más alto. Es decir aquel modelo en el que la precisión y sensibilidad sobre la clase menos representada estuviesen equilibradas. Además también se utilizó la métrica AUC para seleccionar el modelo con el mayor área bajo la curva

- El mejor modelo según estos criterios fue un modelo de regresión logística con regularización de Ridge, con una precisión para la clase que necesita compensación de 0.28, sensibilidad de 0.75, f1-score de 0.41 y un AUC de 0.66.
---

## DISCUSION SOBRE LIMITACIONES Y MEJORAS

- La principal limitación del modelo es que no es capaz de separar con alta precisión y sensibilidad ambas clases. Esto es debido al desbalance que existe entre los casos de respuesta por parte de las empresas. 

- El feature enginnering realizado en el proyecto hasta ahora puede ser mejorado incluyendo variables o recodificando las ya existentes de una forma que ayuden a clasificar mejor la respuesta.

- Una búsqueda má amplia de hiperparámetros, especialmente en las técnicas de ensembles con votación, puede contribuir a que los errores cometidos por los diferentes algoritmos sean compensados y mejore la clasificación.

---

# DEMOSTRACION PRACTICA

[Modelo Predictivo](https://huggingface.co/spaces/FerMC/Modelo_clasificacion_reclamaciones)
