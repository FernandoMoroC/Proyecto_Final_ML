# MEMORIA DE PROYECTO

## INTRODUCCION Y OBJETIVO

Las quejas recogidas por los servicios  de protección al consumidor de EEUU (*Consumer Financial Protection Bureau (CFPB)*) muestran cómo las respuestas que deben dar las empresas implicadas pueden clasificarse como respuestas que requieren algún tipo de acción, monetaria o no, y respuestas que no requieren acciones por parte de las empresas. Éstas últimas pueden cerrarse con o sin explicación al consumidor. 
La mayoría de quejas son cerradas sin ningún tipo de acción por parte de la empresa y dentro de un periodo de 15 días posterior al registro de la queja. Sin embargo, existe aproximadamente un 2% de quejas que no son atendidas a tiempo dentro de dicho periodo de 15 días y a veces la solución proporcionada por las empresas es reclamada por parte del consumidor.

Aunque es conveniente que todas las reclamaciones sean atendidas dentro del periodo estipulado, aquellas que requieren algún tipo de acción correctiva por parte de las empresas deben ser priorizadas sobre las que no requeren acción. Es por esta razón que se establece como objetivo la creación de un modelo de *machine learning* capaz de clasificar las reclamaciones o quejas por parte de los clientes en las categorías de si requerirán acciones correctivas o no.

## METODOLOGIA

### EDA Y FEATURE ENGINEERING

A partir de un set de datos que contenía un alto número de reclamaciones con los valores asociados para diferentes variables se realizó un análisis exploratorio con el fin de estudiar la información, estructura y comportamiento de los datos. En este proceso se decidió binarizar la variable asociada a las respuestas dadas por las empresas, ya que el mayor porcentaje eran para el valor cerrado con explicación, y los demás valores estaban presentes en un porcentaje muy pequeño. La binarización se hizo en respuestas sin compensación y respuestas con compensación. 
Las columnas con variables  que no aportaban información, que presentaban alta cardinalidad, alta cantidad de datos faltantes o la imposibilidad de realizar feature enginnering con los valores que contenían, fueron eliminadas.
Como la mayoría de columnas eran datos nominales, se trataron con técnicas de imputación de datos faltantes, *OneHotEncoding*, *TargetEncoding*, creación de embeddings y recategorización.


### MODELADO DE DATOS

Los datos fueron entrenados con modelos de regresión logística, árboles de decisión, máquinas de vectores y un modelo secuencial de red neuronal. Además se utilizaron diferentes ensembles como voting, randomforest, AdaBoost y XGboost. Todos los modelos se entrenaron teniendo en cuenta el desbalance entre la clases. Además todos los modelos y ensembles se entrenaron utilizando diferentes combinaciones de hiperparámetros con el fin de explorar lo máximo posible el espacio de soluciones.

## RESULTADOS Y PROTOTIPADO

El desbalance entre las clases impidió que la precisión para detectar la clase menos representada alcanzase valores altos. Como el objetivo del proyecto era crear un modelo capaz de clasificar si la reclamación necesitaría de compensación o no, siendo las primeras las menos representadas, se decidió seleccionar el modelo con el f1-score más alto. Es decir aquel modelo en el que la precisión y sensibilidad sobre la clase menos representada estuviesen equilibradas. Esto hizo disminuir la sensibilidad para detectar la clase más abundante de reclamaciones que no necesitan compensación, sin embargo aunque cierto número de reclamaciones que no necesiten acciones compensación sean clasificadas como tal no es un inconveniente, ya que lo importante es que se detecten un mayor número de reclamaciones que sí las necesitan para poder darle prioridad. El mejor modelo según estos criterios fue un modelo de regresión logística con regularización de Ridge.

## CONCLUSIONES

El modelo obtenido es capaz de indicar si las reclamaciones de clientes necesitarán de acciones correctivas o no en un contexto similar al de conjunto de datos utilizado para el entrenamiento.




