# Trabajo fin de Bootcamp Full Stack Big Data, AI & ML de KeepCoding

En el presente respositorio se realiza el Trabajo de Fin de Bootcamp de Full Stack en Big Data, AI & ML.
El objetivo de este proyecto es mejorar los sistemas de clasificación y predicción de precios de 
viviendas de Airbnb utilizando imágenes mosaico estructuradas, esto es, por cada vivienda se creará una imagen
en la que en cada uno de sus cuadrantes se situará la mejor foto que represente el baño de la vivienda,
la mejor foto de dormitorio, la mejor foto de cocina y la mejor foto de sala de estar. A partir de
esta imagen, perfectamente estructurada porque cada cuadrante tiene siempre el mismo tipo de foto
en todas las viviendas, se supone que los resultados mejorarán respecto a los obtenido en las distintas
prácticas que se desarrollaron en diversos módulos del Bootcamp.

Para poder realizar esta imagen mosaico es necesario previamente crear modelos que detecten distintos
tipos de estancias. Para ello, a su vez, es necesario recopilar gran cantidad de imágenes de cada
tipo de estancia que queremos detectar, ya que las necesitamos para entrenar nuestros modelos.

En el documento TrabajoFinalBootcamp-JoséÁngelArrechea.pdf, situado bajo la carpeta docs, se explica
todo el proceso llevado a cabo.

Asimimso, también se realizó un análisis de sentimiento de las reviews asociadas a cada vivienda. 
El resultado de este estudio se añadió como una características más de la vivienda, que fue utilizado
para los modelos de clasificación y predicción de precios.

Como es de suponer, se utilizó un gran número de imágenes, que en cada caso se fueron ordenando y 
nombrando de la forma más apropiada. Estas imágenes, los modelos obtenidos, los recursos utilizados y
demás archivos utilizados, que por su volumen no se encuentran en este directorio, se deben
bajar de los siguientes enlaces, que caducan el martes 18 de 2020. De cada enlace obtendremos unos o 
varios archivos .zip que deben ser descomprimidos. Cada uno de ellos se corresponde con una carpeta
vacía el repositorio. Los archivos .zip tienen un prefijo numérico que indican el orden en el que
se fueron creando/procesando cada una de las carpetas, excepto el correspondiente a los datos nlp.

https://we.tl/t-VZ53VBb17D  -> Archivos 0-resource.zip y 2-dataset.zip  
https://we.tl/t-ntg1AEDmkD -> Archivos 1-rooms.zip y 3-output.zip  
https://we.tl/t-70sYgd8pFU -> Archivos 4-models.zip y 5-rooms-fine-tuning.zip  
https://we.tl/t-vbUjQFxRJO -> Archivos 6-dataset-fine-tuning.zip, 7-models-fine-tuning.zip y 8-tiles.zip  
https://we.tl/t-n30UbCfydi -> Archivo 9-nlp.zip

Este proyecto está relacionado con el repostorio https://gitlab.com/jaarrechea/kc-fine-tuning, donde
de forma separada, se creó un modelo de detección conjunta de estancias, que finalmente se descartó,
porque sus resultados eran algo inferiores a los modelos finalmente utilizados. En este proyecto
se deben descomprimir los archivos 5-rooms-fine-tuning.zip, 6-dataset-fine-tuning.zip y 7-models-fine-tuning.zip.

Los resultados obtenidos fueron mejor de los previsto inicialmente. Hay que destacar que los modelos 
de detección de estancias, que se encuentran bajo la carpeta models, son realmente buenos. Gracias
a ellos los modelos de clasificación y predicción de precios también mejoraron.





