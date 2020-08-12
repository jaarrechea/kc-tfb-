"""

En este programa se va a realizar el análisis de sentimiento de las distintas reviews que se los usuarios hacen
a las viviendas (listings) de airbnb. Para ello utilizaremos el archivo reviews.csv.gz, que se encuentra bajo
la carpeta recursos. En este archivo un mismo listing puede tener uno o más reviews.

Las reviews están escritas en distintos idiomas. Para detectar el idioma utilizamos el paquete langdetect, que debe
estar previamente instalado.

Seguidamente, únicamente seleccionaremos reviews en inglés, a los cuales se les aplicará un análisis de sentimiento con
VADER. Al sentimiento positivo se le asignará el valor 1, al neutro 0.5 y al negativo 0.

Finalmente, se obtendrá un dataset en el que a cada listing se le calculará la media del valor del sentimiento calculado
en el paso anterior.

Modo de uso
-----------
python nlp-sentiment-analysis.py  --detect_language True

Si el parámetro detect_language es True, el programa cargará el dataset de reviews y seleccionará lo que tengan imagen
mosaico. A continuación, detecterá el idioma de cada uno de los comentarios. El resultado se guardará en un archivo
que podrá ser procesado en siguientes ocasiones.

Si el parámetro detect_language es False o no se indica, el programa comenzará cargando el dataset que contiene el
idioma de cada comentario, que previamente se ha tenido que crear en una ejecución anterior de este mismo programa con
el parámetro detect_language = True.

El programa da la posibilidad de crear el archivo intermedio de reviews con el idioma, para poder realizar varias
ejecuciones posteriores, sin tener que volver a detectar el idioma de cada comentario, ya que es un proceso muy costoso.

Con el dataset de reviews y sus idiomas, se procede a realizar el análisis de sentimiento de cada listing, tal y como
se ha explicado anteriormente.

"""
import argparse
import os
import time

import pandas as pd
import numpy as np
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm

import config

# Options por defecto para los pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# ---------
# Funciones
# ---------

def language_detect(row):
    """
    Función que detecta el idioma en el que está escrito un text.

    :param row: listing/viveinda del dataset cuyos comentarios queremos saber en qué idioma están
    :return: código del idioma del texto o xx si no se ha podido obtener
    """

    text = row['comments']
    if text == None or type(text) is float or len(text) < 5:
        return 'xx'
    try:
        return detect(text)
    except:
        print(f"Error in Listing_id {row['listing_id']}. Id {row['id']}. Text {text}")
        return 'xx'


def vader_sentimental_score(sentence):
    """
    Función para calcular el sentimiento que produce una determinada frase según VADER. devuelve un valor numérico
    normalizado entre 0 y 1 para que pueda ser analizar posteriormente en procesos de testeo y training.

    :param sentence: texto cuyo nivel de sentimiento queremos analizar
    :return: Valor numérico: 1: sentimiento positivo; 0.5: sentimiento neutro; 0: sentimiento negativo
    """
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    score=vs['compound']
    if score >= 0.5:
        return 1
    elif (score > -0.5) and (score < 0.5):
        return 0.5
    elif score <= -0.5:
        return 0

# ---------------
# Inicio programa
# ---------------

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detect_language", type=bool, default=False,
	help="If true, detect language of reviewsIf true, extract features from images, else load rewiews with language from features from images_feat.npy")
args = vars(ap.parse_args())

detect_language = args['detect_language']


if (detect_language == True):
    # cargamos el dataset de reviews, seleccionamos aquellos que tienen imagen mosaico de la vivienda y
    # detectamos el idioma de los comentarios
    reviews = pd.read_csv(os.path.sep.join([config.RESOURCES_PATH, config.REVIEWS_CSV_GZ]))

    print(f'Tamaño del dataset de reviews {reviews.shape}')
    print(f'\nColumas del dataset de reviews {reviews.columns}')
    print(f'\nPrimeros registros del dataset de reviews\n{reviews.head()}')

    # Leemos los archivos de las imágenes mosaico para quedarnos solo con las reviews que tengan dicha imagen
    tiles_folders = os.listdir(config.TILES_PATH)
    images = []
    for folder in tiles_folders:
        if (not os.path.isdir(os.path.sep.join([config.TILES_PATH, folder]))) or folder.startswith('.'): continue
        files = os.listdir(os.sep.join([config.TILES_PATH, folder]))
        for file in files:
            if file.startswith('.'): continue
            name = file.strip().split('.')[0]
            data_image = []
            data_image.append(int(name))
            data_image.append(os.sep.join([folder, file]))
            images.append(data_image)

    images = np.array(images)
    reviews = reviews[reviews.listing_id.isin(images[:,0])]

    print(f'\n\nEl tamaño del dataset de reviews seleccionando solo aquellos que tengan imagen mosaico es {reviews.shape}')

    print(f'\n\nDetectando el idioma de los comentarios de los reviews...')
    start = time.time()
    reviews['language'] = reviews.apply(lambda row: language_detect(row), axis=1)
    end = time.time()

    print(f'\nTiempo en segundos utilizado para calcular idioma de los reviews {end-start}')

    print(f'\nPrimeros registros del dataset de reviews {reviews.head()}')

    print(f"Valores diferentes de idiomas\n {reviews['language'].value_counts()}")

    reviews.to_csv(os.path.sep.join([config.NLP_PATH, config.NLP_LANG_FILE]), index=False)
else:
    # Cargamos el dataset de reviews con el idioma asignado, que previamente se ha procesado y guardado
    reviews = pd.read_csv(os.path.sep.join([config.NLP_PATH, config.NLP_LANG_FILE]))

# ----------------------------------------------------
# Análisis de sentimiento de los comentarios en inglés
# ----------------------------------------------------

# Nos quedamos solo con las reviews en inglés, que aproximadamente es casi la mitad del dataset
reviews = reviews.drop(reviews[reviews['language'] != 'en'].index)
print(f"\n\nSeleccionamos solo los reviews en inglés")
print(reviews['language'].value_counts())

print(f"\n\nAnalizando sentimiento VADER de los comentarios de los reviews en inglés...")

tqdm.pandas()

start = time.time()
reviews['vader_sentiment'] = reviews.progress_apply(lambda row: vader_sentimental_score(row['comments']), axis=1)
end = time.time()
print(f'\nTiempo en segundos utilizado para el análisis de sentimiento de los reviews {end - start}')

# Guardarmos el dataset para posteriores
reviews.to_csv(os.path.sep.join([config.NLP_PATH, config.NLP_LANG_AND_SENTIMENT_FILE]), index=False)

# ---------------------------------------------------------------
# Calculamos la media de sentimiento para cada uno de los listing
# ---------------------------------------------------------------
print(f"\n\nCreamos un nuevo dataset con la media del sentimiento vader por cada listing")
reviews = reviews.groupby('listing_id', as_index=False)['vader_sentiment'].mean()
reviews.to_csv(os.path.sep.join([config.NLP_PATH, config.NLP_SENTIMENT_FILE]), index=False)
print(f"Dataset resultante\n {reviews.head()}")
print(f"Tamaño {reviews.shape}")
