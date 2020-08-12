"""
En este programa se realiza una clasificacón y regresión linea, pero incluyendo, además de las imágenes mosaico,
los valores de sentimiento, distancia a monumentos y otras características del dataset de listings.

"""
import argparse
import os
import sys
import time

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

import concurrent.futures
import imageio as io
import matplotlib.pyplot as plt

import config

from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dropout
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l1


from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Options por defecto para los pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# ---------
# Funciones
# ---------
# Creamos la siguiente función específicacon la que se descargará la imagen y
# devolverá la imagen y el índice indicando la posición donde se incrustará la
# imagen en nuestro array
def get_image(data_url, target_size=(224, 224)):
    idx, url = data_url
    try:
        img = io.imread(url)
        # hay alguna imagen en blanco y negro y daría error al incluirla en
        # nuestro array de imagenes que tiene 3 canales, así que convertimos
        # todas las imágenes que tengan menos de 3 dimensiones a color
        if img.ndim < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, dsize=target_size)
        return img, idx
    except IOError as err:
        return (None, idx)

# Función para clasificar el precio del listing en tres grupos: barato, normal y caro
def create_labels(dfx):
  y_class = []
  y_reg = dfx['price']
  for x in y_reg:
      if x <= 50: # barato
          y_class.append(0)
      elif x <=150: # normal
          y_class.append(1)
      else: # caro
          y_class.append(2)
  return y_class

def get_categorical_variables():
    return ['neighbourhood_cleansed', 'property_type'
                         , 'room_type', 'bed_type', 'cancellation_policy']

def mean_encoding_to_categorical_variables(df, mean_price_map, overall_mean_price):
    for var in get_categorical_variables():
        # Media de precio por variable categórica
        df[var] = df[var].map(mean_price_map[var])
        # Si hay una variable categórica que no existe en el mapa, asignamos precio medio
        df[var].fillna(overall_mean_price, inplace=True)

    return df

# ---------------
# Inicio programa
# ---------------

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--extract_features", type=bool, default=False,
	help="If true, extract features from images, else load features from images_sentiment_feat.npy")
args = vars(ap.parse_args())

extract_features = args['extract_features']

# Crear dataframe con
# cargamos el dataset de reviews con el análisis de sentimiento calculado
reviews = pd.read_csv(os.path.sep.join([config.NLP_PATH, config.NLP_SENTIMENT_FILE]))
print(f"Tamaño reviews {reviews.shape}")

# cargamos el dataset de listings
listings = pd.read_csv(os.path.sep.join([config.RESOURCES_PATH, config.LISTINGS_CSV_GZ]))
print(f"Tamaño listings {listings.shape}")

# Hacemos un join entre reviews y listings
listings = reviews.join(listings.set_index('id'), on='listing_id').reset_index()

print(f"Tamaño listings después de join {listings.shape}")

# Leer los nombres de los archivos imagen de tipo mosaico de cada listing
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

# Creamos un dataframe con imágenes
df_images = pd.DataFrame(data=images, columns=['id', 'filename'])
df_images = df_images.astype({"id": int})

print(f"Tamaño df_images {df_images.shape}")

listings = listings.join(df_images.set_index('id'), on='listing_id').reset_index()

print(f"Tamaño listings después de join final {listings.shape}")

# Dejar en el precio solo dígitos. Quitar $ y , (separador de miles)
listings['price'] = listings['price'].str.replace(',', '').str.replace('$','').astype(float)
# Quitamos % a host_response_rate
listings['host_response_rate'] = listings['host_response_rate'].str.replace('%', '').astype(float)
# Dejar en extra_people solo dígitos. Quitar $ y , (separador de miles)
listings['extra_people'] = listings['extra_people'].str.replace(',', '').str.replace('$','').astype(float)

# Aprovechando que a lo largo del Bootcamp hemos hecho varias veces en distintas módulos el EDA de los datos,
# vamos a quedarnos con las columnas más significativas, que previamiente ya conocemos de dichos módulos y
# que han sido comentadas en clase. En este caso no vamos a añadir la columna 'review_scores_value' porque
# conceptualmente es equivalente al vader_sentiment que hemos calculado.
listings = listings[['listing_id', 'price', 'vader_sentiment', 'host_response_rate', 'latitude', 'longitude'
                     , 'bathrooms', 'bedrooms', 'beds', 'guests_included', 'extra_people'
                     , 'minimum_nights', 'maximum_nights', 'availability_365', 'number_of_reviews'
                     , 'host_total_listings_count'
                     , 'property_type', 'room_type', 'bed_type', 'neighbourhood_cleansed', 'cancellation_policy'
                     , 'filename']]
print(f"Verificamos si hay nulos en el dataset resultante")
print(listings.isna().sum())
print(f"\n\nVemos que tenemos valores nulos en host_Response_rate, bathrooms, bedrooms, y beds. Vamos a asignales el valor 0, "
      f"ya que consideramos que si no se han indicado valor es porque su valor real es 0.")
listings['host_response_rate'].fillna(0, inplace=True)
listings['bathrooms'].fillna(0, inplace=True)
listings['bedrooms'].fillna(0, inplace=True)
listings['beds'].fillna(0, inplace=True)
print(f"Verificamos de nuevo que ya no hay nulos\n")
print(listings.isna().sum())

print(f"Examinamos los valores numéricos del dataset\n")
print(listings.describe().T)

print(f"\nPodemos observar que hay unos máximos excesivos en price y maximum_nights, que podemos considerar outliers\n")
print(f"\nVamos a eliminar las filas dichas filas\n")

print(f"Número de listings con precio superior a 3000 {len(listings[listings['price'] > 3000])}")
print(f"Número de listings con máximo número de noches superior a 3000 {len(listings[listings['maximum_nights'] > 2000])}")
print(f"\nConsideramos precios superios a 3000 erróneos y también maximo número de noches a 2000. Borramos dichas filas")
listings = listings[(listings['maximum_nights'] <= 2000)]
listings = listings[(listings['price'] <= 3000)]

listings = listings.reset_index(drop=True)
print(f"\nTamaño final del dataset de listings {listings.shape}")


if extract_features:

    # Aquí creamos nuestra estructura de datos, que va a consistir en la url de la
    # imagen y un índice para saber donde insertarla en nuestro array
    images_paths = [[i, os.sep.join([config.TILES_PATH, img_url])] for i, img_url in enumerate(listings['filename'])]

    # En este array se incrustarán las imágenes conforme se vayan obteniendo
    loaded_images = np.zeros((len(images_paths), 224, 224, 3), dtype=np.uint8)

    print('Cargamos las imágenes mosaico de cada listing en memoria')

    bol_error = False
    start = time.time()
    # Proceso de carga de imágenes. Este proceso tarda bastantes minutos.
    # Se crea un pool de procesos que descargarán las imágenes.
    # Por defecto, se crearán tantos como CPUs tenga vuestra máquina
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Se procesa la lista de urls de imágenes paralelizándola con el pool de procesos
        for (img, idx) in tqdm(executor.map(get_image, images_paths), total=len(images_paths)):
            # metemos la imagen en nuestro array
            if img is not None:
                loaded_images[idx] = img
            else:
                bol_error = True
                print(f'ERROR: No se ha podido cargar la imagen de {images_paths[idx]}')
    end = time.time()
    if bol_error:
        print(f'No se han podido cargar todas las imágenes en memoria')
        sys.exit(1)

    print(f'Imágenes cargadas en memoria. Tiempo transcurrido total en segundos  {end - start}')

    # Procedemos a extraer las características de las imágenes mosaico. Al igual que se hizo en la práctica de
    # deep learning, vamos a utilizar un modelo VGG16GlobalAveragePooling. Se presume, del mismo modo que entonces,
    # que los resultados no serán buenos porque las imágenes de VGG16 no han sido entrenadas específicamente con fotos de
    # viviendas. Sin embargo, en este caso esperamos obtener mejores resultados porque las imágenes que estamos utilizando
    # tienen, en su mayoría, una estructura común de baño, habitación, cocina y salón de estar. En la práctica se utiliza
    # la imagen de Thumbnail que era muy dispar entre los distintos listings.

    model_vgg16 = VGG16(weights='imagenet', include_top=False)

    # creamos un modelo añadiéndole la capa GlobalAveragePooling
    x = model_vgg16.output
    gap = GlobalAveragePooling2D()(x)
    model = Model(inputs=model_vgg16.input, outputs=gap)

    # ---------------------------------------------------------------
    # vamos a procesarlas y a extraer el vector de 512 características
    print (f'Procedemos a extraer las características de la imágenes vinculadas a análisis de sentimiento')
    n_images = loaded_images.shape[0]
    images_feat = np.zeros((n_images, 512))

    for i in tqdm(range(n_images)):
        img = loaded_images[i]
        aux = image.img_to_array(img)
        aux = np.expand_dims(aux, axis=0)
        aux = preprocess_input(aux)
        features = model.predict(aux)
        images_feat[i] = features[0]

    # Para evitar repetir la extracción de features, las guardamos en un archivo
    np.save(os.sep.join([config.TILES_PATH, config.IMAGE_SENTIMENT_FEATURES_FILE]) , images_feat)
else:
    images_feat = np.load(os.sep.join([config.TILES_PATH, config.IMAGE_SENTIMENT_FEATURES_FILE]))

print(f"Tamaño del array de característcias de imágenes {images_feat.shape}")

# Convertir el array de features de las imágenes en un dataframe
df_image_feat = pd.DataFrame(images_feat)
print(f"Convertimos el array numpy de característcias de imágenes a un dataframe  y verificamente el tamaño {df_image_feat.shape}")
print(f"Primeras filas del dataframe de características de imágenes\n{df_image_feat.head()}")

print(f"\nEliminamos columnas del dataframe de listings que ya no nos hacen falta: listing_id y filename")
listings.drop(['listing_id'], axis=1, inplace=True)
listings.drop(['filename'], axis=1, inplace=True)
print(f"\nVerificamos las columnas\n{listings.columns}")
print(f"\nNuevo tamaño {listings.shape}")

print(f"\nUnimos el dataset de atributos de los listing con el de las características de imágenes")
full_data = pd.concat((listings, df_image_feat), axis=1)
print(f"\nTamaño final del dataframe consolidado: {full_data.shape}")

# --------------------------------
# Training, validation and testing
# --------------------------------

# Separamos train y test
print("Separamos train y test 80-20")
train, test = train_test_split(full_data, test_size=0.2, shuffle=True, random_state=0)
print(f"\nTamaño de train {train.shape}")
print(f"Tamaño de test {test.shape}")

# Calculamos precios medios por cada categoría dentro de cada variable categórica
# Almacenamos los precios medios para su uso posterior
mean_price_map = {}
overall_mean_price = 0
for var in get_categorical_variables():
    mean_price_map[var] = train.groupby(var)['price'].mean()

# Guardamos el precio medio general por si en test hay variables categóricas que no están en train
# En este tipo de casos es bastante común hacer medias de medias, pero en el nuestro, hemos preferido utilizar el precio medio
# general, porque no hay mucha diferencia.
overall_mean_price = train['price'].mean()

# Asignamos a train y test los valores medios calculados. De esta forma solo tenemos valores numéricos en el dataset
train = mean_encoding_to_categorical_variables(train, mean_price_map, overall_mean_price)
test = mean_encoding_to_categorical_variables(test, mean_price_map, overall_mean_price)
# print(f'\n\nComprobamos que todas los atributos de train son numéricos\n{train.dtypes}')
# print(f'\nComprobamos que todas los atributos de test son numéricos\n{test.dtypes}')


# Creamos las variables objetivo de tipo clase de train y test
y_train_class = create_labels(train)
y_test_class = create_labels(test)

print("\n\n-------------")
print("Clasificación")
print("-------------\n\n")

# Utilizamos los datos del dataframe, menos el precio, que está en la primera columna
data = train.values
X_train = data[:,1:]

# Normalizamos los datos de train (valores entre 0 y 1).
min_max_scaler = preprocessing.MinMaxScaler()
XtrainScaled = min_max_scaler.fit_transform(X_train)

# Aplicamos OneHot a la variable objetivo
y_training_class_onehot = to_categorical(y_train_class)

# Para crear el modelo debemos dividir el dataset de train, en train y validation
# Como ya sabemos por los procesos anteriores, los precios no están correctamente
# distribuidos. Por esa razón, vamos a estratificar para que los dataset
# obtenidos para los modelos estén balanceados en la medida de lo posible
X_train, X_val, y_train, y_val = train_test_split(XtrainScaled, y_training_class_onehot
                  , test_size=0.1, random_state=0, stratify=y_training_class_onehot)

# Creamos modelo de clasificación con tres capas densas
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))  # Asignamos el valor 3, porque tenemos 3 clases

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

print(f"\nModelo simple de red neuronal de tres capas")
model.summary()

history = model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=50,
          batch_size=8)

# A continuación debemos escalar test del mismo modo que train
data_test = test.values
X_test = data_test[:,1:]  # nos quedamos con todas las columnas menos el precio
# Escalamos test
XtestScaled = min_max_scaler.fit_transform(X_test)
# Transformamos a OneHot la clase de test
y_test_class_onehot = to_categorical(y_test_class)
loss, acc = model.evaluate(XtestScaled, y_test_class_onehot)
print(f"\nLa pérdida y el accuracy que se obtuvieron con imágenes no estructuradas fueron; Loss=8.13186645873205, Acc=0.5718390941619873")
print(f'\nLa pérdida y el accuracy obtenido ahora son: Loss={loss}, Acc={acc}')
print(f'\nComo se puede observar, del mismo modo que en el caso del estudio de solo con imágenes, el accuracy mejora en más de un 10%')
print(f'Se reafirma que las imágenes estructuradas (imagen mosaico) mejoran los resultados.')
print(f'También se debe tener en cuenta que se ha añadido el dato del VADER Sentiment que puede influir algo')

# Creamos modelo de clasificación con tres capas densas con regularización L1 y
# dos Dropout
print(f"\n\nCreamos modelo de clasificación con tres capas densas con regularización L1 y dos Dropout")
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=l1(0.003)))
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu', kernel_regularizer=l1(0.01)))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))  # Asignamos el valor 3, porque tenemos 3 clases

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=50,
          batch_size=8)

loss, acc = model.evaluate(XtestScaled, y_test_class_onehot)
print(f"\nLa pérdida y el accuracy que se obtuvieron con imágenes no estructuradas fueron; Loss=1.5043120713069522, Acc=0.5086206793785095")
print(f'\nLa pérdida y el accuracy obtenido ahora son: Loss={loss}, Acc={acc}')
print(f'\nComo se puede observar mejora los datos obtenidos de la anterior red, más simple.')

print("\n\n-------------")
print("Regresión")
print("-------------\n\n")

# Repetimos operaciones, como la de escalado, realizadas para la clasificación.
# Se podría evitar pero facilita el hecho de poder seleccionar esta parte del código y poderlo copiar/mover
# a otros programas/módulos
min_max_scaler = preprocessing.MinMaxScaler()

data = train.values
data_scaled = min_max_scaler.fit_transform(data)
y_scaled_train = data_scaled[:,0:1]     # En la 1ª columna está Price, nuestra variable objetivo
X_scaled_train = data_scaled[:,1:] # Resto de columnas

# Realizamos la separación train/validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled_train, y_scaled_train
                  , test_size=0.1, random_state=0)

# Creación de una primera red neural simple para la regresión
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))  # En este caso solo es una neurona porque solo tenemos una salida: directamente el precio

# Utilizamos funciones loss y optimizer acordes con la regresión lineal
model.compile(loss='mean_squared_error', optimizer='RMSProp')

print(f"\nModelo simple de red neuronal de tres capas")
model.summary()

history = model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=50,
          batch_size=8)

# plt.title('Loss / Mean Squared Error')
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='val')
# plt.legend()
# print(plt.show())

# Normalizamos los datos de test (valores entre 0 y 1).
min_max_scaler = preprocessing.MinMaxScaler()
data_test = test.values
data_test_scaled = min_max_scaler.fit_transform(data_test)
y_scaled_test = data_test_scaled[:,0:1] # En la 1ª columna está Price, nuestra variable objetivo
X_scaled_test = data_test_scaled[:,1:] # Resto de columnas
loss = model.evaluate(X_scaled_test, y_scaled_test)
print(f"La pérdida obtenida con la misma red sin imágenes estructuradas fue Loss=0.037447664230148454")
print(f'La pérdida obtenida actual obtenida con los datos de test es: Loss={loss}')
print(f'Como se puede observar la pérdida con las imágenes estructuradas en mosaico es bastante menor, por lo '
      f'que queda confirmado una vez más, que las redes mejoran notablemente con información estructurada.')

print(f'\n\nCONCLUSIONES')
print(f'------------')
print(f'\n\nLa VGG16 utilizada para la extracción de características de las imágenes de Tumbnail de las viviendas\n'
      f'no son correctas o no proprocionan la suficiente información para poder realizar tareas de clasificación\n'
      f'o regresión sobre los precios. Sin embargo, si se crean modelos para la clasificación correcta de tipos de\n'
      f'estancias y esta información se aplica de forma estructurada, los sistemas mejoran notablemente. En una red\n\n'
      f'de clasificación se ha obtenido un accuracy de uno 77%, lo cual permitiría, en algunos entornos, utilizar el\n '
      f'modelo como sistema de predicción.\n\n'
      f'Llama la atención lo precisos que son los modelos de clasificación de estancias utilizando ResNet50 a pesar de que imagenet nunca\n'
      f'ha realizado un estudio sobre este tipo de imágenes. Se concluye que los pesos que proporciona imagenet se pueden\n'
      f'utilizar para casi cualquier tipo de clasificación de imágenes.\n\n'
      f'Se podrían hacer más pruebas clasificación de estancias utilizando otras soluciones diferentes a ResNet50, como\n'
      f'Xception o MobileNet. Es muy probable que con estas soluciones se consiga mejorar algo, a pesar de que ResNet50\n'
      f'proporciona valores muy buenos.'
      )
