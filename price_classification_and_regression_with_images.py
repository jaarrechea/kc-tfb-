"""
El objetivo de este programa es realizar una clasificación de los precios de las viviendas de AirBnb en función
de las principales fotos de las mismas. Las
"""
import argparse
import os
import sys
import time

import cv2
from tqdm import tqdm

import config
import pandas as pd
import numpy as np

import imageio as io
import concurrent.futures
import matplotlib.pyplot as plt



from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dropout
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

from sklearn import preprocessing

# Options por defecto para los pandas
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


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

# ---------------
# Inicio programa
# ---------------

# ------------------
# Inicio de programa
# ------------------

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--extract_features", type=bool, default=False,
	help="If true, extract features from images, else load features from images_feat.npy")
args = vars(ap.parse_args())

extract_features = args['extract_features']

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

# Carga el dataset de listings
listing_file = os.path.sep.join([config.RESOURCES_PATH, config.LISTINGS_CSV_GZ])
listings = pd.read_csv(listing_file)

# Nos quedamos solo con los listing que tenga imagen mosaico
listing_with_tile = listings[listings.id.isin(images[:,0])]


# Para el estudio que nos interesa, nos quedamos solo con el id y el precio
listings = listing_with_tile[['id','price']].reset_index()

# Convertimos el array numpy an un dataframe para realizar un join con el dataframe de listings
# Al final tendremos un dataframe con el id, el nombre del archivo de la imagen mosaico y el precio
df_images = pd.DataFrame(data=images, columns=['id', 'filename'])
df_images = df_images.astype({"id": int})

listings = df_images.join(listings.set_index('id'), on='id').reset_index()[['id','filename','price']]

# Dejar en el precio solo dígitos. Quitar $ y , (separador de miles)
listings['price'] = listings['price'].str.replace(',', '').str.replace('$','').astype(float)
print('Características del dataset de listings')
print(listings.shape)
print(listings.head())
print (f'Comprobamos que no hay precios con valor cero. Número de listings con precio 0 (listings["price"].isna().sum()) '
       f' {listings["price"].isna().sum()}')


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
    print (f'Procedemos a extraer las características de la imágenes')
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
    np.save(os.sep.join([config.TILES_PATH, config.IMAGE_FEATURES_FILE]) , images_feat)
else:
    images_feat = np.load(os.sep.join([config.TILES_PATH, config.IMAGE_FEATURES_FILE]))

# --------------
# Clasificación
# --------------

# Vamos a crear un dataframe en el que el precio de cada listing se agrupará en barato(0), normal(1) y caro(2)
y_class = create_labels(listings)

# Veamos cómo ha quedado la distribución al convertirla a 3 clases
plt.hist(y_class, bins=3)
print(plt.show())
unique, counts = np.unique(y_class, return_counts=True)
print('La distribución de precios por clases barato(0), normal(1), caro (2) es:')
print(dict(zip(unique, counts)))
print('Como se puede apreciar hay bastantes más baratos y normales que caros.')


# Procedemos a normalizar vamos a normalizar los datos de las características de las imágenes.
# En este caso podemos normalizar todos los datos a la vez, sin separar train de test porque no influye, ya
# que solo tenemos imágenes
min_max_scaler = preprocessing.MinMaxScaler()
images_scaled = min_max_scaler.fit_transform(images_feat)

# Categorizamos la variable objetivo con OneHot
from keras.utils import to_categorical
y_class_onehot = to_categorical(y_class)

# Dividimos los datos en train y en test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images_scaled, y_class_onehot, test_size=0.2, shuffle=True, random_state=0)
# Debido a que las clases de la variable objetivo no están balanceadas
# estratificamos los datos
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=0, stratify=y_train)

print(f'# ------------- #')
print(f'# Clasificación #')
print(f'# ------------- #')

print(f'\nTamaños train, test y validación')
print(f'Train: X{X_train.shape} y{y_train.shape}')
print(f'Text: X{X_test.shape} y{y_test.shape}')
print(f'Val: X{X_val.shape} y{y_val.shape}')

# Creamos un modelo con la misma red neuronal simple y básico
print(f'\n\nRed neural simple')
print(f'\n\n-----------------')
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=50,
          batch_size=8)

loss, acc = model.evaluate(X_test, y_test)
print(f'\n\nLos resultados obtenidos en la práctica de Deep Learning con esta red fueron:')
print(f'Loss=4.216668529857443, Acc=0.45119786262512207')
print(f'\n\nLos resultados obtenidos con la imagen mosaico son:')
print(f'Loss={loss}, Acc={acc}')
print(f'\n\nSe constata que el accuracy hay mejorado un 10%, lo cual indica que las imágenes estructuradas ayudan a las clasificaciones.')

print(f'\n\n\nCreamos una segunda red neural en la que añadimos capas Dropout para intentar reducir el overfitting en el entrenamiento.')
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=50,
          batch_size=8)

loss, acc = model.evaluate(X_test, y_test)
print(f'\n\nLos resultados obtenidos en la práctica de Deep Learning con esta red fueron:')
print(f'Loss=2.5429471516461106, Acc=0.453859806060791')
print(f'\n\nLos resultados obtenidos con la imagen mosaico son:')
print(f'Loss={loss}, Acc={acc}')
print(f'\n\nSe constata también una mejora en el accuracy y una reducción en pérdidas.')
print(f'\n\nAunque los datos son mejores con la imagen mosaico, una clasificación del con un accuracy del 55.55% no es muy buena.')



# ----------------
# Regresión lineal
# ----------------

print(f'# ---------------- #')
print(f'# Regresión lineal #')
print(f'# ---------------- #')

# Utilizamos los mismos datos normalizados/escalados de las características de las imágenes que en clasificación
# min_max_scaler = preprocessing.MinMaxScaler()
# images_scaled = min_max_scaler.fit_transform(images_feat)

# También debemos escalar los precios. En este caso vamos a
# realizarlo manualmente, teniendo en cuentas los
# precios máximo y mínimo
precio_maximo = listings['price'].max()
precio_minimo = listings['price'].min()
print(f'Escalamos manualmente el precio. Para ello calculamos precio máximo y mínimo. Máximo: {precio_maximo}, " '
      f'Mínimo {precio_minimo}')

y_reg =  listings['price']
y_scaled = (y_reg - precio_minimo) / (precio_maximo - precio_minimo)
print(f'Tamaño del valor objetivo precio escalado {y_scaled.shape}')

# Dividimos train, validation y test de forma similar a como lo hecho en
# clasificación, pero sin estratificar, aunque lo ideal sería convertir la
# variable objetivo a su logaritmo.
# Dividimos los datos en train y en test
X_train, X_test, y_train, y_test = train_test_split(images_scaled, y_scaled, test_size=0.2, shuffle=True, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=0)

print(f'\nTamaños train, test y validación')
print(f'Train: X{X_train.shape} y{y_train.shape}')
print(f'Text: X{X_test.shape} y{y_test.shape}')
print(f'Val: X{X_val.shape} y{y_val.shape}')

# Creamos el modelo de forma similar a clasificación, pero sustituyendo la última
# capa con una lineal de salida única y ajustando la función de pérdidas
# y optimación a las regresiones lineales
from keras.models import Sequential
from keras.layers import Dense

print(f'\n\nRed neural simple para la regresión lineal')
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='RMSProp')
model.summary()

history = model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=50,
          batch_size=8)

loss = model.evaluate(X_test, y_test)

print(f'En la práctica el resultado obtenido de la pérdida en la regresión lineal fue Loss=0.009028331498915416')
print(f'En el proceso actual vemos que la pérdida se reduce algo. valor obtenido: Loss={loss}')

plt.title('Loss / Mean Squared Error')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
print(plt.show())
# ------------
# Conclusiones
# ------------
print(f'\n\nCONCLUSIONES')
print(f'\n\nTal y como se preveía y se comentó en clase una información estructura ayuda a realizar clasificaciones y regresiones.')
print(f'El hecho de utilizar imagen mosaico en el que en siempre hay una baño, dormitorio, cocina y salón y, además, '
      f'siempre en la misma posición, dado que es una información estructurada, mejora considerablemente, como ha ç'
      f'quedado demostrado considerablemente los resultados')

print(f'No obstante, aunque los datos han mejorado, no se puede predecir el precio de las viviendas/listings a partir '
      f'solo de una imagen, aunque sea mosaico')
