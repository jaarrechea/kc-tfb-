import os

import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.python.keras.applications.resnet import ResNet50

import config
from imutils import paths
import numpy as np

model = ResNet50(weights="imagenet", include_top=False)

# loop over the images and labels in the current batch
imagePaths = list(paths.list_images(config.PREDICTION_PATH))
images = []
for imagePath in imagePaths:
    # load the input image using the Keras helper utility
    # while ensuring the image is resized to 224x224 pixels
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)

    # preprocess the image by (1) expanding the dimensions and
    # (2) subtracting the mean RGB pixel intensity from the
    # ImageNet dataset
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # add the image to the batch
    images.append(image)

images = np.vstack(images)
features = model.predict(images, batch_size=config.BATCH_SIZE)
features = features.reshape((features.shape[0], 7 * 7 * 2048))
bestPhotos = []
for room in config.ROOMS:
    print(f"[INFO] loading model {room}...")
    model_path = os.path.sep.join([config.MODEL_PATH, room + "_model.h5"])
    room_model = load_model(model_path)

    # pass the image through the network to obtain our predictions
    predictions = room_model.predict(x=features)
    bestPhotos.append(imagePaths[np.argmax(predictions, axis=0)[0]])

# Build mosaic with the best 4 photos with size 224 * 224
# initialize our list of input images along with the output image
outputImage = np.zeros((224, 224, 3), dtype="uint8")
inputImages = []
# loop over the best photos
for photo in bestPhotos:
    # load the input image, resize it to be 112 112, and then
    # update the list of input images
    image = cv2.imread(photo)
    image = cv2.resize(image, (112, 112))
    inputImages.append(image)
# tile the four input images in the output image such the first
# image goes in the top-right corner, the second image in the
# top-left corner, the third image in the bottom-left corner,
# and the final image in the bottom-right corner
outputImage[0:112, 0:112] = inputImages[0]
outputImage[0:112, 112:224] = inputImages[1]
outputImage[112:224, 0:112] = inputImages[2]
outputImage[112:224, 112:224] = inputImages[3]
# add the tiled image to our set of images the network will be
# trained on
tile = os.path.sep.join([config.PREDICTION_PATH, "tile.jpg"])
cv2.imwrite(tile, outputImage)

