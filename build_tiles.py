"""
This program builds a mosaic image for every Airbnb listing

"""

import argparse
import os
import time
import traceback

import cv2

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

import imageio as io
import pandas as pd
import numpy as np
import config

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import smart_resize
from tensorflow.keras.models import load_model
from tensorflow.python.keras.applications.resnet import ResNet50

# ---------
# Functions
# ---------

def get_photos(url):
    global driver

    driver.get(url)
    time.sleep(1)

    images = []
    attempts = 0
    while True:
        attempts = attempts + 1
        # Wait for the page to load
        try:
            element = driver.find_element(By.CSS_SELECTOR, ".\\_13xgr5hw")
            break
        except:
            # check if it is a non-active listing
            try:
                time.sleep(0.2)
                element = driver.find_element(By.CSS_SELECTOR, ".\\_1wc4ltr")
                return None # listing is not active
            except:
                if attempts == 3: return images # Impossible to load the page
                time.sleep(0.2*attempts)

    actions = ActionChains(driver)
    actions.move_to_element(element).perform()
    element.click()
    time.sleep(0.5)

    number_images = 0
    bolFirst = True
    while True:
        try:
            img = driver.find_element(By.XPATH, "//div[@role='presentation']//img[@class='_6tbg2q']")
        except:
            try:
                img = driver.find_element(By.XPATH, "//div[@role='presentation']//img[@class='_9ofhsl']")
            except:
                break
        try:
            src = img.get_attribute('src')
        except: # ignore error
            continue
        read_number = 0
        read_src = False
        while True:
            try:
                image = io.imread(src)
                read_src = True
                break
            except:
                read_number = read_number + 1
                if read_number == 3:
                    break
        if read_src:
            images.append(image)
            number_images = number_images + 1
            if number_images == 32:
                break
        try:
            next_buttons = driver.find_elements(By.CSS_SELECTOR, ".\\_v25a70")
            if (len(next_buttons) == 1):
                if bolFirst :
                    bolFirst = False
                    next_button = next_buttons[0]
                else:
                    break
            else:
                next_button = next_buttons[1]
            # element = driver.find_element(By.CSS_SELECTOR, ".\\_v25a70")
            actions = ActionChains(driver)
            actions.move_to_element(next_button).perform()
            next_button.click()
            if (number_images >= 26):
                time.sleep(0.3) # wait longer every 26 photos
            elif (number_images >= 16):
                time.sleep(0.2)  # wait longer every 16 photos
            else:
                time.sleep(0.1)
        except:
            break

    return images

def startWebDriver():
    global driver
    options = Options()
    options.add_argument("--disable-infobars")
    driver = webdriver.Chrome(chrome_options=options)
    # driver = webdriver.Firefox()

def buildTile(main_model, imgs, rooms, bs, room_models, folder, subfolder, listing_name):
    images = []
    for img in imgs:
        # load the input image using the Keras helper utility
        # while ensuring the image is resized to 224x224 pixels
        # Si la imagen es en blanco y negro, se produce error porque no tiene 3 canales,
        # así pues, la convertimos a color para que tenga 3 canales
        if img.ndim < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        image = smart_resize(img, size=(224, 224))
        image = img_to_array(image)

        # preprocess the image by (1) expanding the dimensions and
        # (2) subtracting the mean RGB pixel intensity from the
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # add the image to the batch
        images.append(image)

    images = np.vstack(images)
    features = main_model.predict(images, batch_size=bs)
    features = features.reshape((features.shape[0], 7 * 7 * 2048))
    bestPhotos = []
    for room in rooms:
        room_model = room_models[room]

        # pass the image through the network to obtain our predictions
        predictions = room_model.predict(x=features)
        bestPhotos.append(imgs[np.argmax(predictions, axis=0)[0]])

    # Build mosaic with the best 4 photos with size 224 * 224
    # initialize our list of input images along with the output image
    outputImage = np.zeros((224, 224, 3), dtype="uint8")
    inputImages = []

    # loop over the best photos
    for photo in bestPhotos:
        # load the input image, resize it to be 112 112, and then
        # update the list of input images
        image = smart_resize(photo, (112, 112))
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
    tile = os.path.sep.join([folder, subfolder, listing_name + ".jpg"])
    cv2.imwrite(tile, outputImage)

def getTileFromAirbnb(url, listing_name, num_listings, subfolder):
    attempts_reading_airbnb = 0
    while True:
        attempts_reading_airbnb = attempts_reading_airbnb + 1
        images = get_photos(url)
        if images == None:
            print (f'Listing {listing_name} is not active')
            return
        elif len(images) == 0 :
            time.sleep(2) # wait for two seconds because this page takes long time to be loaded
            if (attempts_reading_airbnb == 3):
                print(f'Impossible to load page of listing {listing_name}')
                return
        else:
            listing_name = str(int(listing_name)).zfill(8)
            try:
                buildTile(main_model, images, config.ROOMS, config.BATCH_SIZE, room_models, config.TILES_PATH, subfolder, listing_name)
            except Exception:
                print(f'Impossible to build tile for listing {listing_name}')
                traceback.print_exc()
            return


# ------------
# Init program
# ------------

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--start", type=int, required=True,
	help="index/row of first listing in dataset to be processed")
ap.add_argument("-e", "--end", type=int, required=True,
	help="index/row of last listing (not included) in dataset to be processed")
args = vars(ap.parse_args())

# Load listings
listing_file = os.path.sep.join([config.RESOURCES_PATH, config.LISTINGS_CSV_GZ])
listing_urls = pd.read_csv(listing_file)
listing_urls = listing_urls[['listing_url']]
start_listing = args['start']
end_listing = args['end']
listing_urls = listing_urls[start_listing:end_listing]

# Load models
main_model = ResNet50(weights="imagenet", include_top=False)
room_models = {}
for room in config.ROOMS:
    model_path = os.path.sep.join([config.MODEL_PATH, room + "_model.h5"])
    room_model = load_model(model_path)
    room_models.update({room : room_model})

startWebDriver()
num_listings = start_listing
total_time = 0
mean_time = 0
folder = ""
before_subfolder = ""
for index, listing_url in listing_urls.iterrows():
    start_time = time.time()

    # Get url of listing
    url = listing_url['listing_url']

    # get listing name/code from url
    listing_name = url.split('/')[-1]

    # Count listings
    num_listings = num_listings + 1

    # Subfolder where tile will be saved
    subfolder = str(num_listings // 1000 + 1).zfill(2)
    if (subfolder != before_subfolder):
        tiles_folder = os.path.sep.join([config.TILES_PATH, subfolder])
        # if the output directory does not exist, create it
        if not os.path.exists(tiles_folder):
            os.makedirs(tiles_folder)
        before_subfolder = subfolder

    # Build a tile image from listing's photos
    getTileFromAirbnb(url, listing_name, num_listings, subfolder)

    # Measure time
    end_time = time.time()
    partial_time = end_time - start_time
    total_time = total_time + partial_time
    mean_time = total_time / (num_listings - start_listing)
    if (num_listings % 100 == 0):
        print (f'Total listings {num_listings}. Mean time by listing {mean_time}. Total time {total_time}')

print(f'\n\nTotal listings {num_listings}. Mean time by listing {mean_time}. Total time {total_time}')
driver.quit()

