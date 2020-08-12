# import the necessary packages
import os

# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "rooms"

# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "dataset"

# define the names of the training, testing, and validation
# directories
TRAIN = "training"
TEST = "test"
VAL = "validation"

# initialize the list of types of rooms
ROOMS = ["bathrooms", "bedrooms", "kitchens", "livingrooms"]

# Number of classes
CLASSES_NUMBER = 2
# set the batch size
BATCH_SIZE = 32

# initialize the label encoder file path and the output directory to
# where the extracted features (in CSV file format) will be stored
BASE_CSV_PATH = "output"

# Folder where models will be saved
MODEL_PATH = "models"

PREDICTION_PATH = "predictions"

RESOURCES_PATH = "resources"

TILES_PATH = "tiles"

IMAGE_FEATURES_FILE = "images_feat.npy"

IMAGE_SENTIMENT_FEATURES_FILE = "images_sentiment_feat.npy"

LISTINGS_FILE = "listings.csv"

LISTINGS_CSV_GZ = "listings.csv.gz"

REVIEWS_CSV_GZ = "reviews.csv.gz"

NLP_PATH = "nlp"

NLP_LANG_FILE = "reviews_with_language.csv"
NLP_LANG_AND_SENTIMENT_FILE = "reviews_with_language_and_sentiment.csv"
NLP_SENTIMENT_FILE = "reviews_with_sentiment.csv"