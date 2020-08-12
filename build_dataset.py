# USAGE
# python build_dataset.py

# import the necessary packages
import config
from imutils import paths
import shutil
import os

# loop over the data splits
for room in config.ROOMS:
	# loop over the data splits
	for split in (config.TRAIN, config.TEST, config.VAL):
		# grab all image paths in the current split
		print(f"[INFO] processing '{room}/{split} split'...")
		p = os.path.sep.join([config.ORIG_INPUT_DATASET, room, split])
		imagePaths = list(paths.list_images(p))

		# loop over the image paths
		for imagePath in imagePaths:
			# extract class label from the filename
			filename = imagePath.split(os.path.sep)[-1]
			prefix = ""
			if (int(filename.split("_")[0]) == 0):
				label = "non_" + room
			else:
				label = room

			# construct the path to the output directory
			dirPath = os.path.sep.join([config.BASE_PATH, room, split, label])

			# if the output directory does not exist, create it
			if not os.path.exists(dirPath):
				os.makedirs(dirPath)

			# construct the path to the output image file and copy it
			p = os.path.sep.join([dirPath, filename])
			shutil.copy2(imagePath, p)