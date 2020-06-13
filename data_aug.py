from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from imutils import paths, resize
import numpy as np
import argparse
import os
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
ap.add_argument("-s", "--save", required=True,help="path to output image")
args = vars(ap.parse_args())

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20,width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,horizontal_flip=True,vertical_flip=True, fill_mode="nearest")

# grab the list of images that we’ll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

i=0
for path in imagePaths:
	img = cv2.imread(path)
	img = cv2.resize(img, (224, 224))
	img = img_to_array(img)
	img = img.reshape((1,) + img.shape)
	for batch in aug.flow(img, batch_size=1,save_to_dir=args["save"],save_prefix=classNames[0], save_format='png'):
		i += 1
		if i > 99:
			break  # otherwise the generator would loop indefinitely 

