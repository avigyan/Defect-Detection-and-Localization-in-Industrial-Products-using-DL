from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.applications import VGG16
from keras.layers import Input
from keras.models import Model
from keras.utils import to_categorical
from imutils import paths
import numpy as np
import argparse
import os
from fcheadnet import FCHeadNet
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,help="path to input dataset")
ap.add_argument("-m", "--model", required=True,help="path to output model")
args = vars(ap.parse_args())

# grab the list of images that we’ll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

numclasses = len(classNames)

data = []
labels = []


for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label assuming
	# that our path has the following format:
	# /path/to/dataset/{class}/{image}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-2]
	data.append(image)
	labels.append(label)

data=np.array(data)
labels=np.array(labels)
data = data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,test_size=0.25, random_state=42)
# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

trainY = to_categorical(trainY, numclasses)
testY = to_categorical(testY, numclasses)

# load the VGG16 network, ensuring the head FC layer sets are left
# off
baseModel = VGG16(weights="imagenet",include_top=False,input_tensor=Input(shape=(224, 224, 3)))

# initialize the new head of the network, a GAP layer
# followed by a sigmoid/softmax classifier

headModel = FCHeadNet.build(baseModel, len(classNames), 256)

model = Model(inputs=baseModel.input, outputs=headModel)

for (i, layer) in enumerate(model.layers):
	print("[INFO] {}\t{}".format(i, layer.__class__.__name__))

model.summary()

# loop over all layers in the base model and freeze them so they
# will *not* be updated during the training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model (this needs to be done after our setting our
# layers to being non-trainable
print("[INFO] compiling model...")


opt = RMSprop(lr=0.001)
#model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the head of the network for a few epochs (all other
# layers are frozen) -- this will allow the new layers to
# start to become initialized with actual "learned" values
# versus pure random
print("[INFO] training head...")

model.fit(trainX, trainY, batch_size=10,validation_data=(testX, testY), epochs=50, verbose=1)


# evaluate the network after initialization
print("[INFO] evaluating after initialization...")
predictions = model.predict(testX, batch_size=10)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=classNames))

# now that the head layers have been trained/initialized, lets
# unfreeze the final set of CONV layers and make them trainable
for layer in baseModel.layers[15:]:
	layer.trainable = True

# for the changes to the model to take affect we need to recompile
# the model, this time using SGD with a *very* small learning rate
print("[INFO] re-compiling model...")
opt = SGD(lr=0.001)
#model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])
# train the model again, this time fine-tuning *both* the final set
# of CONV layers along with our set of new layers
print("[INFO] fine-tuning model...")
model.fit(trainX, trainY, batch_size=10,validation_data=(testX, testY), epochs=50, verbose=1)

# evaluate the network on the fine-tuned model
print("[INFO] evaluating after fine-tuning...")
#predictions = model.predict(testX, batch_size=32)
predictions = model.predict(testX, batch_size=10)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=classNames))
# save the model to disk
print("[INFO] serializing model...")
model.save(args["model"])








