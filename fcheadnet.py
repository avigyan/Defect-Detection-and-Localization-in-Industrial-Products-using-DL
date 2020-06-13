# import the necessary packages
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.pooling import AveragePooling2D
from keras.layers import GlobalAveragePooling2D

class FCHeadNet:
	@staticmethod
	def build(baseModel, classes, D):
		# initialize the head model that will be placed on top of
		# the base, then add a GAP layer
		headModel = baseModel.output
		headModel=GlobalAveragePooling2D()(headModel)
		
		# add a sigmoid/softmax layer
		if(classes>2):
			headModel = Dense(classes, activation="softmax")(headModel)
		else:
			headModel = Dense(classes, activation="sigmoid")(headModel)
		# return the model
		return headModel

