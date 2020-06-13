import numpy as np
import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array

model=load_model('defect_detect.model',compile=False)
image1 = cv2.imread('good_image.png')
image1=cv2.resize(image1, (224, 224))
image1 = img_to_array(image1)
image1 = image1.reshape((1,) + image1.shape)
image1=np.array(image1)
image1 = image1.astype("float") / 255.0
predictions1 = model.predict(image1)
print(predictions1)

image2 = cv2.imread('bad_image.png')
image2=cv2.resize(image2, (224, 224))
image2 = img_to_array(image2)
image2 = image2.reshape((1,) + image2.shape)
image2=np.array(image2)
image2 = image2.astype("float") / 255.0
predictions2 = model.predict(image2)
print(predictions2)

