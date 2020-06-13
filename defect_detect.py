import sys
import os
import numpy as np
from PIL import Image
# %tensorflow_version 1.14
#import tensorflow
import tensorflow as tf
from tensorflow.python.framework import ops
import cv2
from matplotlib import pyplot as plt
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import argparse

def visualize_class_activation_map(model_path, img_path):
  model = load_model(model_path,compile=False)
  #model = load_model(model_path)
  original = cv2.imread(img_path, 1)
  original = cv2.resize(original, (224, 224))
  original_img = image.load_img(img_path, target_size=(224, 224))
  #original_img=original.copy()
  #original = image.img_to_array(original)
  original_img = image.img_to_array(original_img)
  original_img = np.expand_dims(original_img, axis=0)
  original_img = preprocess_input(original_img)
  
  _, width, height, _ = original_img.shape
  
  img = original_img
       
  #Get the 512 input weights to the sigmoid/softmax.
  class_weights = model.layers[-1].get_weights()[0]
  
  final_conv_layer = get_output_layer(model, "block5_conv3")
  get_output = K.function([model.layers[0].input],[final_conv_layer.output,model.layers[-1].output])
  [conv_outputs, predictions] = get_output([img])
  target_class = np.argmax(predictions[0])
  conv_outputs = conv_outputs[0, :, :, :]
  
  if(target_class==1):# may have to change this if more than 2 classes present
    print("No defect present")
  else:
    print("Defect present")
    #Create the class activation map.
    cam = np.zeros(dtype = np.float32, shape = conv_outputs.shape[0:2])#14x14
 
  
    for i, w in enumerate(class_weights[:, target_class]):#512x1
      cam += w * conv_outputs[:, :, i]#14x14

    cam /= np.max(cam)
    cam = cv2.resize(cam, (height, width))
    #heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_VIRIDIS)

    #heatmap[np.where(cam < 0.2)] = 0
    heatmap[np.where(cam < 0.7)] = 0 #set threshold on cam
    #img = heatmap*0.5 + original_img[0, :, :, :]
  
    gray = cv2.cvtColor(heatmap,cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    # morphological gradient
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, morph_kernel)
    # connect horizontally oriented regions
    connected = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, morph_kernel)
    mask = np.zeros(thresh1.shape, np.uint8)
    # find contours
    contours, hierarchy = cv2.findContours(connected,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    img2=np.uint8(img[0, :, :, :])
    # filter contours
    for idx in range(0, len(hierarchy[0])):
      rect = x, y, rect_width, rect_height = cv2.boundingRect(contours[idx])
      # fill the contour
      mask = cv2.drawContours(mask, contours, idx, (255, 255, 2555), cv2.FILLED)
      # ratio of non-zero pixels in the filled region
      r = float(cv2.countNonZero(mask)) / (rect_width * rect_height)
      if r > 0.45 and rect_height > 8 and rect_width > 8:
        result = cv2.rectangle(original, (x, y+rect_height), (x+rect_width, y), (0,0,255),1)

    #Image.fromarray(result).show()
    output = cv2.addWeighted(np.uint8(img[0, :, :, :]), 0.3, heatmap, 1 - 0.3, 0)
    #plt.title('CAM')
    #plt.imshow(cam)
    #plt.show()
    #plt.imshow(img)
    #plt.imshow(heatmap, cmap='jet', alpha=0.5)
    #plt.imshow(output)
    #plt.imshow(thresh1)
    #plt.imshow(original)
    #plt.show()
    cv2.imshow('Defect_detection', original)
    cv2.waitKey(0)
    #cv2.imwrite(output_path, img)

def get_output_layer(model, layer_name):
  # get the symbolic outputs of each "key" layer (we gave them unique names).
  layer_dict = dict([(layer.name, layer) for layer in model.layers])
  layer = layer_dict[layer_name]
  return layer

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to test image")
ap.add_argument("-m", "--model", required=True,help="path to trained model")
args = vars(ap.parse_args())

visualize_class_activation_map(args["model"], args["image"])
