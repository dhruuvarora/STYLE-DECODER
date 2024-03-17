import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D,Flatten,Dense,Reshape


import tensorflow.compat.v1

import cv2
import numpy as np
from numpy.linalg import norm

import pickle

tf.losses.sparse_softmax_cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy

#tf.compat.v1.executing_eagerly_outside_functions = tf.executing_eagerly_outside_functions

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3)) 
model.trainable = False

model.summary()

model= tf.keras.Sequential([model, GlobalMaxPooling2D()])
# #model.add(MaxPooling2D)
model.summary()

img=cv2.imread('dataset-card.jpg',0)

cv2.namedWindow("Frame")
img=cv2.imread("C:\One Drive Data\Desktop\dataset-card.jpg")
cv2.imshow("Frame", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img=cv2.resize(img,(224,224))

img=np.array(img)

img.shape

# (number_of_image , 224,224,3)
expand_img=np.expand_dims(img,axis=0)

expand_img.shape

pre_image=preprocess_input(expand_img)

pre_image.shape

result=model.predict(pre_image).flatten()

result.shape

normalized=result/norm(result)

normalized.shape

def extract_feature(image_path, model):
    img=cv2.imread(image_path)
    img=cv2.resize(img,(224,224))
    img=np.array(img)
    expand_img=np.expand_dims(img, axis=0)
    pre_image=preprocess_input(expand_img)
    result=model.predict(pre_image).flatten()
    normalized=result/norm(result)

    return normalized


extract_feature('C:\One Drive Data\Desktop\dataset-card.jpg',model)

import os
from tqdm import tqdm

filename=[]
feature_list=[]
for file in os.listdir(r'C:\JupyterNotebook\Dataset\data\Images'):
    filename.append(os.path.join(r'C:\JupyterNotebook\Dataset\data\Images',file))

filename[0:5]

for file in tqdm(filename):
    feature_list.append(extract_feature(file,model))

pickle.dump(feature_list, open('featurevector.pkl', 'wb'))
pickle.dump(filename, open('filenames.pkl', 'wb'))
