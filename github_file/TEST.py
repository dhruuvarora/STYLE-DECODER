import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D, Flatten, Dense, Reshape

import tensorflow.compat.v1
tf.losses.sparse_softmax_cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy

import cv2
import numpy as np
from numpy.linalg import norm
import pickle

from sklearn.neighbors import NearestNeighbors

feature_list=np.array(pickle.load(open("featurevector.pkl","rb")))
filename=pickle.load(open("filenames.pkl","rb"))


model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3)) 
model.trainable = False

model= tf.keras.Sequential([model, GlobalMaxPooling2D()])
model.summary()

# def extract_feature(image_path, model):
img=cv2.imread('1637.jpg')
img=cv2.resize(img,(224,224))
img=np.array(img)
expand_img=np.expand_dims(img, axis=0)
pre_image=preprocess_input(expand_img)
result=model.predict(pre_image).flatten()
normalized=result/norm(result)

    # return normalized



