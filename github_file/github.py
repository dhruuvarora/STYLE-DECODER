import tensorflow as tf
from keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D,Flatten,Dense,Reshape


import tensorflow.compat.v1



tf.losses.sparse_softmax_cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy

#tf.compat.v1.executing_eagerly_outside_functions = tf.executing_eagerly_outside_functions

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3)) 
model.trainable = False

model.summary()

model= tf.keras.Sequential([model, GlobalMaxPooling2D()])
# #model.add(MaxPooling2D)
model.summary()