import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
import tensorflow as tf
#from keras.preprocessing.image import ImageDataGenerator

# Build Convolutional Blocks
# Sources:
# https://pysource.com/2022/10/04/vgg16-from-scratch-computer-vision-with-keras-p-7/
# https://builtin.com/machine-learning/vgg16 
# https://arxiv.org/abs/1409.1556 

# VGG16 (VGG = Visual Geometry Group of the University of Oxford)
def build_vgg16_base(input_shape=(224, 224, 3)):
    model = Sequential()
    
    # First block: 2 conv layers with 64 filters, then max pooling
    model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    # Second block: 2 conv layers with 128 filters, then two max pooling with strides (2,2)
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    # Third block: 3 conv layers with 256 filters, then max pooling with strides (2,2)
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    # Fourth block: 3 conv layers with 512 filters, then max pooling with strides (2,2)
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    # Fifth block: 3 conv layers with 512 filters, then max pooling with strides (1,1)
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(1,1)))

    return model


# Load image test
img=cv2.imread('data/train/pneumonia/1.2.276.0.7230010.3.1.2.8323329.1472.1517874291.114974.png')
cv2.imshow('frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()