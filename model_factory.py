# Imports
import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input

# Build Convolutional Blocks
# Sources:
# https://pysource.com/2022/10/04/vgg16-from-scratch-computer-vision-with-keras-p-7/
# https://builtin.com/machine-learning/vgg16 
# https://arxiv.org/abs/1409.1556 - We followed the implementation of the VGG16 based on the description included in the paper.

# VGG16 (VGG = Visual Geometry Group of the University of Oxford, 16 = number of convolutional layers)
def build_vgg16_base(input_shape=(224, 224, 3)):
    """
    Builds the VGG16 model

    Input parameters:
    input_shape: tuple, the shape of the unput images (height, width, number of channels). Must be (224, 224, 3)

    Output:
    model: keras sequential model
    
    The function also prints the summary of the model at the end
    """

    model = Sequential()
    print("Building VGG16...")

    # To make sure we get no error while using keras, we define the input shape in a seperate input layer (other errors can be ignored concerning computation speed - at least on my pc):
    model.add(Input(shape=input_shape)) 
    
    # First block: 2 conv layers with 64 filters, then max pooling with strides (2,2)
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
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

    # Sixth block: 2 fully connected hidden layer containing 4096 units, then a 1000-unit softmax output layer
    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dense(4096, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    # Now we have 16 layers altogether
    print("Done!")
    print(model.summary())

    return model

model= build_vgg16_base()
