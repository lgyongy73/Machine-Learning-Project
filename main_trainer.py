# Imports

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input
from keras.optimizers import Adam
from pathlib import Path
import cv2
from keras.utils import to_categorical
import sklearn.preprocessing
import matplotlib.pyplot as plt

from data_reading import build_dataset   # preprocessing script
from model_factory import build_vgg16_base  # VGG16 architecture

# Functions

def load_split(split_path):
    '''
    Iterates through the subfolders of a split (train or test or val),
    loads images, and assigns labels.
    '''

    data = []
    categories = ['normal', 'pneumonia']
    
    for category in categories:
        # path to the specific category folder (e.g. 'data/train/normal')
        path = Path(split_path) / category
        label = categories.index(category) # 0 for normal, 1 for pneumonia
        
        # finds all .png files in the folder
        for img_path in path.glob('*.png'):
            # loads the image in color as required
            image = cv2.imread(str(img_path))
            if image is not None:
                # appends a tuple of (image_matrix, label)
                data.append((image, label))
                
    return data
def extract_data():
    '''
    Converts the images stored on the local disk
    into variables that can be referred to in python.
    '''

    print('Extracting images into memory...')
    train_data = load_split('data/train')
    val_data = load_split('data/val')
    test_data = load_split('data/test')

    X_train = np.array([x for x, y in train_data])
    y_train = np.array([y for x, y in train_data])

    X_val = np.array([x for x, y in val_data])
    y_val = np.array([y for x, y in val_data])

    X_test = np.array([x for x, y in test_data])
    y_test = np.array([y for x, y in test_data])

    # scales pixel values for better optimization
    X_train = X_train.astype('float32') / 255.0
    X_val = X_val.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # converts labels to one-hot encoding for the softmax classifier
    y_train = to_categorical(y_train, 2)
    y_val = to_categorical(y_val, 2)
    y_test = to_categorical(y_test, 2)

    return X_train, y_train, X_val, y_val, X_test, y_test

# Runtime

X_train, y_train, X_val, y_val, X_test, y_test = extract_data()

# imports the 13-layer VGG16 base
model = build_vgg16_base()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Fitting model...')
model.fit(X_train, y_train, epochs=10, batch_size=128)

pred = model.predict(X_train)
res = sklearn.preprocessing.OneHotEncoder.inverse_transform(pred).reshape(-1)

print('Deriving the confusion matrix...')
confusion_matrix = confusion_matrix(res, y_train)
plt.imshow(confusion_matrix)

print(accuracy_score(y_train,res))

"""
def train_model():
    '''
    Function to manage the classification, optimization, and evaluation stages.
    '''

#if __name__ == "__main__":
    model_done = train_model()
"""

'''
STORE

build_dataset('../Pneumonia_Dataset/mdai_rsna_project_x9N20BZa_images_2018-07-20-153330')

# generating report
print("Generating report...")
y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)
print("\nClassification report:\n", classification_report(y_true, y_pred_classes))

# generating confusion matrix
print("Generating confusion matrix...")
cm = confusion_matrix(y_true, y_pred_classes)
print("Confusion Matrix:\n", cm)
'''