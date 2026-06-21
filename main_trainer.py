# IMPORTS

import cv2
from keras.layers import Dense, Flatten, Dropout, Rescaling, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical8
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf

from data_reading import build_dataset   # preprocessing script
from model_factory import build_vgg16_base  # VGG16 architecture

# CHECK

# check if GPU is available (runs too slow, if not)
print("Is GPU available: ", tf.config.list_physical_devices('GPU'))

# FUNCTIONS

def load_data():
    '''
    Reads the data from the folder structure built for the project,
    the code can utilize the created variables.
    '''

    # generators for all three splits
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'data/train',
        image_size = (224, 224),
        batch_size = 32,
        label_mode = 'categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        'data/val',
        image_size = (224, 224),
        batch_size = 32,
        label_mode = 'categorical'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        'data/test',
        image_size = (224, 224),
        batch_size = 32,
        label_mode = 'categorical'
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

def build_model():
    '''
    Creates the full model structure,
    imports the base, then adds flattening and dense layers.
    '''
    
    # imports the VGG16 base
    model_base = build_vgg16_base(input_shape = (224, 224, 3))
    # freezes the base / reduces parameters to be fitted
    model_base.trainable = False
    inputs = Input(shape=(224, 224, 3))
    x = Rescaling(1./255)(inputs) 

    x = model_base(x)

    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(2, activation='softmax')(x)

    # pieces the model together
    model = Model(inputs = inputs, outputs = predictions)
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(learning_rate=0.001),
                  metrics = ['accuracy'])
    
    return model

def fitting(model, train_ds, val_ds):
    
    print('Fitting model...')
    history = model.fit(
        train_ds,
        epochs = 10,
        validation_data = val_ds
    )

    return history

def evaluation(model, history, test_ds):
    '''
    Generates the evaluation report,
    derives the confusion matrix
    and plots the training history.

    Aims to characterize the precision of the fit.
    '''

    print("Generating evaluation report...")

    # predicts on the test set
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_true = np.argmax(y_true, axis=1)

    y_pred = model.predict(test_ds)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(classification_report(y_true, y_pred_classes))

    print('Deriving the confusion matrix...')
    cm = confusion_matrix(y_true, y_pred_classes)
    print(cm)
    plt.imshow(cm)
    plt.show()

    print('Plotting training history...')
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()

    return cm

# RUNTIME

if __name__ == "__main__":

    build_dataset('../Pneumonia_Dataset/mdai_rsna_project_x9N20BZa_images_2018-07-20-153330')
    train_ds, val_ds, test_ds = load_data()
    model = build_model()
    history = fitting(model, train_ds, val_ds)
    confusionmatrix = evaluation(model, history, test_ds)