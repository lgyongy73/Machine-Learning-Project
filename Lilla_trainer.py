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

import tensorflow as tf
import os

# CHANGE 2: Callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

"""from tensorflow.keras import mixed_precision

# Set policy to mixed_float16
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

print(f"Compute dtype: {policy.compute_dtype}")
print(f"Variable dtype: {policy.variable_dtype}")"""

# ==================================================
# CHANGE 2: XLA compilation
# TensorFlow graph optimization
# ==================================================
tf.config.optimizer.set_jit(True)
# Force the math execution engine to utilize all processor cores
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(2)

# ==================================================
# CHANGE 3: GPU memory growth
# Prevent TensorFlow from reserving all GPU memory
# ==================================================
"""gpus = tf.config.list_physical_devices('GPU')

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(
            gpu,
            True
        )
"""
np.random.seed(42)
tf.random.set_seed(42)

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
                image = cv2.resize(image, (128, 128)) # Resize to match VGG16 input size
                data.append((image, label))
                
    return data
def extract_data():
    '''
    Converts the images stored on the local disk
    into variables that can be referred to in python.
    '''

    print('Extracting images into memory...')
    train_data = load_split('data/train')
    np.random.shuffle(train_data)
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
#for debug
train_labels = np.argmax(y_train, axis=1)
unique, counts = np.unique(train_labels, return_counts=True)
print("Training set distribution:")
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count}")
#X_train = X_train[:1000]
#y_train = y_train[:1000]

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

class_weight = {
    0: weights[0],
    1: weights[1]
}

print("Class weights:", class_weight)

# imports the 13-layer VGG16 base
model = build_vgg16_base()

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# ==================================================
# CHANGE 4: Explicit Adam optimizer
# ==================================================
optimizer = Adam(
    learning_rate=5e-5
)

# ==================================================
# CHANGE 5:
# Softmax + one-hot labels -> categorical_crossentropy
# ==================================================
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# ==================================================
# CHANGE 6: Early stopping
# ==================================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=6,
    restore_best_weights=True,
    verbose=1
)

# ==================================================
# CHANGE 7: Reduce learning rate
# ==================================================
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1
)

# ==================================================
# CHANGE 8: Save best model
# ==================================================
checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print('Fitting model...')
#model.fit(X_train, y_train, epochs=10, batch_size=128)
# ==================================================
# CHANGE 10: Larger batch size
# Increase if GPU memory allows
# ==================================================
BATCH_SIZE = 128

# 1. Convert NumPy arrays to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

# 2. Optimize the pipeline
# Cache reads, shuffle, batch, and crucially—PREFETCH
BUFFER_SIZE = len(X_train)

train_dataset = (train_dataset
                 .shuffle(buffer_size=BUFFER_SIZE)
                 .batch(BATCH_SIZE)
                 .prefetch(buffer_size=tf.data.AUTOTUNE))

val_dataset = (val_dataset
               .batch(BATCH_SIZE)
               .prefetch(buffer_size=tf.data.AUTOTUNE))

# 3. Fit the model using the datasets instead of raw X, y arrays
history = model.fit(
    train_dataset,              # Pass dataset directly
    validation_data=val_dataset,# Pass dataset directly
    epochs=20, 
    class_weight=class_weight,
    callbacks=[early_stop, lr_scheduler, checkpoint]
)

"""history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20, 
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=[
        early_stop,
        lr_scheduler,
        checkpoint
    ]
)"""

"""pred = model.predict(X_train)
res = sklearn.preprocessing.OneHotEncoder.inverse_transform(pred).reshape(-1)

print('Deriving the confusion matrix...')
confusion_matrix = confusion_matrix(res, y_train)
plt.imshow(confusion_matrix)

print(accuracy_score(y_train,res))"""

# ==================================================
# CHANGE 11: Evaluate on TEST data
# ==================================================

print("Generating predictions...")

test_dataset = tf.data.Dataset.from_tensor_slices(X_test).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
y_pred_prob = model.predict(test_dataset)

#y_pred_prob = model.predict(X_test)

y_pred = np.argmax(
    y_pred_prob,
    axis=1
)
print("Prediction distribution:")
print(np.bincount(y_pred))

y_true = np.argmax(
    y_test,
    axis=1
)

# ==================================================
# CHANGE 12: Accuracy
# ==================================================

print(
    "Accuracy:",
    accuracy_score(
        y_true,
        y_pred
    )
)

# ==================================================
# CHANGE 13: Classification report
# ==================================================

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "normal",
            "pneumonia"
        ]
    )
)

# ==================================================
# CHANGE 14: Confusion matrix
# ==================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6,6))

plt.imshow(cm)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("True")

plt.colorbar()

plt.show()

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