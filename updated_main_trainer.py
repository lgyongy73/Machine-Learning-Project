import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Dense, Dropout, Rescaling, Input, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from data_reading import build_dataset 
from model_factory import build_vgg16_base 

def load_data():
    # Downscaled to 128x128 for CPU training speed. 
    # Added shuffle=False to test_ds to keep labels in a fixed position!
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'data/train', image_size=(128, 128), batch_size=32, label_mode='categorical'
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        'data/val', image_size=(128, 128), batch_size=32, label_mode='categorical'
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        'data/test', image_size=(128, 128), batch_size=32, label_mode='categorical', shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE
    return (train_ds.prefetch(buffer_size=AUTOTUNE), 
            val_ds.prefetch(buffer_size=AUTOTUNE), 
            test_ds.prefetch(buffer_size=AUTOTUNE))

def build_model():
    # Modified to look at the new 128x128 shape footprint
    model_base = build_vgg16_base(input_shape=(128, 128, 3))
    model_base.trainable = True  # Keep the base frozen
    
    inputs = Input(shape=(128, 128, 3))
    x = Rescaling(1./255)(inputs) 
    x = model_base(x)
    
    # Swapped Flatten for GlobalAveragePooling2D to eliminate millions of parameters
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x) # Reduced units from 4096 to 256 to stop overfitting
    x = Dropout(0.5)(x)
    
    predictions = Dense(2, activation='softmax', dtype='float32')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0001), # Slightly smaller step size for stability
                  metrics=['accuracy'])
    
    print(model.summary())
    return model

def calculate_class_weights(directory='data/train'):
    """
    Calculates balanced class weights based on the number of files in the directory.
    """
    # Count normal images
    normal_count = len(os.listdir(os.path.join(directory, 'normal')))
    # Count pneumonia images (adjust folder name if yours is 'opacity' or similar)
    pneumonia_count = len(os.listdir(os.path.join(directory, 'pneumonia')))
    
    total = normal_count + pneumonia_count
    
    # Standard formula: total_samples / (num_classes * class_samples)
    weight_for_0 = total / (2.0 * normal_count)
    weight_for_1 = total / (2.0 * pneumonia_count)
    
    class_weight = {0: weight_for_0, 1: weight_for_1}
    
    print(f"Dataset Counts -> Normal: {normal_count}, Pneumonia: {pneumonia_count}")
    print(f"Calculated Class Weights -> {class_weight}")
    
    return class_weight

def fitting(model, train_ds, val_ds):
    print('Fitting model...')

    # 1. Compute the class weights before fitting
    class_weights = calculate_class_weights('data/train')

    # Added EarlyStopping so it stops on its own if validation metrics degrade
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        callbacks=[early_stop],
        class_weight=class_weights
    )
    return history

def evaluation(model, history, test_ds):
    print("Generating evaluation report...")

    # Safely extract labels now that shuffle=False is enforced on test_ds
    y_true = np.concatenate([y for x, y in test_ds], axis=0)
    y_true = np.argmax(y_true, axis=1)

    y_pred = model.predict(test_ds)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print(classification_report(y_true, y_pred_classes, target_names=['Normal', 'Pneumonia']))

    print('Deriving the confusion matrix...')
    cm = confusion_matrix(y_true, y_pred_classes)
    print(cm)
    
    # Cleaned up confusion matrix visual plotting configuration
    plt.figure(figsize=(5,5))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    print('Plotting training history...')
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()

    return cm

if __name__ == "__main__":
    build_dataset('../Pneumonia_Dataset/mdai_rsna_project_x9N20BZa_images_2018-07-20-153330')
    train_ds, val_ds, test_ds = load_data()
    model = build_model()
    history = fitting(model, train_ds, val_ds)
    confusionmatrix = evaluation(model, history, test_ds)