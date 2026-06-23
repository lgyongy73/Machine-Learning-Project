# IMPORTS

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Flatten, Dropout, Rescaling, Input
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from data_reading import build_dataset   # preprocessing script
from model_factory import build_vgg16_base  # VGG16 architecture

# CHECK

# check if GPU is available (runs too slow, if not)
print("GPU available: ", tf.config.list_physical_devices('GPU'))

# FUNCTIONS

def load_data():
    '''
    Reads the data from the folder structure built for the project,
    the code can utilize the created variables.
    '''

    bs = 8

    # generators for all three splits, data shuffled / in random order
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'data/train',
        image_size = (224, 224),
        batch_size = bs,
        label_mode = 'categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        'data/val',
        image_size = (224, 224),
        batch_size = bs,
        label_mode = 'categorical',
        shuffle = False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        'data/test',
        image_size = (224, 224),
        batch_size = bs,
        label_mode = 'categorical',
        shuffle = False
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

    x = GlobalAveragePooling2D()(x) # or Flatten()(x)
    extractor = Model(inputs = model_base.input, outputs = x)
    '''
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(2, activation='softmax')(x)

    # pieces the model together
    model = Model(inputs = inputs, outputs = predictions)
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(learning_rate=1e-4),
                  metrics = ['accuracy'])
    
    return model, extractor
    '''

    return extractor

def fitting(model, train_ds, val_ds):
    
    # takes a smaller subset from the front of the shuffled datasets
    #train_ds_subset = train_ds.take(100) 
    #val_ds_subset = val_ds.take(20)

    # local minimum escape
    reduce_lr = ReduceLROnPlateau(
        monitor = 'val_accuracy', 
        factor = 0.2, 
        patience = 2, 
        min_lr = 1e-7,
        verbose = 1
    )

    # quit if loss does not decrease
    early_stop = EarlyStopping(
        monitor = 'val_loss', 
        patience = 3, # number of epochs to wait before stopping
        restore_best_weights = True,
        verbose = 1
    )

    print('Computing weights...')

    train_labels_raw = np.concatenate([y for x, y in train_ds], axis = 0)
    train_labels = np.argmax(train_labels_raw, axis = 1)

    weights = compute_class_weight(
        class_weight = 'balanced',
        classes = np.unique(train_labels),
        y = train_labels
    )

    class_weight = {
        0: weights[0],
        1: weights[1]
    }

    print("Class weights:", class_weight)

    print('Fitting model...')
    weight_cb = WeightTracker()
    history = model.fit(
        train_ds,
        epochs = 10,
        validation_data = val_ds,
        class_weight = class_weight,
        callbacks = [early_stop, reduce_lr, weight_cb]
    )

    return history, weight_cb

def fitting_head():
    # Load the pre-calculated features
    train_X = np.load('data/train_features.npy')
    train_y = np.load('data/train_labels.npy')
    val_X = np.load('data/val_features.npy')
    val_y = np.load('data/val_labels.npy')

    print('Building model head...')
    model_head = Sequential([
        Dense(4096, activation='relu', input_shape=(512,)), # input_shape is 512 if you used GlobalAveragePooling2D
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model_head.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # addresses class imbalance
    y_ints = np.argmax(train_y, axis = 1)
    weights = compute_class_weight('balanced', classes = np.unique(y_ints), y = y_ints)
    class_weight_dict = {0: weights, 1: weights[7]}

    # stopping conditions
    callbacks = [
        EarlyStopping(monitor = 'val_loss', patience = 3, restore_best_weights = True),
        ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.2, patience = 2),
        WeightTracker()
    ]
    weight_cb = WeightTracker()

    print('Fitting model head...')
    history = model_head.fit(
        train_X, train_y,
        epochs = 10, 
        batch_size = 8,
        validation_data = (val_X, val_y),
        class_weight = class_weight_dict,
        callbacks = callbacks
    )
    
    return model_head, history, weight_cb

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

    print("Accuracy evaluation...")

    y_fit = np.argmax(y_pred, axis = 1)
    print("Prediction distribution:", np.bincount(y_fit))
    y_test = np.concatenate([y for x, y in test_ds], axis = 0)
    y_true = np.argmax(y_test, axis = 1) 

    print("Accuracy:", accuracy_score(y_true, y_fit))

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

class WeightTracker(tf.keras.callbacks.Callback):
    '''
    For keeping track of and storing the weights used during training.
    '''

    def on_train_begin(self, logs = None):
        self.means = []
        self.stds = []
        self.sample_history = [] # store a small sample

    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.layers[-2].get_weights() # from 4096-unit hidden layer
        self.means.append(np.mean(weights))
        self.stds.append(np.std(weights))
        self.sample_history.append(weights.flatten()[:100]) # to avoid running out of memory
        print(f" - Saved weight stats for epoch {epoch+1}")

def plot_weights(weight_cb):

    epochs = range(1, len(weight_cb.means) + 1)
    
    plt.figure(figsize = (12, 5))
    
    # average weight values
    plt.subplot(1, 2, 1)
    plt.plot(epochs, weight_cb.means, marker='o', color='b')
    plt.title("Mean weight evolution (4096 dense layer)")
    plt.xlabel("Epoch")
    plt.ylabel("Average weight value")
    
    # distribution of weights for the last epoch
    plt.subplot(1, 2, 2)
    plt.hist(weight_cb.sample_history[-1], bins=30, color='g', alpha=0.7)
    plt.title("Weight distribution (final epoch)")
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()

def save_offline_features(dataset, name, extractor):
    print('Extracting features...')
    
    features = extractor.predict(dataset) # slow, but runs only once
    
    labels = np.concatenate([y for x, y in dataset], axis = 0)
    
    np.save(f'data/{name}_features.npy', features)
    np.save(f'data/{name}_labels.npy', labels)
    print('Saved features and labels!')

# RUNTIME

if __name__ == "__main__":

    # enough to run once
    #build_dataset('../Pneumonia_Dataset/mdai_rsna_project_x9N20BZa_images_2018-07-20-153330')
    
    train_ds, val_ds, test_ds = load_data()
    extractor = build_model() #model
    save_offline_features(train_ds, 'train', extractor)
    save_offline_features(val_ds, 'val', extractor)
    save_offline_features(test_ds, 'test', extractor)
    model_head, history, weight_cb = fitting_head()
    #history, weight_cb = fitting(model, train_ds, val_ds)
    confusionmatrix = evaluation(model_head, history, test_ds)
    plot_weights(weight_cb)

'''
Sources:

https://www.comet.com/site/blog/improving-the-accuracy-of-your-neural-network/
https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/
'''