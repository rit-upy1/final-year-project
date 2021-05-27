from __future__ import print_function
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import os
from kerastuner.tuners import BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

num_classes = 7
img_rows, img_cols = 48, 48
batch_size = 64

train_data_dir = 'data/train'
validation_data_dir = 'data/test'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=1,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
				target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
				target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

def build_model(hp):

    model = Sequential()

    # Block-1

    model.add(Conv2D(hp.Int("input_units", min_value=32, max_value=2048, step=32), kernel_size=(3, 3), padding='same',
                    kernel_initializer='he_normal', input_shape=(img_rows, img_cols, 1)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same',
                    kernel_initializer='he_normal', input_shape=(img_rows, img_cols, 1)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(hp.Float("input_dropout",min_value = 0.0, max_value = 0.99,step = 0.05)))

    # Block-2
    for i in range(hp.Int('num_layers', 1, 20)):
        model.add(Conv2D(hp.Int(f"conv_{i}_units", min_value=32, max_value=2048, step=32), kernel_size=(3, 3), padding='same',
                        kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(hp.Int(f"conv_{i}_units", min_value=32, max_value=2048, step=32), kernel_size=(3, 3), padding='same',
                        kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(hp.Float(f"dropout_{i}",
                                   min_value=0.0, max_value=0.99, step=0.05)))

    model.add(Dense(num_classes, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    return model 




checkpoint = ModelCheckpoint('Emotion_little_vgg.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

tuner = BayesianOptimization(build_model,
                     objective="val_accuracy",
                     max_trials=3,
                     executions_per_trial=1)

tuner.search(train_generator, epochs=50, validation_data=validation_generator)
tuner.results_summary()
