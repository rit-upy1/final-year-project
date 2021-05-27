import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Activation, Dropout, Flatten
import numpy as np
from kerastuner import HyperParameters
from kerastuner.tuners import BayesianOptimization,RandomSearch
import os
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


num_classes = 7  # angry, disgust, fear, happy, sad, surprise, neutral
batch_size = 256
epochs = 25
def build_model(hp):
#construct CNN structure
    model = Sequential()

    #1st convolution layer
    model.add(Conv2D(hp.Int('input_units', min_value=32,max_value=1024,step=32), (5, 5), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    #2nd convolution layer
    model.add(Conv2D(hp.Int('2nd_layer_units',min_value=32,max_value=1024,step=32), (3, 3), activation='relu'))
    model.add(Conv2D(hp.Int('2nd_layer_units', min_value=32,max_value=1024, step=32), (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    #3rd convolution layer
    model.add(Conv2D(hp.Int('3rd_layer_units', min_value=32,max_value=1024, step=32), (3, 3), activation='relu'))
    model.add(Conv2D(hp.Int('3rd_layer_units', min_value=32,max_value=1024, step=32), (3, 3), activation='relu'))
    model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())

    #fully connected neural networks
    model.add(Dense(hp.Int('output_layer_units', min_value=32,max_value=1024, step=32), activation='relu'))
    model.add(Dropout(hp.Float('3rd_layer_units',min_value=0.05, max_value=0.95, step=0.05)))
    model.add(Dense(hp.Int('output_layer_units', min_value=32,max_value=1024, step=32), activation='relu'))
    model.add(Dropout(hp.Float('3rd_layer_units',min_value=0.05, max_value=0.95, step=0.05)))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-2, 1e-3, 1e-4])), metrics=['accuracy']
                )
    return model 


with open("/home/ritvik/Final Year Project/Facial-Expressions-Recognition-master/fer2013.csv") as f:
    content = f.readlines()

lines = np.array(content)

num_of_instances = lines.size-1500
print("number of instances: ", num_of_instances)
print("instance length: ", len(lines[1].split(",")[1].split(" ")))

#------------------------------
#initialize trainset and test set
x_train, y_train, x_test, y_test = [], [], [], []

#------------------------------
#transfer train and test set data
for i in range(1, num_of_instances):
    try:
        emotion, img, usage = lines[i].split(",")

        val = img.split(" ")

        pixels = np.array(val, 'float32')

        emotion = keras.utils.to_categorical(emotion, num_classes)

        if 'Training' in usage:
            y_train.append(emotion)
            x_train.append(pixels)
        elif 'PublicTest' in usage:
            y_test.append(emotion)
            x_test.append(pixels)
    except:
        print("", end="")

#------------------------------
#data transformation for train and test sets
x_train = np.array(x_train, 'float32')
y_train = np.array(y_train, 'float32')
x_test = np.array(x_test, 'float32')
y_test = np.array(y_test, 'float32')

x_train /= 255  # normalize inputs between [0, 1]
x_test /= 255

x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_train = x_train.astype('float32')
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)
x_test = x_test.astype('float32')

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=1,
    executions_per_trial=1)
print(f'x_train {x_train}\ny_train{y_train}\nx_test{x_test}\ny_test{y_test}')
tuner.search(x_train, y_train,epochs=40, validation_data=(x_test, y_test))
tuner.results_summary()
