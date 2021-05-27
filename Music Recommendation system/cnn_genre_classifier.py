import json
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import pickle
import os


MODEL_NAME = "cnn_genre_classifier_model.music_classifier"
HISTORY_NAME = "history_cnn_genre_classifier.history"

def load_data(json_path):
    with open(json_path, "r") as fp:
        data = json.load(fp)

    x = data["mfcc"]
    y = data["labels"]

    x = np.array(x)
    y = np.array(y)
    return x, y

def prepare_datasets(test_size, validation_size):
    x, y = load_data(
        "/home/ritvik/Final Year Project/Testing/Music Recommendation system/data.json")
    #training and test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    #training and validation test
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train,test_size=validation_size)

    x_train = x_train[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    
    return x_train, y_train, x_validation, y_validation, x_test, y_test
    
def build_model(input_shape):
    model = keras.Sequential()
    
    #1st convolution layer
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu',input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    #2nd convolution layer
    model.add(keras.layers.Conv2D(
        32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    #3rd convolution layer
    model.add(keras.layers.Conv2D(
        32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())

    #flatting the output and feeding it into dense layer
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.4))

    #output layer

    model.add(keras.layers.Dense(10, activation='softmax'))
    
    return model


def plot_history(history):
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epochs")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epochs")
    axs[1].set_title("Loss eval")

    plt.show()

def dump(file_name,object):
    with open(file_name, 'wb ') as pickle_file:
        pickle.dump(obj=object, file=pickle_file)

def load(file_name):
    with open(file_name,'rb') as pickle_file:
        return pickle.load(pickle_file)        

x_train, y_train, x_validation, y_validation, x_test, y_test = prepare_datasets(test_size=0.3, validation_size=0.2)
input_shape = x_train.shape[1], x_train.shape[2], x_train.shape[3]

if os.path.exists(MODEL_NAME):
    model = load(MODEL_NAME)
    history = load(HISTORY_NAME)
else:
    model = build_model(input_shape=input_shape)
    #compiling the training set
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser, loss="sparse_categorical_crossentropy", metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=32, epochs=50)
    
    dump(MODEL_NAME, model)
    dump(HISTORY_NAME,history)
#evaluating the test set 
test_accuracy, test_error = model.evaluate(x_test, y_test)

print(f"Test accuracy is {test_accuracy}")

plot_history(history)
