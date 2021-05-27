import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense,LSTM,Dropout,BatchNormalization
import matplotlib.pyplot as plt
import os
import pickle
import librosa
import song
from kerastuner.tuners import RandomSearch,BayesianOptimization
from kerastuner.engine.hyperparameters import HyperParameters
import time
import pickle

LOG_DIR = f"{int(time.time())}"



# ** changing warning level to 3 to remove the tensorflow warnings and logs **
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

DATA_PATH = "data.json"
path_of_song = "/home/ritvik/Music/Doom Eternal OST - The Only Thing they Fear is You (Mick Gordon).mp3"
epochs = 30


def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs

        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.

    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split

    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=validation_size)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(hp):
    """Generates RNN-LSTM model

    :param input_shape (tuple): Shape of input set
    :return model: RNN-LSTM model
    """

    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(LSTM(hp.Int("input_units",min_value = 32,max_value = 512,step = 32),
                                input_shape=(130,13), return_sequences=True))
    model.add(LSTM(hp.Int("input_units", min_value=32, max_value=512, step=32)))
    model.add(BatchNormalization())

    for i in range(hp.Int("n_layers", 1, 10)):
        model.add(Dense(hp.Int(f"dense_{i}_layers",min_value=16,max_value=512,step = 32), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float(f"dropout_{i}_layers",min_value=0.05,max_value = 0.9,step = 0.5)))
        

    # output layer
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def predict(song_path, sr=22050, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    signal, sr = librosa.load(song_path, duration=30, sr=sr)
    mfcc = librosa.feature.mfcc(
        signal, sr=sr, n_fft=n_fft, n_mfcc=num_mfcc, hop_length=hop_length)
    mfcc = mfcc.T
    mfcc = mfcc[np.newaxis, ...]
    model = keras.models.load_model("model.h5")
    value = model.predict(mfcc, batch_size=32)
    predicted_index = np.argmax(value, axis=1)
    predicted_index = predicted_index[0]
    label = [
        "rock",
        "reggae",
        "country",
        "pop",
        "disco",
        "jazz",
        "blues",
        "metal",
        "classical",
        "hiphop"
    ]
    return label[predicted_index]


def get_key(val, dict):
    for key, value in dict.items():
        if val == value:
            return key

    return None



    # get train, validation, test splits
X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(
    0.25, 0.2)
print(f'Shape of X_train is {X_train.shape}')

# create network
input_shape = (X_train.shape[1], X_train.shape[2])  # 130, 13

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=2,
    executions_per_trial=1,
    directory=LOG_DIR)

tuner.search(X_train,y_train,batch_size = 16,epochs=epochs,validation_data = (X_validation,y_validation))

tuner.results_summary()

models = tuner.get_best_models(num_models=2)

for i in range(len(models)):
    models[i].save_weights(f'model{i}.h5')









    # # compile model
    # optimiser = keras.optimizers.Adam(learning_rate=0.001)
    # model.compile(optimizer=optimiser,
    #               loss='sparse_categorical_crossentropy',
    #               metrics=['accuracy'])

    # model.summary()

    # train model
    # history = model.fit(X_train, y_train, validation_data=(
    #     X_validation, y_validation), batch_size=32, epochs=epochs)

    # model.save("model.h5")

    # # plot accuracy/error for training and validation
    # plot_history(history)

    # # evaluate model on test set
    # test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    # print('\nTest accuracy:', test_acc)

# value = predict(song_path=path_of_song)
# print(f'{value} label of song')

# #search genres_with_mood json for mood
# with open("genres_with_mood.json") as fp:
#     moods = json.load(fp)
# print(type(moods))

# mood = moods[value]
# genre = get_key(mood, moods)
# #play song based on mood
# song.play_song_based_on_genre(genre)
