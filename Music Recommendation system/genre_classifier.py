import json
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

def load_data(json_path):
    with open(json_path,"r") as fp:
        data = json.load(fp)

    x = data["mfcc"]
    y = data["labels"]

    x = np.array(x)
    y = np.array(y)
    return x, y

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

#loaded data    
inputs, outputs = load_data("data.json")

#splitting into training set and testing set

inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=0.3)

#building network

model = keras.Sequential([
    #input layer
    keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
    #1st hidden layer
    keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.09),
    #2nd hidden layer
    keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.09),
    #3rd hidden layer
    keras.layers.Dense(64, activation="relu",
                       kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.09),
    #output layer
    keras.layers.Dense(10,activation="softmax")
])

optimizer = keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy']
              )

model.summary()

history = model.fit(x=inputs_train,
          y=outputs_train,
          validation_data=(inputs_test, outputs_test),
          batch_size=32,
          epochs=50)

plot_history(history)
