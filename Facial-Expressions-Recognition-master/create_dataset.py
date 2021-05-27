import os
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

from kerastuner.tuners import BayesianOptimization
from gc import collect

import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from imutils.video import WebcamVideoStream
print("Passsed angry")

print("Passsed tensorflow")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Passsed imports")
# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
mode = ap.parse_args().mode

# plots accuracy and loss curves


def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['accuracy'])+1),
                model_history.history['accuracy'])
    axs[0].plot(range(1, len(model_history.history['val_accuracy'])+1),
                model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy'])+1),
                      len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss'])+1),
                model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss'])+1),
                model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(
        1, len(model_history.history['loss'])+1), len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()


# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 2


############################

train_images = []
train_labels = []
shape = (48, 48)
train_path = 'data/train'
df = pd.DataFrame(columns=['image', 'label'])

for foldername in os.listdir(train_path):
    path = os.path.join(train_path,foldername)
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
        # Resize all images to a specific shape
        # img = cv2.resize(img,shape)
        zeros = np.zeros((7,))
        moods = os.listdir(train_path)
        index = moods.index(foldername)
        zeros[index] = 1
        df = df.append({'image':img,'label':zeros},ignore_index = True)
        # Spliting file names and storing the labels for image in list
        train_labels.append(foldername)

        train_images.append(img)
        collect()
# Converting labels into One Hot encoded sparse matrix
train_labels = pd.get_dummies(train_labels).values

# Converting train_images to array
train_images = np.array(train_images)

no_of_classes = len(train_labels[0])

df.to_csv('images_with_label.csv')
