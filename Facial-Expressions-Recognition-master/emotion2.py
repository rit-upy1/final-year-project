import os
import time
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from kerastuner.tuners import BayesianOptimization 
from keras import backend as K
K.image_data_format()
#K.common.set_image_dim_ordering('th')


import gc
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

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
train_dir = 'home/data/train'
val_dir = 'home/data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 2

train_images = []
train_labels = []
shape = (48, 48)
train_path = 'data/train'
#what is next?????? we r in difficult in fac
for foldername in os.listdir(train_path):
    for filename in os.listdir(train_path+"/"+foldername):
        img = cv2.imread(os.path.join(
            train_path+"/"+foldername, filename), cv2.IMREAD_GRAYSCALE)
        # Resize all images to a specific shape
        # img = cv2.resize(img,shape)

        # Spliting file names and storing the labels for image in list
        train_labels.append(foldername)

        train_images.append(img)

# Converting labels into One Hot encoded sparse matrix
train_labels = pd.get_dummies(train_labels).values

# Converting train_images to array
train_images = np.array(train_images)
# Splitting Training data into train and validation dataset
x_train,x_test,y_train,y_test = train_test_split(train_images,train_labels,random_state=1)
# print(x_train.shape)


# print   (x_test.shape[0],x_test.shape)
x_train = x_train.reshape(x_train.shape[0], 48, 48, 1)
x_test = x_test.reshape(x_test.shape[0], 48, 48, 1)



######################
# ###########################################################################




# train_datagen = ImageDataGenerator(rescale=1./255)
# val_datagen = ImageDataGenerator(rescale=1./255)

# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(48, 48),
#     batch_size=batch_size,
#     color_mode="grayscale",
#     class_mode='categorical')

# validation_generator = val_datagen.flow_from_directory(
#     val_dir,
#     target_size=(48, 48),
#     batch_size=batch_size,
#     color_mode="grayscale",
    # class_mode='categorical')

# Create the model
def build_model(hp):
    
    model = Sequential()

    model.add(Conv2D(hp.Int('input_units', min_value=32, max_value=1024, step=32),
                     kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(hp.Int('input_units', min_value=32, max_value=1024,
                            step=32), kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(hp.Float('input_dropout',min_value=0.0,max_value=.95,step=0.05)))

    

    for i in range(hp.Int('n_layers', 1, 4)):  # adding variation of layers.
       
        model.add(Conv2D(hp.Int(f'conv_{i}_units', min_value=32, max_value=1024, step=32), kernel_size=(
            3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Conv2D(hp.Int(f'conv_{i}_units', min_value=32, max_value=1024, step=32), kernel_size=(
            3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout((hp.Float(f'dropout_{i}_units',min_value=0.0,max_value=.95,step=0.05))))
   
    model.add(Flatten())
    model.add(Dense(hp.Int('output_layer', min_value=32,max_value=1024, step=32), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])       
    return model




print('Tuning model')

tuner = BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
)
# # train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# # valid_data = tf.data.Dataset.from_tensor_slices((x_test, testVocals))
# # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
# # y_train = y_train.reshape(y_train.shape[0],y_train.shape[1])
# print(f'shape of x is {x_train.shape} and y is {y_train.shape}')
# print(x_train.shape[1], x_train.shape[2])
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# #y_train = y_train.reshape(y_train.shape[0],y_train.shape[1],y_train.shape[2],1)
# print("running tuner")
tuner.search(x=x_train,
             y=y_train,
             verbose=2, # just slapping this here bc jupyter notebook. The console out was getting messy.
             epochs=1,
             batch_size=64,
             #callbacks=[tensorboard],  # if you have callbacks like tensorboard, they go here.
             validation_data=(x_test, y_test))
tuner.results_summary()


# # If you want to train the same model or try other models, go for this
# if mode == "train":
#     model.compile(loss='categorical_crossentropy', optimizer=Adam(
#         lr=0.0001, decay=1e-6), metrics=['accuracy'])
#     model_info = model.fit(
#         train_generator,
#         steps_per_epoch=num_train // batch_size,
#         epochs=num_epoch,
#         validation_data=validation_generator,
#         validation_steps=num_val // batch_size)
#     plot_model_history(model_info)
#     model.save_weights('model.h5')

# # emotions will be displayed on your face from the webcam feed
# def display():
#     model.load_weights('model.h5')

#     # prevents openCL usage and unnecessary logging messages
#     cv2.ocl.setUseOpenCL(False)

#     # dictionary which assigns each label an emotion (alphabetical order)
#     emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
#                     3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
#     stored_emotions = {"Angry": 0,  "Disgusted": 0,  "Fearful": 0,
#                        "Happy": 0,  "Neutral": 0,  "Sad": 0, "Surprised": 0}
#     # start the webcam feed
#     cap = WebcamVideoStream(src = 0).start()
#     t1 = time.time()
#     while True:
#         # Find haar cascade to draw bounding box around face
#         ret, frame = cap.read()
#         # ret, frame = cv2.imencode('.jpg',image)
#         if not ret:
#             break
#         facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#         gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAYp)
#         faces = facecasc.detectMultiScale(
#             gray, scaleFactor=1.3, minNeighbors=5)

#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
#             roi_gray = gray[y:y + h, x:x + w]
#             cropped_img = np.expand_dims(np.expand_dims(
#                 cv2.resize(roi_gray, (48, 48)), -1), 0)
#             prediction = model.predict(cropped_img)
#             maxindex = int(np.argmax(prediction))
#             print(emotion_dict[maxindex])
#             stored_emotions[emotion_dict[maxindex]] += 1
#             cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#         cv2.imshow('Video', cv2.resize(
#             frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
#         cv2.waitKey(1)
#       #  if cv2.waitKey(1) & 0xFF == ord('q'):
#         # break
#         t2 = time.time()
#         if(round(t2-t1) == 10):
#             break
#     m = max(stored_emotions.items(), key=lambda x: x[1])
#     print(stored_emotions)
#     print(m)

#     cap.stop()
#     cv2.destroyAllWindows()

# #display()
