import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras

from keras.models import load_model
model = load_model('model25.h5')

# model.load_weights("really.h5")

def display(frame):
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    stored_emotions={ "Angry":0,  "Disgusted" :0,  "Fearful" :0,  "Happy" :0,  "Neutral" : 0,  "Sad" : 0, "Surprised" :0 }
    # start the webcam feed
    # Find haar cascade to draw bounding box around face
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(gray)
    faces = facecasc.detectMultiScale(gray,scaleFactor=5, minNeighbors=5)
    print(faces)
    #print(f"running for loop {faces}")
    for (x, y, w, h) in faces:

        print(f"inside for loop {faces}")
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        print("ritvik",prediction)
        maxindex = int(np.argmax(prediction))
        # print("ritvik" +prediction)
        print("ritvik yes " ,maxindex)

        print(emotion_dict[maxindex])
        stored_emotions[emotion_dict[maxindex]]+=1
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame,(140,150),interpolation = cv2.INTER_CUBIC))
    cv2.waitKey(2500)
    m=max(stored_emotions.items(),key=lambda x:x[1])

    print(stored_emotions)
    print(m)
    return m[0]


#frame = cv2.imread("/home/samyak/final_year_project/Emotion-detection/src/data/train/angry/im27.png")
import os
image_path = "data/train/happy"
count=5
for image in os.listdir(image_path):
    if(not count):
        break
    count-=1
    frame = cv2.imread(image_path+"/"+image)
    display(frame)
