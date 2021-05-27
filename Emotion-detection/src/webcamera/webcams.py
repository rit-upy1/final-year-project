import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt




from keras.models import load_model


model = load_model('/home/samyak/Documents/fyp/model25.h5')







def capture_pictures(picture_count):
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    # log.basicConfig(filename='webcam.log',level=log.INFO)

    video_capture = cv2.VideoCapture(0)
    anterior = 0
    count=0


    print("Started camera")
    while count<=picture_count:
        if not video_capture.isOpened():
            print('Unable to load camera.')
            sleep(5)
            break
        print("Running camera")
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        print("got camera",ret)
        if not ret :
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        sub_face = None



        print("got Faces")











        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            sub_face = gray[y:y+h, x:x+w]

            img_pixels = image.img_to_array(sub_face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)

            img_pixels /= 255

            predictions = model.predict(img_pixels)
 
            #find max indexed array
            max_index = np.argmax(predictions[0])
             
            emotions= ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion = emotions[max_index]
             
            cv2.putText(img, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


            
#            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
#            prediction = model.predict(cropped_img)
#            maxindex = int(np.argmax(prediction))
#            print(emotion_dict[maxindex])
#            stored_emotions[emotion_dict[maxindex]]+=1
#            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60),              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
#            print("for loop ends")
            
      
        cv2.imwrite(filename=f'image/saved_img{count}.jpg', img=sub_face)
        count+=1
    #    video_capture.release()
    #    img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
    #        img_new = cv2.imshow("Captured Image", img_new)
        cv2.waitKey(10) #waiting for 1 macro seconds for next photo. 
  
        
    print("Closing camera")
    # When everything is done, release the capture
    video_capture.release()
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    cv2.waitKey(10)
    print("Done with everything")

