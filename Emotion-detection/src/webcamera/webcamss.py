import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

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
            
        # if anterior != len(faces):
        #     anterior = len(faces)
        #     log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

        
        #print("Showing image")

        # Display the resulting frame
        cv2.imshow('Video', frame)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        #print("Showing image")
        # cv2.imwrite('capture.jpg',frame)

    #    if cv2.waitKey(2) or 0xFF == ord('s'): 
            
    #        check, frame = video_capture.read()
    #        cv2.imshow("Capturing", frame)
        cv2.imwrite(filename=f'/home/samyak/Documents/final_year_project/Emotion-detection/src/webcamera/image/saved_img{count}.jpg', img=sub_face)
        count+=1
    #    video_capture.release()
    #    img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
    #        img_new = cv2.imshow("Captured Image", img_new)
        cv2.waitKey(10) #waiting for 1 macro seconds for next photo. 
        print("Image Saved")
#        if cv2.waitKey(2) or count:
    #        break
    #    cv2.destroyAllWindows()
#        if count>10:
#            break
    #    if cv2.waitKey(1) & 0xFF == ord('q'):  ## take capture buttinh and end
    #        print("Turning off camera.")
    #        video_capture.release()
    #        print("Camera off.")
    #        print("Program ended.")
    #        cv2.destroyAllWindows()
    #        break
    #
        # Display the resulting frame
    #    cv2.imshow('Video', frame)
    print("Closing camera")
    # When everything is done, release the capture
    video_capture.release()
    cv2.waitKey(10)
    cv2.destroyAllWindows()
    cv2.waitKey(10)
    print("Done with everything")

