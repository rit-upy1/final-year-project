import cv2
from keras.models import load_model
model = load_model('model25.h5')      
def facecrop(image):  
    facedata = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(facedata)
        # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    stored_emotions={ "Angry":0,  "Disgusted" :0,  "Fearful" :0,  "Happy" :0,  "Neutral" : 0,  "Sad" : 0, "Surprised" :0 }

    img = cv2.imread(image)

    try:
    
        minisize = (img.shape[1],img.shape[0])
        miniframe = cv2.resize(img, minisize)

        faces = cascade.detectMultiScale(miniframe)

        for f in faces:
            x, y, w, h = [ v for v in f ]
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            sub_face = img[y:y+h, x:x+w]

            
            cv2.imwrite('capture.jpg', sub_face)
            prediction = model.predict(sub_face)
            #print ("Writing: " + image)
            maxindex = int(np.argmax(prediction))
        # print("ritvik" +prediction)
            print("ritvik yes " ,maxindex)

            print(emotion_dict[maxindex])
            stored_emotions[emotion_dict[maxindex]]+=1
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


    except Exception as e:
        print (e)
    cv2.waitKey(2500)
    cv2.imshow(image, img)


if __name__ == '__main__':
    import os
    image_path = "data/train/happy"
    count=5
    for image in os.listdir(image_path):
        if(not count):
            break
        count-=1
#        frame = cv2.imread(image_path+"/"+image)
        facecrop(image_path+"/"+image)

