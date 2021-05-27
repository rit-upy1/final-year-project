from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model


model = load_model('/home/ritvik/Final Year Project/fyp/model25.h5')
import os
def analyse_image(file):
    # print(file)
    true_image = image.load_img(file)
    img = image.load_img(file, grayscale=True, target_size=(48, 48))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)

    x /= 255

    custom = model.predict(x)
		    
		    # x = np.array(x, 'float32')
		    # x = x.reshape([48, 48])

		    # plt.gray()
		    # plt.imshow(true_image)
		    # plt.show()
    return emotion_analysis(custom[0])
	#returns mood


def emotion_analysis(emotions):
#     print(emotions)
	emotions=list(emotions)
	objects = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
	y_pos = np.arange(len(objects))
	#print(objects[emotions.index(max(emotions))])
	return (objects[emotions.index(max(emotions))]) ### final emotion per image 
#plt.bar(y_pos, emotions, align='center', alpha=0.5)
#plt.xticks(y_pos, objects)
#plt.ylabel('percentage')
#plt.title('emotion')
#     print(emotions[y_pos])

def get_mood(path):
	emotions={ "angry":0,  "disgusted" :0,  "fear" :0,
	"happy" :0,  "neutral" : 0,  "sad" : 0, "surprise" :0 }
	for i in os.listdir(path):
		mood = analyse_image(os.path.join(path,i))
		emotions[mood] += 1


	m=max(emotions.items(),key=lambda x:x[1])
	return m[0]
#plt.show()

