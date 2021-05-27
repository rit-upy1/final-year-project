from keras.models import load_model
model = load_model('model25.h5')


from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt



import os
image_path = "/home/samyak/final_year_project/Emotion-detection/src/data/train/sad"
count=5
for imagey in os.listdir(image_path):
    if(not count):
        break
    count-=1
#     print()
    file = image_path+"/"+imagey
    true_image = image.load_img(file)
    img = image.load_img(file, grayscale=True, target_size=(48, 48))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)

    x /= 255

    custom = model.predict(x)
    emotion_analysis(custom[0])

    x = np.array(x, 'float32')
    x = x.reshape([48, 48]);

    plt.gray()
    plt.imshow(true_image)
    plt.show()
