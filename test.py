from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.io import imsave
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from PIL import Image
import os
import h5py
import sys

# Load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("./Models/color_tensorflow_end.h5")
print("Loaded model from disk..")

loaded_model.compile(optimizer='adam', loss='mse')

def train():
    #Load Train32
    for filename in os.listdir(sys.argv[1]):
        image = img_to_array(load_img(sys.argv[1]+filename))
        image = np.array(image, dtype=float)
        X = rgb2lab(1.0/255*image)[:,:,0]
        X = X.reshape(1, 32, 32, 1)

        #open image with Image object for resize later
        img = Image.open(sys.argv[1]+filename)

        #Prdict
        output = loaded_model.predict(X)
        output *= 128


        #resize black and white photo to 96X96X1
        new_img = img.resize((96, 96), Image.ANTIALIAS)
        new_img = np.array(new_img, dtype=float)
        X_96 = rgb2lab(1.0/255*new_img)[:,:,0]
        X_96= X_96.reshape(1, 96, 96, 1)


        # Output colorizations
        cur = np.zeros((96, 96 ,3))
        cur[:,:,0] = X_96[0][:,:,0]
        cur[:,:,1:] = output[0]

        #Prepare photo to evaluate
        Y = rgb2lab(1.0/255*lab2rgb(cur))[:,:,1:]
        Y /= 128
        Y = Y.reshape(1, 96, 96, 2)
        imsave("./Results/ResultImg_"+filename+".jpg", lab2rgb(cur))
        print(loaded_model.evaluate(X, Y, batch_size=1))

def main():
    train()

main()
