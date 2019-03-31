
import os
import sys
import json
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, Model, load_model

sentiment = [0.7, 0.7]
length = 60
FPS = 30
NEG_COLOR = np.array([1, 0.2, 0.9])
POS_COLOR = np.array([0.7, 0.9, 0.2])

def change_noise(noise):
    inc = 0.0001
    rand_idx = int(np.random.normal(0, noise.shape[0], 1))
    noise[0, rand_idx] += inc
    return noise

def tile_image(inputArray, cols, rows):
	tile = Image.fromarray(inputArray, 'RGB')
	width, height = tile.size
	complete = Image.new("RGB", (height*rows,width*cols))
	for row in range(0, rows):
		for col in range(0, cols):
			complete.paste(tile, (row*height, col*width))
	return np.array(complete)

def gen_video():
    
    latent_dim = 100
    print("loading model")
    model = load_model('model.h5')

    print("opening video to write")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #out = cv2.VideoWriter('output.avi',fourcc, FPS, (640,480))
    #fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter("output.avi", fourcc, 30.0, (640,480))

    pos_weight = max((sentiment[0] - 0.5), 0)*2
    neg_weight = max((1 - sentiment[0] - 0.5), 0)*2
    sent_color = NEG_COLOR * neg_weight + POS_COLOR * pos_weight
    noise_vec = np.random.normal(0, 1, (1, latent_dim))

    for i in range(FPS * length):
        noise_vec = change_noise(noise_vec)
        gen_img = model.predict(noise_vec)
        gen_img = gen_img[0]
        print(gen_img)
        gen_img = (0.5 * gen_img + 0.5)*255
        it = np.nditer(gen_img, flags=['multi_index'],op_flags=['readwrite'])
        for x in range(gen_img.shape[0]):
            for y in range(gen_img.shape[1]):
                gen_img[x,y]*=sent_color
                #print(gen_img[x,y].shape)
                #gen_img[x,y,0]=int(gen_img[x,y,0])
                #gen_img[x,y,1]=int(gen_img[x,y,1])
                #gen_img[x,y,2]=int(gen_img[x,y,2])

        #print(noise_vec)
        #print(gen_img)

        tiled = tile_image(gen_img, 12, 16)
        cv2.imshow("image", tiled)
        #cv2.imwrite("image.jpg", tiled)
        #print(tiled.shape)
        out.write(tiled)
        #print("image")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            continue

    out.release()


        




if __name__ == "__main__":
    gen_video()