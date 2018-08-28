import cv2
import numpy as np
from glob import glob
import os

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from keras import regularizers

from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')

from keras.preprocessing.image import ImageDataGenerator
import keras

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(200, input_dim=num_pixels, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def cal_central(img):
    width, height = img.shape[0], img.shape[1]
    point_list = []
    for x in range(width):
        for y in range(height):
            if img[x,y].all():
                point_list.append((x,y))
    c_x, c_y = 0, 0
    for each in point_list:
        c_x += each[0]
        c_y += each[1]
    c_x, c_y = c_x//len(point_list), c_y//len(point_list)
    return c_x, c_y

def scale_img(img):
    # get the box of the front
    width, height = img.shape[0], img.shape[1]
    point_list = []
    x_min, y_min, x_max, y_max = width, height, -1 , -1
    for x in range(width):
        for y in range(height):
            if img[x,y] != 0:
                if x_min > x:
                    x_min = x
                if y_min > y:
                    y_min = y
                if x_max < x:
                    x_max = x
                if y_max < y:
                    y_max = y
    
    resize_front = np.zeros((200, 200, 1), np.uint8)
    factor = 200 / max((x_max-x_min), (y_max-y_min) )
    # it is weird that sometime factor is less than 0
    if factor > 0:
        resize_front = cv2.resize(img[x_min:x_max+1:1, y_min:y_max+1:1], None, fx=factor, fy=factor )
        x_boarder, y_boarder = (280 - resize_front.shape[0]) // 2, (280 - resize_front.shape[1]) // 2
        final_img = cv2.copyMakeBorder(resize_front, x_boarder, x_boarder, y_boarder, y_boarder, cv2.BORDER_CONSTANT, value=0)
        return final_img
    else:
        return img

def center_img(img):
    c_x, c_y = cal_central(img)
    dx, dy = img.shape[0]//2 - c_x, img.shape[1]//2 - c_y
    M = np.float32([[1,0,dy],[0,1,dx]])
    width, height = img.shape[0], img.shape[1]
    # do image shifting
    dst = cv2.warpAffine(img, M, (width, height))
    return dst

def folder2narray(filepath):
    x_filename = []
    y = []
    for i in range(10):
        file_list = glob(filepath + '/%s/*.png' % i)
        for each in file_list:
            y.append(i)
            x_filename.append(each)
    y_array = np.array(y)
    x_array = np.zeros((len(x_filename), 28, 28), np.uint8)
    for idx, each in enumerate(x_filename):
        img_array = cv2.imread(each)
        # switch to single channel
        img_array = img_array[:,:,0]
        # you can add some prepocession here
        img_array = center_img(img_array)
        img_array = scale_img(img_array)

        img_array = cv2.resize(img_array, (28, 28))
        x_array[idx] = img_array
    return x_array, y_array


# load my data
(x_digi, y_digi) = folder2narray('/home/grains2/Templates/tmp/mytest')


x_digi = x_digi.reshape(x_digi.shape[0], 1, 28, 28).astype('float32')
x_digi = x_digi / 255
y_digi = np_utils.to_categorical(y_digi, 10)

print('cnn')
model_bp = keras.models.load_model('/home/grains2/Templates/tmp/cnn')
scores = model_bp.evaluate(x_digi, y_digi, verbose=0)
print("Baseline Error: %.2f%%" % (scores[1]*100))

print('cnn_aug')
model_bp = keras.models.load_model('/home/grains2/Templates/tmp/cnn_aug')
scores = model_bp.evaluate(x_digi, y_digi, verbose=0)
print("Baseline Error: %.2f%%" % (scores[1]*100))

print('bp')
model_bp = keras.models.load_model('/home/grains2/Templates/tmp/bp')
scores = model_bp.evaluate(x_digi, y_digi, verbose=0)
print("Baseline Error: %.2f%%" % (scores[1]*100))

print('bp_aug')
model_bp = keras.models.load_model('/home/grains2/Templates/tmp/bp_aug')
scores = model_bp.evaluate(x_digi, y_digi, verbose=0)
print("Baseline Error: %.2f%%" % (scores[1]*100))