import cv2
import numpy as np
from glob import glob
import os

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import Dropout
from keras.utils import np_utils
from keras import regularizers

from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')

from keras.preprocessing.image import ImageDataGenerator

# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(200, input_dim=num_pixels, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def baseline_bp():
    # This returns a tensor
    inputs = Input(shape=(1, 28, 28))
    x = Flatten()(inputs)
    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(200, kernel_initializer='normal', activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)
    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
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
        #img_array = center_img(img_array)
        #img_array = scale_img(img_array)

        img_array = cv2.resize(img_array, (28, 28))
        x_array[idx] = img_array
    return x_array, y_array

# load data for cnn
(X_train_cnn, y_train_cnn), (X_test_cnn, y_test_cnn) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train_cnn = X_train_cnn.reshape(X_train_cnn.shape[0], 1, 28, 28).astype('float32')
X_test_cnn = X_test_cnn.reshape(X_test_cnn.shape[0], 1, 28, 28).astype('float32')

# load data for bp
(X_train_bp, y_train_bp), (X_test_bp, y_test_bp) = mnist.load_data()
num_pixels = X_train_bp.shape[1] * X_train_bp.shape[2]
X_train_bp = X_train_bp.reshape(X_train_bp.shape[0], num_pixels).astype('float32')
X_test_bp = X_test_bp.reshape(X_test_bp.shape[0], num_pixels).astype('float32')
X_train_bp = X_train_bp / 255
X_test_bp = X_test_bp / 255
y_test_bp = np_utils.to_categorical(y_test_bp)
y_train_bp = np_utils.to_categorical(y_train_bp)


# normalize inputs from 0-255 to 0-1
X_train_cnn = X_train_cnn / 255
X_test_cnn = X_test_cnn / 255
# one hot encode outputs
y_train_cnn = np_utils.to_categorical(y_train_cnn)
y_test_cnn = np_utils.to_categorical(y_test_cnn)
num_classes = y_test_cnn.shape[1]

# load my data
(x_digi, y_digi) = folder2narray('/home/grains2/Templates/tmp/mytest')

######333
for ii in range(x_digi.shape[0]):
    cv2.imwrite('/home/guest/alliance/tmp/digi/{}.png'.format(ii), x_digi[ii,:,:])

x_digi = x_digi.reshape(x_digi.shape[0], 1, 28, 28).astype('float32')
x_digi = x_digi / 255
y_digi = np_utils.to_categorical(y_digi, 10)

# data generator
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2)

# build the model
use_cnn = False
data_augment = False
if use_cnn:
    model = larger_model()
    # Fit the model
    if data_augment:
        model.fit_generator(datagen.flow(X_train_cnn, y_train_cnn, batch_size=200),
                    steps_per_epoch=50, epochs=500, validation_data=(X_test_cnn, y_test_cnn))
    else:
        model.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=500, batch_size=200, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(x_digi, y_digi, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    model.save('cnn_aug')
else:
    model_bp = baseline_bp()
    if data_augment:
        model_bp.fit_generator(datagen.flow(X_train_cnn, y_train_cnn, batch_size=200),
                    steps_per_epoch=50, epochs=500, validation_data=(X_test_cnn, y_test_cnn))
    else:
        # Fit the model
        model_bp.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=500, batch_size=200, verbose=2)
    # Final evaluation of the model
    scores = model_bp.evaluate(x_digi, y_digi, verbose=0)
    print("Baseline Error: %.2f%%" % (100-scores[1]*100))
    model_bp.save('bp')