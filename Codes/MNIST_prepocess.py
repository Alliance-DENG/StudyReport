import numpy as np
import cv2

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
            if img[x,y].all():
                if x_min > x:
                    x_min = x
                if y_min > y:
                    y_min = y
                if x_max < x:
                    x_max = x
                if y_max < y:
                    y_max = y
    
    resize_front = np.zeros((200, 200, 1), np.uint8)
    factor = 200 // max((x_max-x_min), (y_max-y_min) )
    resize_front = cv2.resize(img[x_min:x_max+1:1, y_min:y_max+1:1, 0], None, fx=factor, fy=factor )
    x_boarder, y_boarder = (280 - resize_front.shape[0]) // 2, (280 - resize_front.shape[1]) // 2
    final_img = cv2.copyMakeBorder(resize_front, x_boarder, x_boarder, y_boarder, y_boarder, cv2.BORDER_CONSTANT, value=0)
    return final_img

def center_img(img):
    c_x, c_y = cal_central(img)
    dx, dy = img.shape[0]//2 - c_x, img.shape[1]//2 - c_y
    M = np.float32([[1,0,dy],[0,1,dx]])
    width, height = img.shape[0], img.shape[1]
    # do image shifting
    dst = cv2.warpAffine(img, M, (width, height))
    return dst

img = cv2.imread(r'C:\Users\Alliance\Desktop\1\46518.png')
img = center_img(img)
cv2.imshow('center', img)
img = scale_img(img)
cv2.imshow('center&rescale', img)
cv2.waitKey(0)