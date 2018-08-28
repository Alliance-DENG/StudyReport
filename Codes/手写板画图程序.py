import cv2
import numpy as np

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




drawing = False # true if mouse is pressed
ix,iy = -1,-1


def nothing(x):
    pass


# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y


    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            #cv2.circle(img,(x,y),5,255,-1)
            cv2.line(img, (ix,iy), (x,y), 255, 10)
            ix, iy = x, y


    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


# Create a black image, a window
img = np.zeros((280,280,1), np.uint8)
cv2.namedWindow('image')
#cv2.namedWindow('processed')


idx = np.random.randint(0,100000)
while(1):
    #cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    # get current positions of four trackbars
    if k == 27:
        break
    if k == 13:
        cv2.imwrite('C:\\Users\\Alliance\\Desktop\\9\\{}.png'.format(idx), img)
        
        if False:
            img = center_img(img)
            img = scale_img(img)
            cv2.imshow('processed', img)
        
        img[:] = 0
        idx += 1
    if k == 99:
        img[:] = 0
    else:
        if k == 27:
            break
        cv2.setMouseCallback('image',draw_circle,(None))
        cv2.imshow('image',img)


cv2.destroyAllWindows()