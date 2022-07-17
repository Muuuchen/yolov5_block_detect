import numpy as np
import cv2
from classes.hsvMan import hsvRange
R1 = np.array([[0, 68, 59], [20, 255, 255]], dtype=np.uint8)
R2 = np.array([[150, 68, 59], [179, 255, 255]], dtype=np.uint8)
B = np.array([[48,160, 0], [144, 255, 255]], dtype=np.uint8)

def adaptx(x): return max(min(x,639),0)

def adapty(y): return max(min(y,479),0)

def incolor(a, colorflag):
    '''
    judge if a in blue block or red block
    :param a: hsv instance of a pixel
    :param colorflag: Red 0 Blue 1
    :param RB: none
    :return:
    '''
    #R1 = hsvRange([[0, 68, 59], [20, 255, 255]])
    R1 = hsvRange([[0, 0, 0], [0, 0, 0]])
    R2 = hsvRange([[150, 68, 59], [179, 255, 255]])
    B = hsvRange([[48, 43, 0], [144, 255, 255]])
    return 1 if not colorflag and (R1.inRange(a) or R2.inRange(a)) or colorflag and B.inRange(a) else 0

def Getupstate(img, getpoint, colorflag):
    up_isside = 0
    up_isfront = 0
    up_isfivefront = 0
    up_isback =0
    up_isonefront = 0
    l = adaptx(int(getpoint[0][0]))  # leftup x
    u = adapty(int(getpoint[0][1]))  # leftup y
    r = adaptx(int(getpoint[0][2]))  # rightdown x
    d = adapty(int(getpoint[0][3]))  # rightdown y

    # RB = np.array([[[129, 57, 84], [179, 255, 255]], [[48, 43, 0], [144, 255, 255]]], dtype=np.uint8)

    Rect = np.array([[l, u], [l, d], [r, d], [r, u]])
    img_roi = cv2.fillConvexPoly(np.zeros(img.shape, np.uint8), Rect, (255, 255, 255))
    img = cv2.bitwise_and(img, img_roi)
    Gaus = cv2.GaussianBlur(img, (7, 7), 0)
    imghsv = cv2.cvtColor(Gaus, cv2.COLOR_BGR2HSV)

    x1 = int((l + r) / 2)
    y1 = int((u + d) / 2)
    Colormask = cv2.bitwise_or(cv2.inRange(imghsv, R1[0], R1[1]), cv2.inRange(imghsv, R2[0], R2[1])) \
        if not colorflag else cv2.inRange(imghsv, B[0], B[1])
    mask = Colormask
    erosion = cv2.erode(mask, (3, 3), iterations=1)
    dilation = cv2.dilate(erosion, (3, 3), iterations=1)
    mask = dilation
    up_isside = mask[y1][x1] == 255

    if mask[y1][x1] == 0:
        rsum = 1e-10
        rsumc = 0
        for i in range(r - x1):
            rsum += 1
            rsumc += incolor(imghsv[y1][x1+i],colorflag)

        lsum = 1e-10
        lsumc = 0
        for i in range(x1 - l):
            lsum += 1
            lsumc += incolor(imghsv[y1][x1-i],colorflag)
        if(rsumc/rsum > 0.05  or lsumc / lsum > 0.05):
            up_isfront = 1
        else:
            up_isback = 1
        if((rsumc/rsum > 0.5  or lsumc / lsum > 0.5)):
            up_isfront = 2
    return up_isside,up_isfront,up_isback,abs((r-l)*(d-u)), \
           1 if abs(d-u)/(abs(r-l)+1e-4) < 0.65 else 0, 1 if (abs(d-u)/(abs(r-l)+1e-4))>1.1 else 0 #  stand stand
