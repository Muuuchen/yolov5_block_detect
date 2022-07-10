import traceback

import cv2, math
import numpy as np
from init import imshowFlag

def adaptx(x): return max(min(x,639),0)

def adapty(y): return max(min(y,479),0)

# Histogram equalization
def histequ(gray, nlevels=256):
    # Compute histogram
    histogram = np.bincount(gray.flatten(), minlength=nlevels)
    # print ("histogram: ", histogram)

    # Mapping function
    uniform_hist = (nlevels - 1) * (np.cumsum(histogram)/(gray.size * 1.0))
    uniform_hist = uniform_hist.astype('uint8')
    # print ("uniform hist: ", uniform_hist)

    # Set the intensity of the pixel in the raw gray to its corresponding new intensity
    height, width = gray.shape
    uniform_gray = np.zeros(gray.shape, dtype='uint8')  # Note the type of elements
    for i in range(height):
        for j in range(width):
            uniform_gray[i,j] = uniform_hist[gray[i,j]]

    return uniform_gray

def myShow(imgdep=None, Colormask=None, imdep1=None, dilation=None, img=None):
    if imshowFlag and not (imgdep == None).any(): cv2.imshow("imgdep", histequ(imgdep))
    if imshowFlag and not (Colormask == None).any(): cv2.imshow("Color Mask", Colormask)
    if imshowFlag and not (imdep1 == None).any(): cv2.imshow("depmask",imdep1)
    if imshowFlag and not (dilation == None).any(): cv2.imshow("dilation", dilation)
    if imshowFlag and not (img == None).any(): cv2.imshow("imgCB", img)

def MSE(tensor):
    try:
        arr = np.array(tensor)
        num = 1
        shape = np.shape(arr)
        for i in range(len(shape)): num *= shape[i]
        averArr = np.ones(shape) * np.average(arr)

        return np.linalg.norm(arr - averArr) / num
    except:
        traceback.print_exc()
        return float("inf")

def Closest_Block(imgdep, img, getpoint,colorflag):
    '''

    :param imgdep: millimeter
    :param img:
    :param getpoint: center in white
    :param colorflag: Red 0 Blue 1
    :return:    sum1: depth
                flag_temp: if 0, continue in the loop
                cX, cY: the center of the white block
                isside_temp: -1 means side lie down, 1 menas side stand, 0 else
                mask
    '''
    l = adaptx(int(getpoint[0][0])) # leftup x
    u = adapty(int(getpoint[0][1])) # leftup y
    r = adaptx(int(getpoint[0][2])) # rightdown x
    d = adapty(int(getpoint[0][3])) # rightdown y
    if ((r-l)*(d-u)<=30*30):
        return float("inf"), 0, (l+r)/2,(u+d)/2, 0, img

    # RB = np.array([[[129, 57, 84], [179, 255, 255]], [[48, 43, 0], [144, 255, 255]]], dtype=np.uint8)
    R1 = np.array([[0, 68, 59], [20, 255, 255]], dtype=np.uint8)
    R2 = np.array([[150, 68, 59], [179, 255, 255]], dtype=np.uint8)
    B = np.array([[48, 43, 0], [144, 255, 255]], dtype=np.uint8)

    Rect = np.array([[l, u], [l, d], [r, d], [r, u]])
    img_roi = cv2.fillConvexPoly(np.zeros(img.shape, np.uint8), Rect, (255, 255, 255))
    img = cv2.bitwise_and(img, img_roi)
    Gaus = cv2.GaussianBlur(img, (7, 7), 0)
    imghsv = cv2.cvtColor(Gaus, cv2.COLOR_BGR2HSV)

    x1 = int((l + r)/2)
    y1 = int((u + d)/2)
    p = int(imgdep[y1][x1])
    imdep1 = cv2.inRange(imgdep,max(p-500,1),p+500)
    imgdep_roi = cv2.fillConvexPoly(np.zeros(imgdep.shape,np.uint8), Rect, (255))
    imdep1 = cv2.bitwise_and(imdep1,imgdep_roi)
    imdep1 = cv2.erode(imdep1, (3, 3), iterations=1)
    imdep1 = cv2.dilate(imdep1, (21, 21), iterations=1)


    Colormask = cv2.bitwise_or(cv2.inRange(imghsv, R1[0], R1[1]), cv2.inRange(imghsv, R2[0], R2[1])) \
        if not colorflag else cv2.inRange(imghsv, B[0], B[1])
    mask = cv2.bitwise_xor(Colormask, imdep1)
    erosion = cv2.erode(mask, (3, 3), iterations=1)
    dilation = cv2.dilate(erosion, (3, 3), iterations=1)
    mask = dilation

    # 将四周往里面缩一点
    Rect1 = np.array([[l, u], [l, u+5], [r, u+5], [r, u]])
    Rect2 = np.array([[l, d-5], [l, d], [r, d], [r, d-5]])
    Rect3 = np.array([[l, u], [l, d], [l+5, d], [l+5, u]])
    Rect4 = np.array([[r-5, u], [r-5, d], [r, d], [r, u]])
    mask = cv2.fillConvexPoly(mask, Rect1, 0)
    mask = cv2.fillConvexPoly(mask, Rect2, 0)
    mask = cv2.fillConvexPoly(mask, Rect3, 0)
    mask = cv2.fillConvexPoly(mask, Rect4, 0)

    cX = (l+r)/2
    cY = (u+d)/2

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return float("inf"),0,cX,cY,1, mask
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(img, contours, 0, (0, 255, 0), 3)
    myShow(imgdep=imgdep, Colormask=Colormask, imdep1=imdep1, dilation=dilation, img=img)

    cont = contours[0]
    norm = int(imgdep[y1][x1]) if not int(imgdep[y1][x1]) < 50 else float("inf")
    if (cv2.contourArea(cont) * 8 > (r-l) * (d-u)):
        M = cv2.moments(cont)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
        if imgdep[cY][cX]!=0:
            return norm, 1,cX,cY,0, mask
        else:
            return float("inf"),0,cX,cY,0, mask
    elif imgdep[y1][x1]!=0:
        num = 10
        tensory = []
        tensorx = []

        stampy = int((d - u) / (num + 2))
        for i in range(-num, num + 1, 1):
            if adapty(y1 + i * stampy) == 479 or adapty(y1 + i * stampy) == 0 or adaptx(x1) == 639 or adaptx(x1) == 0: continue
            if adapty(y1 + i * stampy) < u or adaptx(y1 + i * stampy) > d: continue
            if abs(int(imgdep[y1 + i * stampy][adaptx(x1)]) - norm) >= norm * 2 / 3: continue
            tensory.append(imgdep[adapty(y1 + i * stampy)][x1])

        stampx = int((r - l) / (2 * num + 10))
        for i in range(-num, num + 1, 1):
            if adapty(y1) == 479 or adapty(y1) == 0 or adaptx(x1 + i * stampx) == 639 or adaptx(x1 + i * stampx) == 0: continue
            if adapty(x1 + i * stampx) < l or adaptx(x1 + i * stampx) > r: continue
            if abs(int(imgdep[y1][adaptx(x1 + i * stampx)]) - norm) >= norm * 2 / 3: continue
            tensorx.append(imgdep[y1][adaptx(x1 + i * stampx)])
        if MSE(tensory) > MSE(tensorx): return imgdep[y1][x1],1,cX,cY,1, mask
        else: return imgdep[y1][x1], 1, cX, cY, -1, mask

        # wid < len
        if abs(getpoint[0, 2] - getpoint[0, 0]) <abs(getpoint[0, 1] - getpoint[0, 3]): return norm, 1, cX, cY, 1, mask
        else: return norm, 1, cX, cY, -1, mask


    else:
        return float("inf"),0,cX,cY,1, mask