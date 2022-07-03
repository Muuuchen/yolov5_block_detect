import cv2
import numpy as np

imshowFlag = True

def adaptx(x):
    return max(min(x,639),0)
def adapty(x):
    return max(min(x,479),0)

def myShow(Colormask=None, imdep1=None, mask=None):
    if imshowFlag and not (Colormask == None).any(): cv2.imshow("Color Mask", Colormask)
    if imshowFlag and not (imdep1 == None).any(): cv2.imshow("depmask",imdep1)
    if imshowFlag and not (mask == None).any(): cv2.imshow("mask", mask)

def Closest_Block(imgdep, img, getpoint,colorflag):
    '''

    :param imgdep:
    :param img:
    :param getpoint: center in white
    :param colorflag:
    :return:    sum1: depth
                flag_temp: if 0, continue in the loop
                cX, cY: the center of the white block
                isside_temp
    '''
    l = adaptx(int(getpoint[0][0]))
    u = adapty(int(getpoint[0][1]))
    r = adaptx(int(getpoint[0][2]))
    d = adapty(int(getpoint[0][3]))
    if ((r-l)*(d-u)<=30*30):
        return float("inf"), 0, (l+r)/2,(u+d)/2, 0

    # colorflag: Red 0 blue 1
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
    imdep1 = cv2.inRange(imgdep,max(p-150,1),p+150)
    imgdep_roi = cv2.fillConvexPoly(np.zeros(imgdep.shape,np.uint8), Rect, (255))
    imdep1 = cv2.bitwise_and(imdep1,imgdep_roi)
    imdep1 = cv2.erode(imdep1, (3, 3), iterations=1)
    imdep1 = cv2.dilate(imdep1, (21, 21), iterations=1)


    Colormask = cv2.bitwise_or(cv2.inRange(imghsv, R1[0], R1[1]), cv2.inRange(imghsv, R2[0], R2[1])) \
        if not colorflag else cv2.inRange(imghsv, B[0], B[1])
    mask = cv2.bitwise_xor(Colormask,imdep1)
    erosion = cv2.erode(mask, (3, 3), iterations=1)
    dilation = cv2.dilate(erosion, (3, 3), iterations=1)
    mask = dilation

    # 将四周往里面缩一点
    Rect1 = np.array([[l, u], [l, u+5], [r, u+5], [r, u]])
    Rect2 = np.array([[l, d-5], [l, d], [r, d], [r, d-5]])
    Rect3 = np.array([[l, u], [l, d], [l+5, d], [l+5, u]])
    Rect4 = np.array([[r-5, u], [r-5, d], [r, d], [r, u]])
    mask = cv2.fillConvexPoly(mask, Rect1, (0, 0, 0))
    mask = cv2.fillConvexPoly(mask, Rect2, (0, 0, 0))
    mask = cv2.fillConvexPoly(mask, Rect3, (0, 0, 0))
    mask = cv2.fillConvexPoly(mask, Rect4, (0, 0, 0))

    cX = (l+r)/2
    cY = (u+d)/2

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return float("inf"),0,cX,cY,1
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    myShow(Colormask=Colormask, imdep1=imdep1, mask=img)

    cont = contours[0]
    if (cv2.contourArea(cont) * 8 > (r-l) * (d-u)):
        M = cv2.moments(cont)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
        if imgdep[cY][cX]!=0:
            return imgdep[y1][x1], 1,cX,cY,0
        else:
            return float("inf"),0,cX,cY,0
    elif imgdep[y1][x1]!=0:
        return imgdep[y1][x1],1,cX,cY,1
    else:
        return float("inf"),0,cX,cY,1