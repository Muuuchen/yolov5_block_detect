import math
import cv2
import numpy as np

def CalVariance(para, actual):
    x = para[0][0]
    y = para[0][1]
    a = para[1][0]
    b = para[1][1]
    theta = (para[2]/180)*math.pi

    if a< 30 or b < 30:
        return float("inf")
    A = (a**2)*(math.sin(theta)**2)+(b**2)*(math.cos(theta)**2)
    B = 2*(a**2-b**2)*math.sin(theta)*math.cos(theta)
    C = (a**2)*(math.cos(theta)**2)+(b**2)*(math.sin(theta)**2)
    f = -a**2*b**2
    mVar = []
    for each in actual:
        ax = each[0][0]
        ay = each[0][1]
        Var = (A * (ax - x) ** 2 + B * (ax - x) * (ay - y) + C * (ay - y) ** 2 + f)/(A*C)
        mVar.append(Var)
    return abs(np.mean(mVar))


def HoughEllipse(img,xy1,xy2):
    l = xy1[0]
    r = xy2[0]
    u = xy1[1]
    d = xy2[1]

    #get the rectangle of block
    Rect = np.array([[l,u],[l,d],[r,d],[r,u]])
    img_zero = np.zeros(img.shape,np.uint8)
    img_roi = cv2.fillConvexPoly(img_zero, Rect, (255,255,255))
    img = cv2.bitwise_and(img,img_roi)

    #image processing
    Gaus = cv2.GaussianBlur(img, (7,7), 0)
    imghsv = cv2.cvtColor(Gaus, cv2.COLOR_BGR2HSV)

    low_hsv1 = np.array([0,0, 134], dtype=np.uint8)
    upper_hsv1 = np.array([180, 30, 255], dtype=np.uint8)

    low_hsv2 = np.array([21, 43, 46], dtype=np.uint8)
    upper_hsv2 = np.array([38, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(imghsv, low_hsv1, upper_hsv1)
    mask2 = cv2.inRange(imghsv, low_hsv2, upper_hsv2)
    mask = cv2.bitwise_or(mask1,mask2)

    erosion = cv2.erode(mask, (9, 9), iterations=1)
    dilation = cv2.dilate(erosion, (9, 9), iterations=1)

    #
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=len)
    minVar =  float("inf")
    minPara = (((l+r)/2,(u+d)/2), (r-l,d-u),360)
    if not len(contours):
        return minPara, 0, 0, 0

    maxcon = contours[-1]
    if len(maxcon) < 6:
        return minPara, 0, 0, 0


    para = cv2.fitEllipse(maxcon)
    minPara = para
    Para = [minPara[0][0], minPara[0][1]]
    mx = Para[0]
    my  =Para[1]
    mx = int(mx)
    my = int(my)
    if mx < 0 or mx >= 640 or my < 0 or my >= 480 or mx < xy1[0] or mx > xy2[0] or my < xy1[1] or my > xy2[1]:
        flag = 0
    ## bug
    else:
        if mx >= 640 or my >= 480 or mx < 0 or my < 0:
            flag = 0
            return minPara, flag,mx,my
        flag = mask[my][mx]
        cv2.circle(img, (int(Para[0]), int(Para[1])), 5, (0, 0, 255), 5)
        # cv2.imshow("mask",mask)
        # cv2.imshow("roi", img)
    return minPara, flag,mx,my

def cal_angle(para):
    pass
