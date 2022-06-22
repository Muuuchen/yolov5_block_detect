import math
import cv2
import numpy as np

def CalVariance(para, actual):#ming tian shishi jiaodain ju li de wucha gusuan fangshi
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

#blue hsv h(78,116) s(113,255) v(38,255)
    low_hsv_b = np.array([78, 113,38], dtype=np.uint8)
    high_hsv_b = np.array([116, 255,255], dtype=np.uint8)
    mask = cv2.inRange(imghsv, low_hsv_b, high_hsv_b)
    #mask = cv2.bitwise_not(mask)
    erosion = cv2.erode(mask, (9, 9), iterations=1)
    dilation = cv2.dilate(mask, (9, 9), iterations=1)
    #
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=len,reverse=True)
    cv2.drawContours(img,contours,-1,(0,255,0))
    minVar =  float("inf")
    minPara = (((l+r)/2,(u+d)/2), (r-l,d-u),360)
    if not len(contours):
        return minPara, 0, 0, 0, 0
    num_ellipse = 0
    s_sum = 0
    for cont in contours:
        if len(cont) < 6:
            continue
        ell = cv2.fitEllipse(cont)
        s1 = cv2.contourArea(cont)
        s_sum+=s1
        s2 = math.pi * ell[1][0] * ell[1][1]
        if s1/(s2+0.001)>0.5 and s1/(s2+0.001)<1.5 and len(cont)>50:
            minPara = ell
            num_ellipse += 1
            break
    if num_ellipse == 0:
        if s_sum/abs(r-l)*abs(d-u)<0.2:
            print("back")
        else:
            print("side_side")
        
    Para_ab = [minPara[0][0], minPara[0][1]]
    mx = Para_ab[0]
    my  =Para_ab[1]
    mx = int(mx)
    my = int(my)
    if mx < 0 or mx >= 640 or my < 0 or my >= 480 or mx < xy1[0] or mx > xy2[0] or my < xy1[1] or my > xy2[1]:
        flag = 0
    ## bug
    else:
        if mx >= 640 or my >= 480 or mx < 0 or my < 0:
            flag = 0
            return Para_ab, flag,mx,my,num_ellipse
        flag = 1
        cv2.circle(img, (int(Para_ab[0]), int(Para_ab[1])), 5, (0, 0, 255), 5)
        cv2.imshow("mask",mask)
        cv2.imshow("roi", img)
    return Para_ab, flag,mx,my,num_ellipse

def cal_angle(para):
    pass
