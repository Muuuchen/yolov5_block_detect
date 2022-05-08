#这个版本已废弃!!!!!





import numpy as np
import cv2

def shrink(getpoint, n=2/4):
    x1 = int(getpoint[0][0])
    y1 = int(getpoint[0][1])
    x2 = int(getpoint[0][2])
    y2 = int(getpoint[0][3])
    x1 = x1 + (1 - n)/2 * abs(x2 - x1)
    x2 = x2 - (1 - n)/2 * abs(x2 - x1)
    y1 = y1 + (1 - n)/2 * abs(y2 - y1)
    y1 = y2 - (1 - n)/2 * abs(y2 - y1)
    return int(x1), int(x2), int(y1), int(y2)

def adaptx(x):
    return max(min(x,639),0)

def adapty(x):
    return max(min(x,479),0)

def Closest_Block(imdep,getpoint):
    x1,y1,x2,y2 = shrink(getpoint)
    if x1<0 or x1>=640 or x2<0 or x2>=640 or x1 > x2:
        x1 = adaptx(x1)
        x2 = adaptx(x2)
    if y1 < 0 or y1 >= 480 or y2 < 0 or y2 >= 480 or y1 > y2:
        y1 = adapty(y1)
        y2 = adapty(y2)
    #l=x1 r=x2 u=y1 d=y2
    Rect = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]])
    img_zero = np.zeros(imdep.shape, np.uint16)
    img_mask = cv2.fillConvexPoly(img_zero, Rect, 1)

    # print(type(imdep), type(img_mask))
    r2, c2 = np.array(imdep).shape
    a = np.mat(np.ones([1, r2]), np.uint16)
    b = np.mat(np.ones([c2, 1]), np.uint16)

    img_after_mask = np.multiply(np.mat(imdep), np.mat(img_mask))
    #print(type(np.mat(imdep).A[0][0]))
    #(type(img_after_mask.A[0][0]))
    #cv2.imshow("imdep", imdep)
    #cv2.imshow("img_after_mask", img_after_mask)
    #print("FFFFF",imdep[y1+5][x1+5],img_after_mask.A[y1+5][x1+5])
    sum = a * img_after_mask * b
    mask_sum = a * np.mat(img_mask, np.uint16) * b
    sum = sum.A[0][0] / mask_sum.A[0][0]
    #print(sum)
    return sum
