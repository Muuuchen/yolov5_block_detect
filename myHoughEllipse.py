#guang rong tui yi kan cloestBlock1
import cv2
import numpy as np

def cnt_area(cnt):
    area = cv2.contourArea(cnt)
    return area



def HoughEllipse(img, xy1, xy2, imgdep, colorflag):
    l = xy1[0]
    r = xy2[0]
    u = xy1[1]
    d = xy2[1]
# blue 1   Red 0
    RB = np.array([[[129,57,109],[179,255,255]],[[48,43,0],[144,255,255]]],dtype=np.uint8)


    # get the rectangle of block
    Rect = np.array([[l, u], [l, d], [r, d], [r, u]])
    img_zero = np.zeros(img.shape, np.uint8)
    img_roi = cv2.fillConvexPoly(img_zero, Rect, (255, 255, 255))
    img = cv2.bitwise_and(img, img_roi)

    x1 = int((l + r)/2)
    y1 = int((u + d)/2)
    p = int(imgdep[y1][x1])
    imdep1 = cv2.inRange(imgdep,max(p-150,1),p+150)

    imgdep_zero = np.zeros(imgdep.shape,np.uint8)
    imgdep_roi = cv2.fillConvexPoly(imgdep_zero, Rect, (255))
    imdep1 = cv2.bitwise_and(imdep1,imgdep_roi)
    imdep1 = cv2.erode(imdep1, (3, 3), iterations=1)
    imdep1 = cv2.dilate(imdep1, (21, 21), iterations=1)

    # image processing
    Gaus = cv2.GaussianBlur(img, (7, 7), 0)
    imghsv = cv2.cvtColor(Gaus, cv2.COLOR_BGR2HSV)

    # blue hsv h(78,116) s(113,255) v(38,255)
    low_hsv_b = RB[colorflag][0]
    high_hsv_b = RB[colorflag][1]
    mask = cv2.inRange(imghsv, low_hsv_b, high_hsv_b)
    #cv2.imshow("Blue Mask", mask)
    #cv2.imshow("depmask",imdep1)
    mask = cv2.bitwise_xor(mask,imdep1)

    erosion = cv2.erode(mask, (3, 3), iterations=1)
    dilation = cv2.dilate(erosion, (3, 3), iterations=1)
    mask = dilation

    #将四周往里面缩一点
    Rect1 = np.array([[l, u], [l, u+5], [r, u+5], [r, u]])
    Rect2 = np.array([[l, d-5], [l, d], [r, d], [r, d-5]])
    Rect3 = np.array([[l, u], [l, d], [l+5, d], [l+5, u]])
    Rect4 = np.array([[r-5, u], [r-5, d], [r, d], [r, u]])
    mask = cv2.fillConvexPoly(mask, Rect1, (0, 0, 0))
    mask = cv2.fillConvexPoly(mask, Rect2, (0, 0, 0))
    mask = cv2.fillConvexPoly(mask, Rect3, (0, 0, 0))
    mask = cv2.fillConvexPoly(mask, Rect4, (0, 0, 0))
    #
    cv2.imshow("Final Mask",mask)
    #
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cnt_area, reverse=True)
    cv2.drawContours(img, contours, -1, (0, 255, 0))
    cont = contours[0]
    #如果有白色部分，那么找重心并画框
    flag = 0
    cX = 0
    cY = 0
    if (cnt_area(cont)*8>(r-l)*(d-u)):
        flag = 1
        M = cv2.moments(cont)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)

    #椭圆拟合
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=cnt_area, reverse=True)
    # minVar = float("inf")
    # minPara = (((l + r) / 2, (u + d) / 2), (r - l, d - u), 360)
    # if not len(contours):
    #     return minPara, 0, 0, 0, 0
    # num_ellipse = 0
    # s_sum = 0
    # for cont in contours:
    #     if len(cont) < 6:
    #         continue
    #     ell = cv2.fitEllipse(cont)
    #     s1 = cv2.contourArea(cont)
    #     s_sum += s1
    #     s2 = math.pi * ell[1][0] * ell[1][1]
    #     if s1 / (s2 + 0.001) > 0.5 and s1 / (s2 + 0.001) < 1.5 and len(cont) > 50:
    #         minPara = ell
    #         num_ellipse += 1
    #         break
    # if num_ellipse == 0:
    #     if s_sum / abs(r - l) * abs(d - u) < 0.2:
    #         print("back")
    #     else:
    #         print("side_side")
    #
    # Para_ab = [minPara[0][0], minPara[0][1]]
    # mx = Para_ab[0]
    # my = Para_ab[1]
    # mx = int(mx)
    # my = int(my)
    # if mx < 0 or mx >= 640 or my < 0 or my >= 480 or mx < xy1[0] or mx > xy2[0] or my < xy1[1] or my > xy2[1]:
    #     flag = 0
    # ## bug
    # else:
    #     if mx >= 640 or my >= 480 or mx < 0 or my < 0:
    #         flag = 0
    #         return Para_ab, flag, mx, my, num_ellipse
    #     flag = 1
    #     cv2.circle(img, (int(Para_ab[0]), int(Para_ab[1])), 5, (0, 0, 255), 5)
    cv2.rectangle(img, (l,u) , (r,d) , (255,0,255), 2, cv2.LINE_AA)
    cv2.imshow("roi", img)
    #flag 是否有中心 ，cX，cY中心点的坐标
    return flag, cX, cY


def cal_angle(para):
    pass
