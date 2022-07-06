import cv2
import numpy as np
from matplotlib import pyplot as plt
kernel = np.ones((5, 5), np.uint8)
img = cv2.imread(r'C:\Users\hzysdbybyd131\Desktop\detect\img\24.jpg')
img = cv2.resize(img, None, fx = 0.2, fy = 0.2)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[20]
M = cv2.moments(cnt)
perimeter = cv2.arcLength(cnt,True)
print(perimeter)

epsilon = 0.1*cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt, epsilon, True)#多边形拟合
# cnt = contours[4]
# cv.drawContours(img, [cnt], 0, (0,255,0), 3)
# cv2.imshow('thresh', thresh)
# cv2.imshow('img', img)
# k = cv2.waitKey(0) & 0xFF
# if k == 27:
#     cv2.destroyAllWindows()


