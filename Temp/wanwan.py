import cv2
import numpy as np
img1 = cv2.imread(r"D:\_A_CS\detect\Temp\1.jpg")

#img1 = cv2.resize(img1,(640,480))
img2 = np.zeros(img1.shape,np.uint8)
img3 = cv2.fillConvexPoly(img2, np.array([[0,0],[0,100],[100,100],[100,0]]), (255,255,255))
img = cv2.bitwise_and(img1,img3)
# cv2.imshow("0",img)
# cv2.waitKey(1000000)