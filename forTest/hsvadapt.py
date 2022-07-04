import cv2
import numpy as np
import pyrealsense2 as rs
import time

def nothing(x):
    pass

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
# Start streaming
pipeline.start(config)
start = time.time()

# use track bar to perfectly define (1/2)
# the lower and upper values for HSV color space(2/2)
cv2.namedWindow("Tracking")
# 参数：1 Lower/Upper HSV 3 startValue 4 endValue
cv2.createTrackbar("LH", "Tracking", 0, 179, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 0, 179, nothing)
cv2.createTrackbar("US", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 0, 255, nothing)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_g = np.array([l_h, l_s, l_v])  # lower green value
    u_g = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_g, u_g)

    res = cv2.bitwise_and(color_image, color_image, mask=mask)  # src1,src2

    cv2.imshow("frame", color_image)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)
    key = cv2.waitKey(1)
    if key == 27:  # Esc
        break

cv2.destroyAllWindows()