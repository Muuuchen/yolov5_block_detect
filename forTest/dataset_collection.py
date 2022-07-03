import cv2
import os
import datetime
import numpy as np
import pyrealsense2 as rs
import sys
print("正在初始化摄像头...")
# cap = cv2.VideoCapture(0)
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
# Start streaming
pipeline.start(config)
print("初始化成功！")

# name = str('%i'%int(100*np.random.rand(1)[0]))
name = sys.argv[1]

savedpath = r'/home/yanzhixue/PycharmProjects/RealSenser/pictures/' + name
isExists = os.path.exists(savedpath)
if not isExists:
    os.makedirs(savedpath)
    print('path of %s is build' % (savedpath))
else: print('path of %s already exist and rebuild' % (savedpath))
print("按N键拍摄图片")
# start = 0
start = int(sys.argv[2])
while (True):
    # ret, frame = cap.read()
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    # Convert images to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(color_image, 1)
    cv2.imshow('test', color_image)

    if cv2.waitKey(1) & 0xFF == ord('n'):  # 按N拍摄
        start += 1
        cv2.imwrite(savedpath + '/' + str(start) + '_ball.jpg', color_image)
        print(savedpath + '/' + str(start) + '_ball.jpg')
        cv2.namedWindow("Image")
        cv2.imshow("Image", color_image)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()