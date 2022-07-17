import numpy as np
import pyrealsense2 as rs
import cv2
import random as rd
import copy
import matplotlib.pyplot as plt
import matplotlib
import traceback


if __name__ == "__main__":
    pc = rs.pointcloud()
    points = rs.points()

    pipeline = rs.pipeline()  # 创建一个管道
    config = rs.config()  # Create a config并配置要流​​式传输的管道。
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming 开启流
    pipe_profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)  # 设置为其他类型的流,意思是我们允许深度流与其他流对齐
    # cap = cv2.VideoCapture(0)
    for i in range(10):
        cap2 = cv2.VideoCapture(i)
        print(cap2.isOpened(), cap2)
    cap2 = cv2.VideoCapture(8)
    print(cap2.isOpened(), cap2)
    try:
        while True:
            ret2, frame2 = cap2.read()
            frames = pipeline.wait_for_frames()  # 等待开启通道
            aligned_frames = align.process(frames)  # 将深度框和颜色框对齐
            depth_frame = aligned_frames.get_depth_frame()  # ?获得对齐后的帧数深度数据(图)
            color_frame = aligned_frames.get_color_frame()  # ?获得对齐后的帧数颜色数据(图)
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            img_color = np.asanyarray(color_frame.get_data())  # 把图像像素转化为数组
            img_depth = np.asanyarray(depth_frame.get_data())  # 把图像像素转化为数组

            xx, yy = int(640 * rd.random()), int(480 * rd.random())

            cv2.imshow("temp", img_color)
            cv2.imshow("temp2", frame2)
            cv2.waitKey(1)
    finally:
        pipeline.stop()
'''

if __name__ == "__main__":
    cap = cv2.VideoCapture(2)
    print(cap.isOpened(), cap)

    try:
        while True:
            ret2, frame2 = cap.read()
            cv2.imshow("temp2", frame2)
            cv2.waitKey(1)
    finally:
        pass
'''