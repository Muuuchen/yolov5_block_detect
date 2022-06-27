import numpy as np
import pyrealsense2 as rs
import cv2
import random as rd
import copy
class myCamera():
    def __init__(self):
        self.K = np.array([[602.2484, 0, 317.7842], [0, 604.6236, 235.7784], [0, 0, 1.0]])
        self.distortion1 = 0.1006
        self.distortion2 = -0.0998
        self.temp = np.array([[[]]])

    def specify(self, u, v, Z):
        '''
        :param u: pixel
        :param v: pixel
        :param Z: meter
        :return: XYZ
        '''
        U = np.array([[u], [v], [1]])
        K1 = np.reshape(self.K[:, 0], (3,1))
        K2 = np.reshape(self.K[:, 1], (3, 1))
        K3 = np.reshape(self.K[:, 2], (3, 1))
        A = np.append(U, -K1, axis=1)
        A = np.append(A, -K2, axis=1)
        unkown = np.linalg.pinv(A).dot(K3) * Z
        return np.array([[unkown[1][0]], [unkown[2][0]], [Z]])

    def specify_norm(self, depth_frame, xx, yy):
        deep = depth_frame.get_distance(xx, yy)
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [xx, yy],
                                                            deep)
        camera_coordinate = np.array(camera_coordinate)
        dis = np.linalg.norm(camera_coordinate)
        return dis, camera_coordinate

class blockSize(myCamera):

    def blockSize(self, Point1, Point2, Ztemp):
        '''
        :param Point1: pixel
        :param Point2: pixel
        :param Ztemp: millimeter
        :return: meter
        '''
        P1 = copy.deepcopy(Point1)
        P2 = copy.deepcopy(Point2)
        Z = copy.deepcopy(Ztemp) * 1e-3
        P1 = np.reshape(P1, (2,1))
        P2 = np.reshape(P2, (2,1))
        PLU = self.specify(P1[0, 0], P1[1, 0], Z)
        PLD = self.specify(P1[0, 0], P2[1, 0], Z)
        PRU = self.specify(P2[0, 0], P1[1, 0], Z)
        PRD = self.specify(P2[0, 0], P2[1, 0], Z)
        Len = np.linalg.norm(PLU - PRU)
        Wid = np.linalg.norm(PLU - PLD)
        return max(Len, Wid)

    def blockSize2(self, Point1, Point2, Ztemp):
        '''
        :param Point1: pixel
        :param Point2: pixel
        :param Ztemp: millimeter
        :return: meter
        '''
        P1 = copy.deepcopy(Point1)
        P2 = copy.deepcopy(Point2)
        Z = copy.deepcopy(Ztemp) * 1e-3
        P1 = np.reshape(P1, (2,1))
        P2 = np.reshape(P2, (2,1))
        PL = self.specify(P1[0, 0], (P1[1, 0] + P2[1,0])/2, Z)
        PD = self.specify((P1[0,0]+P2[0, 0])/2, P2[1, 0], Z)
        PU = self.specify((P1[0,0]+P2[0, 0])/2, P1[1, 0], Z)
        PR = self.specify(P2[0, 0], (P1[1,0]+P2[1, 0])/2, Z)
        Len = np.linalg.norm(PU - PD)
        Wid = np.linalg.norm(PL - PR)

        self.temp = np.append(self.temp, np.array([[[Z,max(Len, Wid)]]]))
        np.savetxt("./test.txt", self.temp)
        if(Z>7):
            return max(Len, Wid)
        elif(Z<3.5):
            return max(Len, Wid) * (1 + Z / 3.5 * 0.1)
        else:
            return max(Len, Wid)*(1+(7-Z)/3.5 * 0.1)

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
    cap = cv2.VideoCapture(0)

    obj = myCamera()
    try:
        while True:
            frames = pipeline.wait_for_frames()  # 等待开启通道
            aligned_frames = align.process(frames)  # 将深度框和颜色框对齐
            depth_frame = aligned_frames.get_depth_frame()  # ?获得对齐后的帧数深度数据(图)
            color_frame = aligned_frames.get_color_frame()  # ?获得对齐后的帧数颜色数据(图)
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            img_color = np.asanyarray(color_frame.get_data())  # 把图像像素转化为数组
            img_depth = np.asanyarray(depth_frame.get_data())  # 把图像像素转化为数组

            xx, yy = int(640 * rd.random()), int(480 * rd.random())

            dis, camera_coordinate = obj.specify_norm(depth_frame, xx, yy)

            my_coor = obj.specify(xx, yy, camera_coordinate[2])
            # print(img_depth[yy][xx]) ## mm
            print(camera_coordinate, my_coor) ## m
            # cv2.imshow("temp", color_frame)
            cv2.imshow("temp2", img_color)
            cv2.waitKey(1)
    finally:
        pipeline.stop()


