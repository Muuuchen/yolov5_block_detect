import numpy as np
import pyrealsense2 as rs
import cv2
import random as rd
import copy
import matplotlib.pyplot as plt
import matplotlib
import traceback

# matplotlib.use('TkAgg')
distortion1 = 0.1006
distortion2 = -0.0998
normList = []
fixList = []

def adaptx(x): return max(min(x,639),0)
def adapty(y): return max(min(y,479),0)

class myCamera():
    def __init__(self):
        self.K = np.array([[602.2484, 0, 317.7842], [0, 604.6236, 235.7784], [0, 0, 1.0]])
        self.K1 = np.reshape(self.K[:, 0], (3, 1))
        self.K2 = np.reshape(self.K[:, 1], (3, 1))
        self.K3 = np.reshape(self.K[:, 2], (3, 1))
        self.invK = np.linalg.pinv(self.K)

        self.px = 640 / 2
        self.py = 480 / 2
        self.norm = []
        self.fix = []

    def L(self, X):
        """

        :param X: array(3, 1), X_dis
        :return: array(3, 1), X_undis
        """
        X_dis = copy.deepcopy(X)
        squaredRho = X_dis[0, 0] ** 2 + X_dis[1, 0] ** 2
        L_rho = 1 + distortion1 * squaredRho
        return L_rho * X_dis
        # return 1 + distortion1 * squaredRho + distortion2 * squaredRho ** 2

    def distort(self, u_dis, v_dis):
        '''

        :param u_dis:
        :param v_dis:
        :return: u_undis, v_undis
        '''
        U_dis = np.array([[u_dis], [v_dis], [1]])
        X_dis = self.invK.dot(U_dis)
        X_undis = self.L(X_dis)
        U_undis = self.K.dot(X_undis)
        return U_undis[0, 0] / U_undis[2, 0], U_undis[1, 0] / U_undis[2, 0]

    def specify(self, u, v, Z):
        '''
        :param u: pixel
        :param v: pixel
        :param Z: meter
        :return: XYZ meter
        '''
        U = np.array([[u], [v], [1]])
        A = np.append(U, -self.K1, axis=1)
        A = np.append(A, -self.K2, axis=1)
        unkown = np.linalg.pinv(A).dot(self.K3) * Z
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

    def save(self, normElement, fixElement):
        print(normElement, fixElement)
        normList.append(normElement)
        fixList.append(fixElement)
        # blockSize_obj.tempShow(tempList)
        np.savetxt("./datatest/X_liedown_norm.txt", np.asarray(normList), fmt='%.6f')
        np.savetxt("./datatest/X_liedown_fix.txt", np.asarray(fixList), fmt='%.6f')

    def deepreturn(self):
        r1 = self.norm[-1]
        r2 = self.fix[-1]
        self.norm = []
        self.fix = []
        return r1, r2

    def specify(self, u, v, Z):
        '''
        :param u: pixel
        :param v: pixel
        :param Z: meter
        :return: XYZ
        '''
        u_undis, v_undis = self.distort(u, v)
        U = np.array([[u_undis], [v_undis], [1]])
        A = np.append(U, -self.K1, axis=1)
        A = np.append(A, -self.K2, axis=1)
        unkown = np.linalg.pinv(A).dot(self.K3) * Z
        return np.array([[unkown[1][0]], [unkown[2][0]], [Z]])


    def blockSize(self, Point1, Point2, Ztemp):
        '''
        leftuper\leftdow\rightup\leftdown four point to calculate
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

    def blockSize2(self, Point1, Point2, vid_cap, C, isside_temp):
        '''
        left\down\right\left four point to calculate, undistorted
        :param Point1: pixel, [x, y] likely
        :param Point2: pixel, [x, y] likely
        :param Ztemp: millimeter
        :param isside_temp: isside_temp == 1: stand; isside_temp == -1: lie down; else 0
        :return: meter
        '''
        assert isside_temp == -1 or isside_temp == 0 or isside_temp == 1

        cY, cX = C
        Ztemp = int(vid_cap[int(adapty(cY))][int(adaptx(cX))])
        P1 = np.reshape(copy.deepcopy(Point1), (2,1))
        P2 = np.reshape(copy.deepcopy(Point2), (2,1))
        Z = Ztemp * 1e-3 # meter
        PL = self.specify(P1[0, 0], (P1[1, 0] + P2[1,0])/2, Z)
        PD = self.specify((P1[0,0]+P2[0, 0])/2, P2[1, 0], Z)
        PU = self.specify((P1[0,0]+P2[0, 0])/2, P1[1, 0], Z)
        PR = self.specify(P2[0, 0], (P1[1,0]+P2[1, 0])/2, Z)
        Len = np.linalg.norm(PU - PD)
        Wid = np.linalg.norm(PL - PR)
        self.norm.append([Z, max(Len, Wid)])

        cY, cX = C
        Ztemp = int(vid_cap[int(adapty(cY))][int(adaptx(cX))])
        P1 = np.reshape(copy.deepcopy(Point1), (2,1))
        P2 = np.reshape(copy.deepcopy(Point2), (2,1))
        Z = Ztemp # millimeter

        i = 0
        while not isside_temp == 0 and abs(Z - Ztemp) < 50:
            Z = Ztemp
            i += 1
            Ztemp = int(vid_cap[int(adapty(cY - i))][int(adaptx(cX))]) if isside_temp == 1 else int(vid_cap[int(adapty(cY))][int(adaptx(cX + i))])
            if isside_temp == 1 and int(adapty(cY - i)) < P1[1, 0] or isside_temp == -1 and int(adaptx(cX + i)) > P2[0, 0]: break
            if int(adapty(cY - i)) == 479 or int(adapty(cY - i)) == 0 or int(adaptx(cX + i)) == 639 or int(adaptx(cX + i)) == 0: break
        Z = Z * 1e-3 # meter
        PL = self.specify(P1[0, 0], (P1[1, 0] + P2[1, 0]) / 2, Z)
        PD = self.specify((P1[0, 0] + P2[0, 0]) / 2, P2[1, 0], Z)
        PU = self.specify((P1[0, 0] + P2[0, 0]) / 2, P1[1, 0], Z)
        PR = self.specify(P2[0, 0], (P1[1, 0] + P2[1, 0]) / 2, Z)
        Len = np.linalg.norm(PU - PD)
        Wid = np.linalg.norm(PL - PR)
        self.fix.append([Z, max(Len, Wid)])

        return [Len, Wid]



    def blockSize3(self, Point1, Point2, Ztemp):
        '''
        left\down\right\left four point to calculate, and father's specify(distorted)
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
        PL = super().specify(P1[0, 0], (P1[1, 0] + P2[1,0])/2, Z)
        PD = super().specify((P1[0,0]+P2[0, 0])/2, P2[1, 0], Z)
        PU = super().specify((P1[0,0]+P2[0, 0])/2, P1[1, 0], Z)
        PR = super().specify(P2[0, 0], (P1[1,0]+P2[1, 0])/2, Z)
        Len = np.linalg.norm(PU - PD)
        Wid = np.linalg.norm(PL - PR)
        return max(Len, Wid)

    def Fix(self, Z):
        return -0.03168283 * Z + 0.00886965

    def tempShow(self, temp=''):
        numStamp = 10
        sizeStamp = 0.355
        if temp == '': temp = self.norm
        tempList = np.asarray(temp)
        numStamp, _ = np.shape(tempList)
        numStamp = min(numStamp, 100)
        try:
            plt.plot(tempList[-numStamp:-1, 0], np.ones((numStamp-1)) * sizeStamp)
            plt.plot(tempList[-numStamp:-1, 0], tempList[-numStamp:-1, 1])
            plt.ylabel('Size')
            plt.xlabel('Time stamp')
            plt.show()
        except:
            traceback.print_exc()
            print("Num < numStamp.")




if __name__ == "__main__":
    pc = rs.pointcloud()
    points = rs.points()

    pipeline = rs.pipeline()  # ??????????????????
    config = rs.config()  # Create a config??????????????????????????????????????????
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming ?????????
    pipe_profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)  # ???????????????????????????,????????????????????????????????????????????????
    cap = cv2.VideoCapture(0)

    obj0 = myCamera()
    obj = blockSize()
    try:
        while True:
            frames = pipeline.wait_for_frames()  # ??????????????????
            aligned_frames = align.process(frames)  # ??????????????????????????????
            depth_frame = aligned_frames.get_depth_frame()  # ?????????????????????????????????????(???)
            color_frame = aligned_frames.get_color_frame()  # ?????????????????????????????????????(???)
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            img_color = np.asanyarray(color_frame.get_data())  # ??????????????????????????????
            img_depth = np.asanyarray(depth_frame.get_data())  # ??????????????????????????????

            xx, yy = int(640 * rd.random()), int(480 * rd.random())

            dis, camera_coordinate = obj.specify_norm(depth_frame, xx, yy)

            my_coor = obj.specify(xx, yy, camera_coordinate[2])
            my_coor_dis = obj0.specify(xx, yy, camera_coordinate[2])
            # print(img_depth[yy][xx]) ## mm
            print(camera_coordinate, my_coor, my_coor_dis) ## m
            # cv2.imshow("temp", color_frame)
            cv2.imshow("temp2", img_color)
            cv2.waitKey(1)
    finally:
        pipeline.stop()


