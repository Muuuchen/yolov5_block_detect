import numpy as np
import pyrealsense2 as rs
import cv2

def specify(depth_frame, xx, yy):
    deep = depth_frame.get_distance(xx, yy)
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [xx, yy],
                                                        deep)
    camera_coordinate = np.array(camera_coordinate)
    dis = np.linalg.norm(camera_coordinate)
    return dis,camera_coordinate

def adaptx(x):
    return max(min(x,639),0)
def adapty(x):
    return max(min(x,479),0)

#判断白色
def inwhite(a):
    low_hsv1 = np.array([0, 0, 134], dtype=np.uint8)
    upper_hsv1 = np.array([180, 30, 255], dtype=np.uint8)
    pd = 1
    for i in range(0,3):
        if a[i]<low_hsv1[i] or a[i]>upper_hsv1[i]:
            pd = 0

    low_hsv2 = np.array([21, 43, 46], dtype=np.uint8)
    upper_hsv2 = np.array([38, 255, 255], dtype=np.uint8)
    pd1 = 1
    for i in range(0,3):
        if a[i]<low_hsv2[i] or a[i]>upper_hsv2[i]:
            pd1 = 0
    return pd or pd1

#判断蓝色
def inblue(a):
    low_hsv = np.array([100, 43, 46], dtype=np.uint8)
    upper_hsv = np.array([124, 255, 255], dtype=np.uint8)
    pd = 1
    for i in range(0,3):
        if a[i]<low_hsv[i] or a[i]>upper_hsv[i]:
            pd = 0
    return pd

def Front_and_Back_Cal(imcol,imdep,mx,my,getpoint):
    x1 = int(getpoint[0][0])
    y1 = int(getpoint[0][1])
    x2 = int(getpoint[0][2])
    y2 = int(getpoint[0][3])
    if mx < 0 or mx >= 640 or my < 0 or my >= 480:
        return -1
    if x1<0 or x1>=640 or x2<0 or x2>=640 or x1 > x2:
        x1 = adaptx(x1)
        x2 = adaptx(x2)
    if y1 < 0 or y1 >= 480 or y2 < 0 or y2 >= 480 or y1 > y2:
        y1 = adapty(y1)
        y2 = adapty(y2)
    Gaus = cv2.GaussianBlur(imcol, (7,7), 0)
    imhsv = cv2.cvtColor(Gaus, cv2.COLOR_BGR2HSV)

    mk,c = specify(imdep, mx, my)
    nowdepth = mk

    sumw = 0
    sumb = 0
    for i in range(0,my-y1):
        dis, c =specify(imdep,mx,my-i)
        if abs(nowdepth-dis)>0.25:
            break
        sumw = sumw + inwhite( imhsv[my-i][mx] )
        sumb = sumb + inblue( imhsv[my-i][mx] )

    if sumb < 5 and sumw < 5:
        return -1

    #此处考虑了最顶上的块的情况，即蓝色量远大于白色量
    if sumb!=0 and sumw/sumb<30:
        return 1
    else:
        return 0
    #1为正，0为负,-1表示不能判断
