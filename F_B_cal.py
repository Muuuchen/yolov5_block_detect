import numpy as np
import pyrealsense2 as rs
import cv2
from classes.hsvMan import hsvRange


R1 = hsvRange([[0, 68, 59], [20, 255, 255]])
R2 = hsvRange([[150, 68, 59], [179, 255, 255]])

B  = hsvRange([[48, 160, 0], [144, 255, 255]])


def incolor(a, colorflag):
    '''
    judge if a in blue block or red block
    :param a: hsv instance of a pixel
    :param colorflag: Red 0 Blue 1
    :param RB: none
    :return:
    '''
    return 1 if not colorflag and (R1.inRange(a) or R2.inRange(a)) or colorflag and B.inRange(a) else 0



def specify(depth_frame, xx, yy):
    '''

    :param depth_frame:
    :param xx:
    :param yy:
    :return:
    '''
    deep = depth_frame.get_distance(xx, yy)
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [xx, yy], deep)
    camera_coordinate = np.array(camera_coordinate)
    dis = np.linalg.norm(camera_coordinate)
    return dis,camera_coordinate

def adaptx(x): return max(min(x,639),0)

def adapty(y): return max(min(y,479),0)

def Front_and_Back_Cal(imcol,imdep,mx,my,getpoint,colorflag):
    '''

    :param imcol:
    :param imdep:
    :param mx:
    :param my:
    :param getpoint:
    :param colorflag:
    :return: 1为正面立, 0为反面立, -1表示不能判断
    '''
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

    nowdepth,c = specify(imdep, mx, my)

    lsum = 1e-10
    lsumb = 0
    for i in range(0,mx-x1):
        dis, c =specify(imdep,mx-i,my)
        if abs(nowdepth - dis) > 0.50: break
        lsum = lsum + 1
        lsumb = lsumb + incolor(imhsv[my][mx-i], colorflag)

    rsum = 1e-10
    rsumb = 0
    for i in range(0,x2-mx):
        dis, c =specify(imdep,mx+i,my)
        if abs(nowdepth - dis) > 0.50: break
        rsum = rsum + 1
        rsumb = rsumb + incolor(imhsv[my][mx-i], colorflag)

    return 1 if ((lsumb / lsum > 0.05)or (rsumb / rsum > 0.05)) else 0

'''
    sum = 0
    sumb = 0
    for i in range(0,y2-my):
        dis, c =specify(imdep,mx,my+i)
        if abs(nowdepth - dis) > 0.50: break
        sum = sum + 1
        sumb = sumb + incolor(imhsv[my-i][mx], colorflag)

    # 此处考虑了最顶上的块的情况，即蓝色量远大于白色量
    return 1 if sumb / sum > 0.02 else 0
    '''