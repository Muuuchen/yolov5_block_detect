import numpy as np
import pyrealsense2 as rs

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

#返回一个识别框的平均深度
def Closest_Block(imdep,getpoint):
    x1 = int(getpoint[0][0])
    y1 = int(getpoint[0][1])
    x2 = int(getpoint[0][2])
    y2 = int(getpoint[0][3])
    if x1<0 or x1>=640 or x2<0 or x2>=640 or x1 > x2:
        x1 = adaptx(x1)
        x2 = adaptx(x2)
    if y1 < 0 or y1 >= 480 or y2 < 0 or y2 >= 480 or y1 > y2:
        y1 = adapty(y1)
        y2 = adapty(y2)
    mx = int((x1+x2)/2)
    my = int((y1+y2)/2)
    mk,c = specify(imdep, mx, my)
    sum = mk
    sump = 1
    for i in range(1,min(min(x2-mx,mx-x1),20)):
        dis, c =specify(imdep,mx-i,my)
        if abs(dis-mk) <= 0.5:
            sum=sum+dis
            sump=sump+1
        dis, c =specify(imdep,mx+i,my)
        if abs(dis-mk) <= 0.5:
            sum=sum+dis
            sump=sump+1
    for i in range(1,min(min(y2-my,my-y1),20)):
        dis, c =specify(imdep,mx,my-i)
        if abs(dis-mk) <= 0.5:
            sum=sum+dis
            sump=sump+1
        dis, c =specify(imdep,mx,my+i)
        if abs(dis-mk) <= 0.5:
            sum=sum+dis
            sump=sump+1

    return sum/sump