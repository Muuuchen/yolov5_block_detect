import math
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

def AngleCal(depth_frame,mx,my):
    horres = 0
    velres = 0
    l0, pix0 = specify(depth_frame, mx, my)
    # specify(depth_frame,x,y) returns the depth of coodinate (x,y) in depth_frame
    theta = []
    for delta in range(20):
        if mx - delta > 0 and mx + delta < 640:
            l1, pix1 = specify(depth_frame, mx - delta, my)
            l2, pix2 = specify(depth_frame, mx + delta, my)
            if(l1 == 0 or l2 == 0):
                break
            x1 = np.linalg.norm(pix0 - pix1)
            x2 = np.linalg.norm(pix0 - pix2)
            # get 3D distance

            if 2 * x1 * l0==0 or (x1 ** 2 + l0 ** 2 - l1 ** 2) / (2 * x1 * l0) > 1 or (
                    x1 ** 2 + l0 ** 2 - l1 ** 2) / (
                    2 * x1 * l0) < -1 or (x2 ** 2 + l0 ** 2 - l2 ** 2) / (2 * x2 * l0) > 1 or (
                    x2 ** 2 + l0 ** 2 - l2 ** 2) / (2 * x2 * l0) < -1:
                continue
            a = (math.acos((x1 ** 2 + l0 ** 2 - l1 ** 2) / (2 * x1 * l0)) / 3.1415926) * 180
            #b = (math.acos((x2 ** 2 + l0 ** 2 - l2 ** 2) / (2 * x2 * l0)) / 3.1415926) * 180
            beta = (math.atan(pix0[2]/pix0[0])/3.1415926)*180
            theta.append(abs(beta) - abs(a))

    theta = np.array(theta[1:])
    mean = np.mean(theta, axis=0)
    std = np.std(theta, axis=0)
    theta_res = [x for x in theta if(x >mean-3*std and x < mean+3*std)]
    if horres == 0 or horres:
        horres = np.mean(theta_res)
    else:
        horres = 0.5 * np.mean(theta_res) + 0.5 * horres

    theta = []
    for delta in range(20):
        if my - delta > 0 and my + delta < 480:
            l1, pix1 = specify(depth_frame, mx, my - delta)
            l2, pix2 = specify(depth_frame, mx, my + delta)
            if(l1 == 0 or l2 == 0):
                break
            x1 = np.linalg.norm(pix0 - pix1)
            x2 = np.linalg.norm(pix0 - pix2)
            if 2 * x1 * l0==0 or (x1 ** 2 + l0 ** 2 - l1 ** 2) / (2 * x1 * l0) > 1 or (
                    x1 ** 2 + l0 ** 2 - l1 ** 2) / (
                    2 * x1 * l0) < -1 or (x2 ** 2 + l0 ** 2 - l2 ** 2) / (2 * x2 * l0) > 1 or (
                    x2 ** 2 + l0 ** 2 - l2 ** 2) / (2 * x2 * l0) < -1:
                continue
            a = (math.acos((x1 ** 2 + l0 ** 2 - l1 ** 2) / (2 * x1 * l0)) / 3.1415926) * 180
            #b = (math.acos((x2 ** 2 + l0 ** 2 - l2 ** 2) / (2 * x2 * l0)) / 3.1415926) * 180
            beta = (math.atan(pix0[2]/pix0[0])/3.1415926)*180
            theta.append(abs(beta) - abs(a))
    theta = np.array(theta[1:])
    mean = np.mean(theta, axis=0)
    std = np.std(theta, axis=0)
    theta_res = [x for x in theta if(x >mean-3*std and x < mean+3*std)]
    if velres == 0 or velres:
         velres = np.mean(theta_res)
    else:
        velres = 0.5 * np.mean(theta_res) + 0.5 * velres
    return horres,velres,l0,pix0