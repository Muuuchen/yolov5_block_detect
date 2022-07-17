import pyrealsense2 as rs
import numpy as np
import cv2
import time

def specify(football_x, football_y):
    deep = depth_frame.get_distance(football_x, football_y)
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [football_x, football_y],
                                                        deep)
    camera_coordinate = np.array(camera_coordinate)
    dis = np.linalg.norm(camera_coordinate)
    print(dis, camera_coordinate)
    cv2.putText(color_image, "dis:{} ".format(dis), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255))
    cv2.putText(color_image, "pixel:{} ".format(camera_coordinate), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 0, 255))
    cv2.circle(color_image, (football_x, football_y), 5, (0, 0, 255))

if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # Start streaming
    pipeline.start(config)
    start = time.time()
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Convert images to numpy arrays

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    # Stack both images horizontally
    #
    #
    #
    #
    tyimg = color_image
    gaus = cv2.GaussianBlur(tyimg, (7, 7), 1.5)
    imghsv = cv2.cvtColor(gaus, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([32, 43, 46], dtype=np.uint8)
    upper_hsv = np.array([47, 225, 225], dtype=np.uint8)
    mask = cv2.inRange(imghsv, low_hsv, upper_hsv)

    dilation = cv2.dilate(mask, (9, 9), iterations=1)
    erosion = cv2.erode(dilation, (9, 9), iterations=1)
    """先膨胀后腐蚀"""
    ret, thresh = cv2.threshold(erosion, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    maxcon = contours[0]
    for eachcon in contours:
        if len(eachcon)>len(maxcon):
            maxcon = eachcon
    print(maxcon)
    # hmin = -999, hmax = 999
    # for each in maxcon:
    #     each[0][0][0]
    cv2.drawContours(tyimg,maxcon,-1,(0,0,255),3)
    # cv2.imshow("img", tyimg)
    #
    #
    #

    specify(480, 320)

    images = np.hstack((color_image, depth_colormap))
    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    # cv2.imshow('RealSense', images)
    end = time.time()
    print("cost:", end - start)
    key = cv2.waitKey(0)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
    pipeline.stop()
