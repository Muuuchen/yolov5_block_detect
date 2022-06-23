import argparse
import time
from pathlib import Path
import pyrealsense2 as rs
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt',
                        help='model.pt path(s)')  # 更改预设以改变网络类型
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')  # 非极大值抑制 iou
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    return parser.parse_args()

if __name__ == "__main__":
    # Configure depth and color streams
    opt = parser()
    set_logging()
    device = select_device(opt.device)
    half = device.type!='cpu'
    model = attempt_load(opt.weights)



    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)
    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            t0 = time.time()

            img = torch.from_numpy(color_image).to(device)
            img = img.half() if half else img.float()
            img /= 255.0


            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('RealSense', images)
            # key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            # if key & 0xFF == ord('q') or key == 27:
            #     cv2.destroyAllWindows()
            #     break
    finally:
        # Stop streaming
        pipeline.stop()