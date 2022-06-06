import argparse
import time
from pathlib import Path
import pyrealsense2 as rs
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import serial
import serial.tools.list_ports
import time
import math
import cv2
import numpy as np
import matplotlib.pyplot
import random
import myHoughEllipse
from AngleCalculate import AngleCal
from ClosestBlock1 import Closest_Block
import threading
from socket import *
import sys
import traceback
'''
此代码中加入了client模块，用于向上位机传输实时检测结果的图像；接收端server.py的代码也在文件夹中
client模块的具体格式：
#client模块
code
#client模块
'''

def specify(depth_frame, xx, yy):
    deep = depth_frame.get_distance(xx, yy)
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [xx, yy],
                                                        deep)
    camera_coordinate = np.array(camera_coordinate)
    dis = np.linalg.norm(camera_coordinate)
    return dis,camera_coordinate

#client模块
def send_img(): #向主机发送数据
    sss.sendto(send_data, addr)
    # print(f'已发送{len(send_data)}Bytes的数据')
    # sss.close()
'''
def fabufa0():
    global faLock, faFlag, ser
    while True:
        print("wait")
        poststr = str(0) + "#" + str(0) + "#" + str(0)
        print("Send successfully!")
        _ = ser.write(poststr.encode())
        data = ser.read(1)
        print(data.decode())

        # sys.stdout.flush()

        faLock.acquire()
        faFlag = True
        faLock.release()
        time.sleep(0.1)
'''

def fabufa():
    # ser.settimeout(0)
    # data = ser.read(1024)
    data = ser.read(1024)
    print(data.decode())
    if len(data.decode()) == 0: return False
    return True

def detect(save_img=False):
    global addr, faFlag, send_data
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        Flagtry = True
        while (Flagtry):
            try:
                dataset = LoadStreams(source, img_size=imgsz)
                Flagtry = False
            except:
                traceback.print_exc()
    else:
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    camera_coordinate = [0, 0, 0]
    #ex = kalf.kalman()

    #client模块
    try:
        addr = (str(sys.argv[1]), 8080)
    except:
        addr = ('192.168.10.213', 8080)          # 127.0.0.1表示本机的IP，相当于我和“自己”的关系
    #client模块

    for path, img, im0s, vid_cap, imdal in dataset:
        # input img from datasets.py
        #imdal[i] is a matrix containing the depth data of ith frame
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)


        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        faFlag2 = False
        if len(port_list) != 0 and False != ser.is_open:
            faFlag = fabufa()
            faFlag2 = faFlag

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                depsum = float("inf")
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        t0 = time.time()

                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        getpoint = torch.tensor(xyxy).view(1, 4)
                        getpoint = getpoint.numpy()
                        #修正选择一个平均距离最近框
                        sum1 = Closest_Block(imdal[i],getpoint)
                        if sum1 < depsum:
                            depsum = sum1
                            xy1 = [int(getpoint[0][0]), int(getpoint[0][1])]
                            xy2 = [int(getpoint[0][2]), int(getpoint[0][3])]
                #----------Modification Start---------
                # if xy2 != []: # 如果有识别到至少一个框
                Para, flag, mx, my = myHoughEllipse.HoughEllipse(im0, xy1, xy2) #识别白色的椭圆区域
                if flag == 0: continue
                horres, velres, l0, pix0 = AngleCal(imdal[i], mx, my) #计算角度
                l1 = np.sqrt(l0**2 - pix0[1]**2)

                zitai = 1 #1为躺 0为立
                stand_str = "位姿为躺"
                direction = "None"
                if velres > -40:
                    zitai =0
                    stand_str = "位姿为立"
                    if horres > 0: direction = "right"
                    else: direction = "left"

                print("距中心的距离为", l0)
                print("距中心的水平距离为", l1)
                print("以相机为基准的方位坐标为", pix0)
                print("水平偏角为:", horres, direction)
                print("俯仰偏角为:", velres, stand_str)
                sys.stdout.flush()

                cv2.circle(im0, (int(pix0[0]), int(pix0[1])), 3, (0,101,255), -1)
                if not math.isnan(horres): txt = int(horres)
                else: txt = horres
                cv2.putText(im0,
                            '{}[H:{}Degree]'.format(['stand', 'lie down'][zitai], txt),
                            (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (252, 218, 252), 2)
                # Serial Block
                if len(port_list) != 0 and False != ser.is_open and faFlag \
                        and not math.isnan(l1) and not math.isnan(horres):
                        # and l0 < 3 and not math.isnan(l1) and not math.isnan(horres):
                    # unit is millimeter
                    poststr = str(l1 * 1000) + "#" + str(horres * 1000) + "#" + str(zitai)
                    for i in range(len(pix0)): poststr += "#" + str(pix0[i] * 1000)
                    faFlag2 = False
                    print("发送成功")
                    _ = ser.write(poststr.encode())
                    # faLock.acquire()
                    # faFlag = False
                    # faLock.release()
        if faFlag2:
            poststr = str("N")
            print("发送N")
            _ = ser.write(poststr.encode())


        if view_img:
            # cv2.imshow(str(p), im0)

            # client模块
            img = im0
            th = threading.Thread(target=send_img)
            th.setDaemon(True)
            _, send_data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
            # print(send_data.size)
            th.start()
            # cv2.putText(img, "client", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # cv2.imshow('client_frame', img)
            # cv2.waitKey(1)  # 1 millisecond
            # client模块


    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    '''send message'''
    # os.system("nmcli device wifi connect yanzhixue password 12344321 ifname wlan0")
    # print("Wifi Connected!")
    # to PC
    sss = socket(AF_INET, SOCK_DGRAM)

    # to STM32
    port_list = list(serial.tools.list_ports.comports())
    if len(port_list) == 0:
        print("无可用串口!!!")
    else:
        for i in range(len(port_list)): print(port_list)
        # ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0) # None
        ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=0)
        if (False == ser.is_open):
            ser = -1
            print("ttyUSB open failed.")
        print("串口参数：", ser)
        # faFlag = False
        # faLock = threading.RLock()
        # t = threading.Thread(target=fabufa, args=()).start()

    cv2.VideoCapture(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/best.pt', help='model.pt path(s)') #更改预设以改变网络类型
    parser.add_argument('--source', type=str, default= '0', help='source')  # file/folder, 0 for webcam 输入路径
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.55, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.55, help='IOU threshold for NMS') #非极大值抑制 iou
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
    # ser.close()
