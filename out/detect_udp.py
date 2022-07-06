#import kalf
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
import threading
import struct
import copy
import sys
import os
import socket

# global variable
Img_color = np.zeros((480, 640, 3), dtype=np.uint8)
Lock_color = threading.RLock()
lock_print = threading.RLock()
Flag_color = False

def udp_send_image(pack_size, socket, ip_port):
    global Img_color, Flag_color
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    Flag = True
    while Flag:
        # Flag = False
        t1 = time.time()

        if Flag_color:
            Lock_color.acquire()
            img = copy.deepcopy(Img_color)
            Flag_color = False
            Lock_color.release()
        _, img_encode = cv2.imencode('.jpg', img)

        data = img_encode.tobytes()
        # print(len(data))
        # 【定义文件头、数据】（打包名为l？不是，l表示长整型，占四个字节）
        fhead = struct.pack('id', len(data), time.time())
        # 【发送文件头、数据】
        socket.sendto(fhead, ip_port)
        # 每次发送x字节，计算所需发送次数
        send_times = len(data) // pack_size + 1
        # print(send_times)
        for count in range(send_times):
            time.sleep(0.005)
            if count < send_times - 1:
                socket.sendto(data[pack_size * count:pack_size * (count + 1)], ip_port)
            else:
                socket.sendto(data[pack_size * count:], ip_port)

        time.sleep(0.05)
        t2 = time.time()
        lock_print.acquire()
        print("thread 2",t2 - t1)
        sys.stdout.flush()
        lock_print.release()

def CalVariance(para, actual):
    x = para[0][0]
    y = para[0][1]
    a = para[1][0]
    b = para[1][1]
    theta = (para[2]/180)*math.pi

    if a< 15 or b < 15 :
        return float("inf")
    A = (a**2)*(math.sin(theta)**2)+(b**2)*(math.cos(theta)**2)
    B = 2*(a**2-b**2)*math.sin(theta)*math.cos(theta)
    C = (a**2)*(math.cos(theta)**2)+(b**2)*(math.sin(theta)**2)
    f = -a**2*b**2
    mVar = []
    for each in actual:
        ax = each[0][0]
        ay = each[0][1]
        Var = (A * (ax - x) ** 2 + B * (ax - x) * (ay - y) + C * (ay - y) ** 2 + f)/(a*b)**4
        mVar.append(Var)
    return abs(np.mean(mVar))

def specify(depth_frame, xx, yy):
    deep = depth_frame.get_distance(xx, yy)
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [xx, yy],
                                                        deep)
    camera_coordinate = np.array(camera_coordinate)
    dis = np.linalg.norm(camera_coordinate)
    #print(dis,camera_coordinate)
    #cv2.putTeext(color_image, "dis:{} ".format(dis), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255))
    #cv2.putText(color_image, "pixel:{} ".format(camera_coordinate), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
    #            (255, 0, 255))
    #print('0',type(camera_coordinate), camera_coordinate)
    return dis,camera_coordinate

def detect(save_img=False):
    global Img_color, Flag_color
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
        dataset = LoadStreams(source, img_size=imgsz)
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
    for path, img, im0s, vid_cap, imdal in dataset:
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
                        #print(xywh)
                        # print("-------------------------")
                        getpoint = torch.tensor(xyxy).view(1, 4)
                        getpoint = getpoint.numpy()
                        #depth_intrin = imdal[i].profile.as_video_stream_profile().intrinsics
                        # football_x = int((getpoint[0][0] + getpoint[0][2])/2)
                        # football_y = int((getpoint[0][1] + getpoint[0][3])/2)
                        # deep = imdal[i].get_distance(football_x, football_y)
                        # #ve = [(rs.rs2_deproject_pixel_to_point(depth_intrin, [football_x, football_y],
                        #  #                                                    dis)[i] - camera_coordinate[i])/0.022 for i in range(len(camera_coordinate)) ]
                        # camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [football_x, football_y],
                        #                                                      deep)
                        # if(ve[0]!=0):
                        #     ex.filter(camera_coordinate,ve)
                        # camera_coordinate = np.array(camera_coordinate)
                        # dis = np.linalg.norm(camera_coordinate)
                        # print(dis, camera_coordinate)
                        # cv2.putText(im0, "dis:{} ".format(dis), (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255))
                        cv2.putText(im0, "pixel:{} ".format(time.time()), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255))
                        xy1 =[int(getpoint[0][0]),int(getpoint[0][1])]
                        xy2 = [int(getpoint[0][2]), int(getpoint[0][3])]
                        Para, flag,mx,my = myHoughEllipse.HoughEllipse(im0,xy1,xy2)
                        # Para= [Para[0][0],Para[0][1]]
                        # cv2.circle(im0, (int(Para[0]),int(Para[1])), 5,(0,0,255),5)
                        """
                        计算角度
                        """
                        if flag == 0:
                            continue
                        horres = 0
                        velres = 0
                        l0, pix0 = specify(imdal[i],mx, my)
                        theta = []
                        for delta in range(20):
                            if mx - delta > 0 and mx + delta < 640:
                                l1, pix1 = specify(imdal[i],mx - delta, my)
                                l2, pix2 = specify(imdal[i],mx + delta, my)
                                x1 = np.linalg.norm(pix0 - pix1)
                                x2 = np.linalg.norm(pix0 - pix2)
                                if (x1 ** 2 + l0 ** 2 - l1 ** 2) / (2 * x1 * l0) > 1 or (
                                        x1 ** 2 + l0 ** 2 - l1 ** 2) / (
                                        2 * x1 * l0) < -1 or (x2 ** 2 + l0 ** 2 - l2 ** 2) / (2 * x2 * l0) > 1 or (
                                        x2 ** 2 + l0 ** 2 - l2 ** 2) / (2 * x2 * l0) < -1:
                                    continue
                                a = (math.acos((x1 ** 2 + l0 ** 2 - l1 ** 2) / (2 * x1 * l0)) / 3.1415926) * 180
                                b = (math.acos((x2 ** 2 + l0 ** 2 - l2 ** 2) / (2 * x2 * l0)) / 3.1415926) * 180
                                theta.append((b - a) / 2)

                        theta = np.array(theta[1:])

                        if horres == 0 or horres:
                            horres = np.mean(theta)
                        else:
                            horres = 0.5 * np.mean(theta) + 0.5 * horres

                        theta = []
                        for delta in range(20):
                            if my - delta > 0 and my + delta < 480:
                                l1, pix1 = specify(imdal[i],mx, my - delta)
                                l2, pix2 = specify(imdal[i],mx, my + delta)
                                x1 = np.linalg.norm(pix0 - pix1)
                                x2 = np.linalg.norm(pix0 - pix2)
                                if (x1 ** 2 + l0 ** 2 - l1 ** 2) / (2 * x1 * l0) > 1 or (
                                        x1 ** 2 + l0 ** 2 - l1 ** 2) / (
                                        2 * x1 * l0) < -1 or (x2 ** 2 + l0 ** 2 - l2 ** 2) / (2 * x2 * l0) > 1 or (
                                        x2 ** 2 + l0 ** 2 - l2 ** 2) / (2 * x2 * l0) < -1:
                                    continue
                                a = (math.acos((x1 ** 2 + l0 ** 2 - l1 ** 2) / (2 * x1 * l0)) / 3.1415926) * 180
                                b = (math.acos((x2 ** 2 + l0 ** 2 - l2 ** 2) / (2 * x2 * l0)) / 3.1415926) * 180
                                theta.append((b - a) / 2)

                        theta = np.array(theta[1:])
                        if velres == 0 or velres:
                            velres = np.mean(theta)
                        else:
                            velres = 0.5 * np.mean(theta) + 0.5 * velres

                        stand_str = "位姿为躺"
                        direction = "None"
                        if velres > -40:
                            stand_str = "位姿为立"
                            if horres > 0:
                                direction = "right"
                            else:
                                direction = "left"

                        print("距中心的距离为", l0)
                        print("以相机为基准的方位坐标为", pix0)
                        print("水平偏角为:", horres, direction )
                        print("俯仰偏角为:", velres, stand_str)
                        poststr =str(l0)+"###"+str(horres)
                        sent = ser.write(poststr.encode())
                        print("发送成功")

                        # depth_image = vid_cap[i].astype(np.float64)
                        # camera_coordinate = rs.rs2_deproject_pixel_to_point(intrin=imgdpin[i], pixel=[football_x, football_y], depth=depth_image)
                        # print(camera_coordinate)
                        # print(football_x, football_y)
                        # print("-------------------------")
                        # print(xywh)
                        #print("-------------------------")

                        # print(dis)
                        # print("-------------------------")

                        # cv2.imshow('depth', vid_cap[i])
                        # k = cv2.waitKey(1) & 0xFF
                        # if k== 27:
                        #     cv2.destroyWindow('depth')

            # Print time (inference + NMS)
            # t2 = time_synchronized()
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                Lock_color.acquire()
                Img_color = copy.deepcopy(im0)
                Flag_color = True
                Lock_color.release()
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer
            #             if vid_cap:  # video
            #                 # fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                 save_path += '.mp4'
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':

    '''send message'''
    # os.system("nmcli device wifi connect yanzhixue password 12344321 ifname wlan0")
    lock_print.acquire()
    print("Wifi Connected!")
    sys.stdout.flush()
    lock_print.release()

    ip_port = ("192.168.43.25", 8000)  # 目标ip和端口
    # local_port = ("192.168.10.233",9000) #发送端ip和端口
    socket_conf = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # udp_send_image(Img_color, 65507, socket_conf, ip_port)
    t = threading.Thread(target=udp_send_image, args=(65507, socket_conf, ip_port)).start()

    #chuan kou mo kuai
    port_list = list(serial.tools.list_ports.comports())
    print(port_list)
    if len(port_list) == 0:
        print("无可用串口")
    else:
        for i in range(len(port_list)):
            print(port_list)
    # portx = "COM7"
    portx = "COM7"

    bps = 115200
    timex = None
    ser = serial.Serial(portx, bps, timeout=timex)
    print("串口参数：", ser)
    if (False == ser.is_open):
        ser = -1
    # cnt = 0


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
