import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from classes.client import client
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import serial.tools.list_ports
import time, math, random, cv2, traceback, os, sys
import numpy as np
from AngleCalculate import AngleCal, calculate_normal_vector
from ClosestBlock1 import Closest_Block
from F_B_cal import Front_and_Back_Cal
from classes import Camera
from init import imshowFlag, R2
from blockManage import sizeNearest
from Camera2 import Getupstate
from ExportState import exports
from myQueue import Queue
resq = Queue(5)

client_obj = client()
blockSize_obj = Camera.blockSize()
sendStr = 'No Serial'
cameras = ['0','2'] if R2 else ['0']


with open('./utils/stream.txt', 'w') as f:
    print("###")
    for camera in cameras:
        f.write(camera)
        if not cameras.index(camera) + 1 == len(cameras): f.write('\n')


class StateParameters():
    def __init__(self):
        self.isside = 0
        self.depsum = float("inf")
        self.mx = 0
        self.my = 0
        self.maxSize = []
        self.flag = 0
        self.normElement = 0
        self.fixElement = 0
        self.horres = 0
        self.velres = 0
        self.F_B_pos = 0
        self.up_isside = 0 # 1/8
        self.up_isfront = 0 # 1 zhengmin 2
        self.up_isback = 0 # <0.05
        self.l0 = 0
        self.l1 = 0
        self.pix0 = []
        self.isfivestand = 0
        self.isonestand= 0
        self.dflag1 = 0
        self.dflag2 = 0
        self.colorflag = 0
        self.blocksize = 0
        self.up_xyxy = []
        self.sp_upsquare = 0
        self.bias = 30
def fabufa():
    # ser.settimeout(0)
    # data = ser.read(1024)
    data = ser.read(1024)
    print(data.decode())
    try:
        if len(data.decode()) == 0: return False
        return True
    except:
        traceback.print_exc()
        return False


def detect(save_img=False):
    global faFlag, sendStr
    time_stamp = time.time()
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

    for path, img, im0s, vid_cap, imdal in dataset:
        # input img from datasets.py
        # imdal[i] is a matrix containing the depth data of ith frame
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

        faNFlag = False
        if len(port_list) != 0 and False != ser.is_open: faFlag = True

        # summary parameters
        sp = StateParameters()
        # Process detections
        for i, det in enumerate(pred):  # detections per imag
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            im1 = np.copy(im0)
            # im0不画框，im1画框和最后imshow
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
                if i == 0:
                    # Write results
                    sp.dflag1 = 1
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or view_img:  # Add bbox to image
                            t0 = time.time()
                            label = f'{names[int(cls)]} {conf:.2f}'
                            colorflag = int(cls)

                            plot_one_box(xyxy, im1, label=label, color=colors[int(cls)], line_thickness=3)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            getpoint = torch.tensor(xyxy).view(1, 4).numpy()

                            sum1, flag_temp, cX, cY, isside_temp, mask = Closest_Block(vid_cap[i], im0, getpoint,
                                                                                       colorflag)
                            #zhe li isside xu yao yong colormask de zhanbi qu panduan

                            diameter = blockSize_obj.blockSize2(getpoint[0, 0:2], getpoint[0, 2:4], vid_cap[i],
                                                                np.array([cY, cX]),
                                                                isside_temp) if True else blockSize_obj.blockSize(
                                getpoint[0, 0:2], getpoint[0, 2:4],
                                vid_cap[i][int((getpoint[0, 1] + getpoint[0, 3]) / 2)][
                                    int((getpoint[0, 0] + getpoint[0, 2]) / 2)], isside_temp)

                            if sum1 < sp.depsum:  # if maxSize <= diameter:
                                sp.normElement, sp.fixElement = blockSize_obj.deepreturn()
                                sp.depsum = sum1
                                sp.mx = int(cX)
                                sp.my = int(cY)
                                sp.maxSize = diameter
                                sp.flag = flag_temp
                                sp.colorflag_dest = colorflag
                                sp.isside = isside_temp
                                finalMask = mask

                    # ----------Modification Start---------
                    print("------")
                    if sp.depsum == float("inf") or sp.flag == 0: continue
                    if float(sp.normElement[1]) > 1e-4 and float(sp.fixElement[1]) > 1e-4 and False: blockSize_obj.save(
                        sp.normElement, sp.fixElement)

                    cv2.circle(im1, (int(sp.mx), int(sp.my)), 8, (0, 101, 255), -1)
                    sp.horres, sp.velres, sp.l0, sp.pix0 = calculate_normal_vector(vid_cap[i], sp.mx,
                                                                                   sp.my) if False else AngleCal(
                        imdal[i], sp.mx, sp.my)
                    sp.l1 = np.sqrt(sp.l0 ** 2 - sp.pix0[1] ** 2)
                    sp.F_B_pos = Front_and_Back_Cal(im0, imdal[i], sp.mx, sp.my, getpoint, sp.colorflag_dest)
                    sp.blocksize = sizeNearest(sp.maxSize, sp.isside, sp.velres)
                    print(sp.blocksize, sp.maxSize)
                if i == 1:
                    sp.dflag2 = 1
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or view_img:  # Add bbox to image
                            t0 = time.time()
                            label = f'{names[int(cls)]} {conf:.2f}'
                            sp.colorflag = int(cls)
                            sp.up_xyxy = xyxy
                            plot_one_box(xyxy, im1, label=label, color=colors[int(cls)], line_thickness=3)
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                            getpoint = torch.tensor(xyxy).view(1, 4).numpy()
                            sp.up_isside, sp.up_isfront, sp.up_isback,sp.up_square,sp.isfivestand,sp.isonestand= Getupstate(im1, getpoint, sp.colorflag)

                cv2.putText(im1, sendStr, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [(252, 218, 252), (255, 0, 0)][1],
                            2)
                if imshowFlag: cv2.imshow(str(p), im1)
                client_obj.sendImg(im1)
        if sp.dflag1 * sp.dflag2 == 1:
            fb, sl, size=exports(sp)
            restr = "#" + str(fb) + str(sl) + str(size)
            if time.time() - time_stamp > 2:
                resq.clear()
            time_stamp = time.time()
            resq.printQ()
            if not resq.full():
                resq.push(restr)
            else:
                # Serial Block
                resq.pop()
                resq.push(restr)
                print("去噪结果：", resq.getmax())
                if len(port_list) != 0 and False != ser.is_open and faFlag and not math.isnan(sp.l1) and not math.isnan(
                        sp.horres) and not math.isnan(sp.velres):  # and l0 < 3
                    poststr = resq.getmax() # unit is millimeter
                    print("发送成功：", poststr)
                    _ = ser.write(poststr.encode())




        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")



if __name__ == '__main__':
    print("Wifi connecting...")
    # os.system("nmcli device wifi connect yanzhixue2 password 12344321 ifname wlan0")

    # to STM32
    port_list = list(serial.tools.list_ports.comports())
    if len(port_list) == 0:
        print("无可用串口!!!")
    else:
        for i in range(len(port_list)): print(port_list)
        ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0)
        if (False == ser.is_open):
            ser = -1
            print("ttyUSB open failed.")
        print("串口参数：", ser)

    cv2.VideoCapture(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/oldbest.pt',
                        help='model.pt path(s)')  # 更改预设以改变网络类型
    parser.add_argument('--source', type=str, default='./utils/stream.txt',
                        help='source')  # file/folder, 0 for webcam 输入路径
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.55, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.55, help='IOU threshold for NMS')  # 非极大值抑制 iou
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
    check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
    # ser.close()
