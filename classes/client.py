import sys, os
import re, cv2
from socket import *
import threading

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from init import ipDefault, portDefault

inRange = 1

class client():
    '''
    to PC
    '''
    def __init__(self):
        self.sss = socket(AF_INET, SOCK_DGRAM)
        self.setAddr()

    def __del__(self):
        self.sss.close()

    def send_func(self):
        self.sss.sendto(self.send_data, self.addr)
        # print(f'已发送{len(send_data)}Bytes的数据')

    def sendImg(self, img):
        _, self.send_data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        th = threading.Thread(target=self.send_func)
        th.setDaemon(True)
        th.start()

    def setAddr(self, ip='0.0.0.0'):
        '''
        Addr = (ip, port)
        :param ip:
        :return:
        '''
        if ip == '0.0.0.0':
            try:
                self.addr = (sys.argv[inRange], portDefault)
                print("Input ip {} suceess.".format(self.addr[0]))
            except:
                self.addr = (ipDefault, portDefault)
                print("ipDefault {}.".format(self.addr[0]))
        else:
            self.addr = (ip, portDefault)
            print("Set ip {} suceess.".format(self.addr[0]))
        assert bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", self.addr[0]))

    def validate_ip_address(ip):
        parts = str(ip).split(".")

        if len(parts) != 4:
            print("IP address {} is not valid".format(ip))
            return False

        for part in parts:
            if not isinstance(int(part), int):
                print("IP address {} is not valid".format(ip))
                return False

            if int(part) < 0 or int(part) > 255:
                print("IP address {} is not valid".format(ip))
                return False

        print("IP address {} is valid".format(ip))
        return True


if __name__ == "__main__":
    obj = client()
    print(obj.addr)
