import traceback

import numpy as np
import cv2
from socket import *
import re
import argparse

ipDefault = '192.168.10.211'
portDefault = 8080

class server():
    def __init__(self):
        self.sss = socket(AF_INET, SOCK_DGRAM)
        self.setAddr(opt.ip, opt.port)
        self.sss.bind(self.addr)
        print("---------------------if close needed, Ctrl+C please---------------------")

    def __del__(self):
        '''
        !-- only work at the period of close normally or Ctrl+C
        and then port would be released
        :return:
        '''
        self.sss.close()

    def setAddr(self, ipLocal = None, portLocal = None):
        ip = ipLocal if not ipLocal == None else ipDefault
        port = int(portLocal) if not portLocal == None and not portLocal == 'None' else portDefault
        try:
            addr = (ip, port)
            assert bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", addr[0]))
            print("ip {} port {} set suceess.".format(addr[0], addr[1]))
        except:
            traceback.print_exc()
            addr = (ipDefault, portDefault)
            print("ip {} port {} set suceess.".format(addr[0], addr[1]))
        assert bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", addr[0]))
        self.addr = addr

    def Recv(self):
        while True:
            data, _ = self.sss.recvfrom(921600)
            receive_data = np.frombuffer(data, dtype='uint8')
            r_img = cv2.imdecode(receive_data, 1)
            r_img = r_img.reshape(480, 640, 3)

            # cv2.putText(r_img, "server", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('server_frame', r_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, help="server ip", required=False)
    parser.add_argument("--port", type=str, help="server ip", required=False)
    opt = parser.parse_args()
    print("--ip = ", opt.ip)
    print("--port = ", opt.port)

    obj = server()
    obj.Recv()

