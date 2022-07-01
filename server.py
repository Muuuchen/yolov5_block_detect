import numpy as np
import cv2
from socket import *
import sys
import re

ipDefault = '192.168.43.25'
port = 8080

class server():
    def __init__(self):
        self.sss = socket(AF_INET, SOCK_DGRAM)
        self.setAddr()
        self.sss.bind(self.addr)
        print("---------------------if close needed, Ctrl+C please---------------------")

    def __del__(self):
        '''
        !-- only work at the period of close normally or Ctrl+C
        and then port would be released
        :return:
        '''
        self.sss.close()

    def setAddr(self, ip = '0.0.0.0'):
        if ip == '0.0.0.0':
            try:
                addr = (sys.argv[1], port)
                print("Input ip {} suceess.".format(addr[0]))
                assert bool(re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", addr[0]))
            except:
                addr = (ipDefault, port)
                print("ipDefault {}.".format(addr[0]))
        else:
            addr = (ip, port)
            print("Set ip {} suceess.".format(addr[0]))
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
    obj = server()
    obj.Recv()

