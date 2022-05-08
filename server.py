import numpy as np
import cv2
from socket import *


s = socket(AF_INET, SOCK_DGRAM)
addr = ('192.168.10.213', 8080)          # 127.0.0.1表示本机的IP，相当于我和“自己”的关系
s.bind(addr)
while True:
    data, _ = s.recvfrom(921600)
    receive_data = np.frombuffer(data, dtype='uint8')
    r_img = cv2.imdecode(receive_data, 1)
    r_img = r_img.reshape(480, 640, 3)

    cv2.putText(r_img, "server", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('server_frame', r_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
