imshowFlag = True
ipDefault = '192.168.10.211'
portDefault = 8080
ipclient = '192.168.10.197'
R2 = True

import numpy as np
R1hsv = np.array([[0, 68, 59], [20, 255, 255]], dtype=np.uint8)  # top
R2hsv = np.array([[150, 68, 59], [179, 255, 255]], dtype=np.uint8)  # side
Bhsv = np.array([[48, 160, 0], [144, 255, 255]], dtype=np.uint8)
B1hsv = np.array([[48, 43, 0], [99, 255, 255]], dtype=np.uint8)  # top
B2hsv = np.array([[100, 43, 0], [144, 255, 255]], dtype=np.uint8)  # side

