imshowFlag = True
ipDefault = '192.168.10.211'
portDefault = 8080
ipclient = '192.168.10.197'
R2 = False




cameras = ['0', '8'] if R2 else ['0']
with open('./forTest/stream.txt', 'w') as f:
    for camera in cameras:
        f.write(camera)
        if not cameras.index(camera) + 1 == len(cameras): f.write('\n')