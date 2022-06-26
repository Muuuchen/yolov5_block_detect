import time
import numpy as np
import matplotlib.pyplot as plt
class kalman():
    def __init__(self):
        self.g = 9.85
        self.delta = 0.22
        self.A = np.mat([[1 , self.delta], [0, 1]])
        self.B = np.mat([[-0.5*self.delta**2], [-self.delta]])
        self.xP = np.ones(2)
        self.yP = np.ones(2)
        self.zP = np.ones(2)
        self.X_x = np.mat([[0.0], [0.0]])
        self.X_y = np.mat([[0.0], [0.0]])
        self.X_z = np.mat([[0.0], [0.0]])
        self.Q = np.mat([[1e-5, 0.0], [0.0, 1e-5]])
        self.R = np.mat([[1e-2, 0.0], [0.0, 1e-2]])

    def filter(self, pos, ve):
        Zx = np.mat([[pos[0]], [ve[0]]])
        Zy = np.mat([[pos[1]], [ve[1]]])
        Zz = np.mat([[pos[2]], [ve[2]]])
        X_xpre = self.A.dot(self.X_x)
        X_ypre = self.A.dot(self.X_y)
        X_zpre = self.A.dot(self.X_z) + self.g*self.B

        self.xP = self.A.dot(self.xP).dot(self.A.T) + self.Q
        self.yP = self.A.dot(self.yP).dot(self.A.T) + self.Q
        self.zP = self.A.dot(self.zP).dot(self.A.T) + self.Q

        Kkx = np.linalg.inv(self.xP).dot(self.A.T) / (self.A.dot(np.linalg.inv(self.xP)).dot(self.A.T) + self.R)
        Kky = np.linalg.inv(self.yP).dot(self.A.T) / (self.A.dot(np.linalg.inv(self.yP)).dot(self.A.T) + self.R)
        Kkz = np.linalg.inv(self.zP).dot(self.A.T) / (self.A.dot(np.linalg.inv(self.zP)).dot(self.A.T) + self.R)

        self.X_x = X_xpre + Kkx.dot(Zx - self.A.dot(X_xpre))
        self.X_y = X_ypre + Kky.dot(Zy - self.A.dot(X_ypre))
        self.X_z = X_zpre + Kkz.dot(Zz - self.A.dot(X_zpre))

        self.xP = (np.eye(2) - Kkx * self.A).dot(self.xP)
        self.yP = (np.eye(2) - Kky * self.A).dot(self.yP)
        self.zP = (np.eye(2) - Kkz * self.A).dot(self.zP)

        pos = [self.X_x[0][0], self.X_y[0][0], self.X_z[0][0]]
        ve = [self.X_x[1][0], self.X_y[1][0], self.X_z[1][0]]
        print(pos, ve)
        return pos, ve
pos_save = []
ve_save = []
kf = kalman()
b = time.time()
pos = []
ve = []
while True:
    if ve[0] == 0 and ve[1] == 0 and ve[2] == 0:
        a = time.time()
        kf.delta = a - b
        b = a
        x, y = kf.filter(pos, ve)
        pos_save.append(x), ve_save.append(y)


#
# ex = kalman()
# dis = []
# for i in range(8):
#     pos, ve = ex.filter(listpos[i], listve[i])
#     pos = np.array(pos)
#     dis.append(listpos[i][1])
# xpoints = np.array([0, 8])
# ypoints = np.array(dis)
#
# plt.plot(ypoints)
# plt.show()