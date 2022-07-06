import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


def visualization(posList, horres, velres):
    plt. clf()
    fig = plt.gcf()
    ax = fig.gca(projection='3d')
    for i in range(len(posList)):
        ax.scatter(posList[i][0], posList[i][1], posList[i][2], c='r', marker='o', linewidth=4)
    alp = [1, 0, 0]
    h  = (horres/180)*math.pi
    v = (velres / 180) * math.pi

    alp_l = [10*math.sin(h)*math.sin(v), 10*math.sin(h)*math.cos(v), 10*math.cos(h)]
    destinatioin = alp_l + posList[0]

    plt.show()
    plt.pause(0)
