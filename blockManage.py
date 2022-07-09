import numpy as np
from multipledispatch import dispatch

"""
features:
    color label
    lie down(the upward surface) or stand up
    diameter
between frames:
    fraction coverage
"""


blockHeight = 200 # millimeter
blockSize = np.array([500, 425, 350, 275, 200]) # millimeter
sign = [5, 4, 3, 2, 1]

def adaptx(x): return max(min(x,639),0)

def adapty(y): return max(min(y,479),0)

def judgeDiaEdge(isside):
    pass

def sizeNearest(size, isside, velres, bias = 37.5):
    '''

    :param size: [Len, Wid]---meter
    :param isside: -1 means side lie down, 1 menas side stand
    :param velres: degree
    :return:
    '''
    assert bias <= (blockSize[0] - blockSize[1]) / 2
    print(size[0] * size[1])
    mysize = size[1] * 1e3 if isside == -1 or abs(velres) > 40 else size[0] * 1e3  # millimeter
    try: return sign[list(abs(mysize - blockSize) < bias).index(True)]
    except: return -1

def findDiaEdge(center, finalMask):
    '''

    :param center: [mx, my]---pixel
    :param finalMask:
    :return: mx---Wid---1, my---Hei---0
    '''
    cX = int(center[0])
    cY = int(center[1])

    blackNum = 0
    yBias = 0
    xBias = 0
    while (blackNum <= 10):
        if finalMask[adapty(cY - yBias)][adaptx(cX)] == 0: blackNum += 1
        yBias += 1
        if adapty(cY - yBias) == 479 or adapty(cY - yBias) == 0 or adaptx(cX) == 639 or adaptx(cX) == 0: break

    blackNum = 0
    while(blackNum <= 10):
        if not finalMask[adapty(cY)][adaptx(cX + xBias)] == 0: blackNum += 1
        xBias += 1
        if adapty(cY) == 479 or adapty(cY) == 0 or adaptx(cX + xBias) == 639 or adaptx(cX + xBias) == 0: break
    return 1 if xBias > yBias else 0 # mx---Wid---1, my---Hei---0

def sizeNearest2(size, isside, center, finalMask, bias = 37.5):
    '''

    :param size: [Len, Wid]---meter
    :param isside: -1 means side lie down, 1 menas side stand
    :param center: [mx, my]---pixel
    :return:
    '''
    assert bias <= (blockSize[0] - blockSize[1]) / 2
    print(size[0] * size[1])
    size = max(size[0], size[1]) * 1e3 if not isside == 0 else size[findDiaEdge(center, finalMask)] * 1e3  # millimeter
    try: return sign[list(abs(size - blockSize) < bias).index(True)]
    except: return -1


if __name__ == "__main__":
    print(sizeNearest(0.235))