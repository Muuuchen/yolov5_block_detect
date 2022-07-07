import numpy as np


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

def sizeNearest(size, bias = 30):
    '''

    :param size: meter
    :return:
    '''
    assert bias < blockSize[0] - blockSize[1]
    size = size * 1e3 # millimeter
    try: return sign[list(abs(size - blockSize) < bias).index(True)]
    except: return -1

if __name__ == "__main__":
    print(sizeNearest(0.235))