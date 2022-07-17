import numpy as np


class hsvInstance():
    def __init__(self, instance):
        '''

        :param instance: a list with mutiple elements, like [0, 68, 59, ...]
        '''
        self.instance = np.array(instance, dtype=np.uint8)

    def __getitem__(self, key):
        return self.instance[key]

    def __lt__(self, other):
        '''

        'self < other' equals self.__lt__(other)
        :param other: a list with the number of elements which equals self.instance's
        '''
        assert len(other) == len(self.instance)
        for i in range(len(self.instance)):
            if not self[i] < other[i]: return False
        return True

    def __gt__(self, other):
        '''

        'self > other' equals self.__gt__(other)
        :param other: a list with the number of elements which equals self.instance's
        '''
        assert len(other) == len(self.instance)
        for i in range(len(self.instance)):
            if not self[i] > other[i]: return False
        return True

    def __eq__(self, other):
        if not len(other) == len(self.instance): return False
        for i in range(len(self.instance)):
            if not self[i] == other[i]: return False
        return True

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __ge__(self, other):
        return self.__gt__(other) or self.__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)


class hsvRange():
    def __init__(self, range):
        '''

        :param range: a list with two sublist which owns three element, like [[0, 68, 59], [20, 255, 255]]
        '''
        self.range = np.array(range, dtype=np.uint8)
        self.low = hsvInstance(self.range[0])
        self.upper = hsvInstance(self.range[1])

    def inRange(self, a):
        '''

        :param a: a hsv instance
        :return:
        '''
        return True if self.low < a and self.upper > a else False