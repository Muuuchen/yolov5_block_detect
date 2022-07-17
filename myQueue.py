class Queue():
    def __init__(self, maxsize):
        self.Q = []
        self.maxsize = maxsize
    def empty(self):
        return 1 if len(self.Q) == 0 else 0
    def full(self):
        return 1 if len(self.Q) == self.maxsize else 0
    def push(self,element):
        if len(self.Q) < self.maxsize:
            self.Q.append(element)
            return True
        else: return False
    def pop(self):
        if len(self.Q):
            out = self.Q[0]
            del(self.Q[0])
            return out
        else:
            return -1
    def printQ(self):
        print("队列元素{},长度{}：".format(self.Q, len(self.Q)))
    def getmax(self):
        outdict ={}
        for item in self.Q:
            outdict[item] = outdict.setdefault(item,0)+1
        maxvalue = 0
        for key,value in outdict.items():
            if value > maxvalue:
                maxvalue = value
                maxkey = key
        return maxkey
    def clear(self):
        self.Q = []
