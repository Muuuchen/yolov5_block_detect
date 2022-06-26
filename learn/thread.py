import threading
from threading import Lock, Thread
import time, os


# def run(n):
#     print("task", n)
#     time.sleep(1)
#     print("2s")
#     time.sleep(1)
#     print("1s")
#     time.sleep(1)
#     print("0s")
#     time.sleep(1)\
#
# cnt = 100
# def work1():
#     global cnt
#     for i in range(3):
#         cnt+=1
#         print("in work1 cnt is : %d" % cnt)
# def work2():
#     global cnt
#     for i in range(5):
#         cnt+=2
#         print("in work2 cnt is : %d" % cnt)

# def work(n,semaphore):
#     global cnt
#     semaphore.acquire()
#     time.sleep(3)
#     print('run the thread:%s\n' % n)
#     semaphore.release()
#
# if __name__ == '__main__':
#     cnt =5
#     semaphore = threading.BoundedSemaphore(5) #最多允许5个线程同时运行
#     for i in range(22):# 每个线程输出一次n
#         t = threading.Thread(target=work, args=('t - %s'%i , semaphore))
#         t.start()
#     while threading.active_count() != 1:
#         pass
#     else:
#         print('end')
#

# class MyThread(threading.Thread):
#     def __init__(self, n):
#         super(MyThread, self).__init__()#子类把父类的__init__()放到自己的__init__()当中，这样子类就有了父类的__init__()的那些东西
#         self.n = n
#
#     def run(self):
#         print("task", self.n)
#         time.sleep(1)
#         print("2s")
#         time.sleep(1)
#         print("1s")
#         time.sleep(1)
#         print("0s")
#         time.sleep(1)
#
# if __name__ == '__main__':
#     t1 = MyThread('t1')
#     t2 = MyThread('t2')
#     t1.start()
#     t2.start()

event = threading.Event() #设置一个线程同步对象
def lighter():
    count = 0
    event.set()         #初始者为绿灯 True
    while True:
        if 5 < count <=10:
            event.clear()  #红灯，清除标志位 False
            print("\33[41;lmred light is on...\033[0m]")#颜色特效控制：
        elif count > 10:
            event.set()    #绿灯，设置标志位
            count = 0
        else:
            print('\33[42;lmgreen light is on...\033[0m')

        time.sleep(1)
        count += 1


def car(name):
    while True:
        if event.is_set():     #判断是否设置了标志位
            print('[%s] running.....'%name)
            time.sleep(1)
        else:
            print('[%s] sees red light,waiting...'%name)
            event.wait()
            print('[%s] green light is on,start going...'%name)


# startTime = time.time()
light = threading.Thread(target=lighter,)
light.start()

car = threading.Thread(target=car,args=('MINT',))
car.start()
endTime = time.time()