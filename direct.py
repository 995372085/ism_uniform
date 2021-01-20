# ---coding: utf-8 ---
# @Time:2021/1/14 14:49
# @Author:100516
# @Description:

from queue import Queue

def compare(q) :
    print(q)
    count = 0
    for i in range(0, len(q)-1) :
        if(q[i] < q[i+1]) :
            count+=1

    return True if count >= 3 else False


if __name__ == '__main__':
    # clsQueue = Queue(5)
    # while True:
    #     yVal = input('请输入')
    #     if(len(clsQueue.queue) >= 5) :
    #         clsQueue.get()
    #
    #     clsQueue.put(yVal)
    #     print(compare(clsQueue.queue))
    a=765.0243705805557
    print(int(a))