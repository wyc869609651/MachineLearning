#梯度下降算法实验
#%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
import math
dataSet = [[2104, 3, 400],
            [1600, 3, 330],
            [2400, 3, 369],
            [1416, 2, 342],
            [3000, 4,  540]]
price = []
rooms = []
area = []
for data in range(0, len(dataSet)):
    area.append(dataSet[data][0])
    rooms.append(dataSet[data][1])
    price.append(dataSet[data][2])

def func_h(theta, x1, x2):
    return theta[0]+theta[1]*x1+theta[2]*x2

def func_j(theta):
    result = 0
    for m in range(0, len(dataSet)):
        result += math.pow(func_h(theta, area[m], rooms[m])-price[m], 2)
    return result*0.5

def func_dj(theta, index):
    result = 0
    for m in range(0, len(dataSet)):
        xj = 1
        if index == 1:
            xj = area[m]
        elif index == 2:
            xj = rooms[m]
        result += (func_h(theta, area[m], rooms[m])-price[m]) * xj
    return result

def gradientDescent(rooms, price, area):
    theta = [1, 0.001, 0.9]
    lr = 0.01
    epochs = 10
    thetas = []
    thetas.append(np.array(theta))
    for i in range(epochs):
        dtheta = []
        # 计算梯度
        for j in range(len(theta)):
            dtheta.append(func_dj(theta, j))
        norm = np.linalg.norm(dtheta, keepdims=True)
        dtheta = dtheta/norm
        # 梯度下降
        theta -= dtheta * lr
        thetas.append(theta)
    return thetas

def demo_gd():
    thetas = gradientDescent(rooms, price, area)
    errors = list(map(func_j, thetas))
    print(thetas)
    print(errors)
    line_x = list(range(len(thetas)))
    plt.plot(line_x, errors, c='b')
    plt.scatter(line_x, errors, c='r')
    plt.show()

if __name__ == '__main__':
    demo_gd()
    # gradientDescent(rooms, price, area)
