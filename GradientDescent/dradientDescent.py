#梯度下降算法实验
#%matplotlib inline

import numpy as np
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
    for i in range(len(dataSet)):
        result += (func_h(theta, area[i], rooms[i])-price[i])**2
    return result * 0.5

def func_dj(theta, index):
    result = 0
    for i in range(len(dataSet)):
        xj = 1
        if index==1:
            xj = area[i]
        elif index==2:
            xj = rooms[i]
        result += (func_h(theta, area[i], rooms[i])-price[i])*xj
    return result

def gradientDescent(rooms, price, area):
    thetas = []
    theta = [1, 0.5, 0.5]
    lr = 0.000000001
    epochs = 200
    gd = [0, 0, 0]
    thetas.append(theta.copy())
    for i in range(epochs):
        for index in range(len(theta)):
            gd[index] = func_dj(theta, index)
        for j in range(len(theta)):
            theta[j] = theta[j] - gd[j]*lr
        thetas.append(theta.copy())
    return thetas

def demo_gd():
    thetas = gradientDescent(rooms, price, area)
    print(thetas)
    line_x = list(range(len(thetas)))
    line_y = list(map(func_j, thetas))
    print(line_y)
    plt.scatter(line_x, line_y, c='b')
    plt.show()

if __name__ == '__main__':
    demo_gd()
    # gradientDescent(rooms, price, area)
