# coding=utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 计算平均损失函数
def computeLoss(w, b, points):
    totalLoss = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalLoss += (y - (w * x + b)) ** 2
    # 求平均
    return totalLoss / float(len(points))

# 计算梯度下降
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (2/N) * ((w_current * x + b_current) - y)
        w_gradient += (2/N) * x * ((w_current * x + b_current) - y)
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]

# 训练：循环迭代
def gradient_decent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    # 循环迭代次数
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]

def run():
    # 读取数据
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    init_b = 0
    init_w = 0
    num_iterations = 1000
    print("Starting gradient descent at b={0}, w={1}, loss={2}".format(init_b, init_w, computeLoss(init_w, init_b, points)))
    print("Running...")
    [b, w] = gradient_decent_runner(points, init_b, init_w, learning_rate, 1000)
    print("After {0} iterations b = {1}, w = {2}, loss = {3}".format(num_iterations, b, w, computeLoss(b, w, points)))
    return b, w

if __name__ == '__main__':
    points = np.genfromtxt("data.csv", delimiter=",")
    points = np.array(points)
    fig1 = plt.figure(1)
    plt.subplot(1, 1, 1)
    points_x = points[:, 0]
    points_y = points[:, 1]
    plt.scatter(x=points_x, y=points_y)

    b, w = run()
    xs = np.arange(0, 100, 0.1)
    y = w * xs + b
    plt.plot(xs, y, color='red')
    plt.show()
