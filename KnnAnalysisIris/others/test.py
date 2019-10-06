# coding:utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.datasets import load_iris
import math
import operator
import matplotlib.pyplot as plt


"""计算欧氏距离"""
def ComputingEuroDistance(datasets, labels, NewVec):
    row, col = datasets.shape
    NewVec = np.tile(NewVec, (row, 1))
    distance = datasets - NewVec
    sqDistance = distance ** 2
    ResSum = sqDistance.sum(axis=1)  # 求矩阵每一行的和
    Res = np.sqrt(ResSum)

    return Res


"""图像可视化"""
def Plot(datasets, labels):
    plt.figure()
    row, col = datasets.shape
    ax = plt.subplot(1, 1, 1)
    for i in range(row):
        if labels[i] == 'A':
            plt.scatter(datasets[i, 0], datasets[i, 1], c='red')
        elif labels[i] == 'B':
            plt.scatter(datasets[i, 0], datasets[i, 1], c='yellow')
    plt.show()

if __name__ == '__main__':

    # 加载iris数据集
    iris = load_iris()
    iris_data = np.array(iris.data)
    iris_label = np.array(iris.target)

    # 定义每一轮准确率的集合
    accuracy = []

    # 采用交叉验证的方法进行验证
    # 特别的是StratifiedKfold将验证集的正负样本比例，保持和原始数据的正负样本比例相同
    kf = StratifiedKFold(n_splits=10)
    for train_index, test_index in kf.split(iris_data, iris_label):
        datasets = np.array(iris_data[train_index])
        labels = np.array(iris_label[train_index])
        print("train_index:{}, \ntest_index:{}".format(train_index, test_index))
        k = 3  # 指定最短的前 3 个

        # 在主函数中完成　KNN　算法
        for i in test_index:
            # 通过i找到需要进行KNN分类的向量
            NewVec = np.array(iris_data[i])                 # 待判断的数据点
            distances = ComputingEuroDistance(datasets, labels, NewVec)  # 计算NewVec到所有其它节点的欧氏距离
            sort = distances.argsort()  # 对距离进行按照从小到大的顺序进行排序
            classCount = {}  # 统计前 k 个键值对的数量
            for j in range(k):
                label = labels[sort[j]]
                classCount[label] = classCount.get(label, 0) + 1

            Count = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

            # 预测类别为 Count[0][0]
            wrong = 0
            if iris_label[i] != Count[0][0]:
                wrong = wrong + 1

        # 计算正确率
        total = len(test_index)
        right = (total - wrong) / total
        accuracy.append(right)

    # 输出最后的正确率集合
    acc = sum(accuracy) / len(accuracy)
    print("Accuracy : {}".format(acc))

