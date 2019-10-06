import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import operator
import random
import matplotlib.pyplot as plt

# 手动划分数据集
def SplitDatasets(datasets, labels, testLength):

    """手动打乱数据"""
    row, col = datasets.shape
    """为每一条数据建立引索集合"""
    index = [i for i in range(row)]
    """随机打乱引索的顺序"""
    random.shuffle(index)
    datasets = datasets[index]
    labels = labels[index]

    """按照比例进行切分数据集"""
    x_train = datasets[testLength:]
    x_test = datasets[:testLength]
    y_train = labels[testLength:]
    y_test = labels[:testLength]

    return x_train, y_train, x_test, y_test

"""计算欧式距离"""
def ComputingEuroDistance(datasets, labels, NewVec):
    row, col = datasets.shape
    NewVec = np.tile(NewVec, (row, 1))
    distance = datasets - NewVec
    sqDistance = distance ** 2
    ResSum = sqDistance.sum(axis=1)
    Res = np.sqrt(ResSum)

    return Res

# KNN 模块
def KNN(NewVec, datasets, labels, k):
    distances = ComputingEuroDistance(datasets, labels, NewVec)
    sort = distances.argsort()
    classCount = {}
    for i in range(k):
        label = labels[sort[i]]
        classCount[label] = classCount.get(label, 0) + 1

    Count = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return Count[0][0]

# 可视化
def Plot(accuracy):
    figure = plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    x = [i for i in range(len(accuracy))]
    plt.xlabel('k values')
    plt.ylabel('Accuracy')
    plt.plot(x, accuracy, c='blue')
    plt.show()


# 主函数
if __name__ == '__main__':
    iris = load_iris()
    datasets = iris.data
    labels = iris.target
    test_Length = 40

    # 手动划分数据集
    x_train, y_train, x_test, y_test = SplitDatasets(datasets, labels, test_Length)

    index = []
    accuracy = list()

    for k in range(2, 100, 1):
        index.append(k)
        predictions = list()

        # 预测
        for i in range(x_test.shape[0]):
            predictions.append(KNN(x_test[i], x_train, y_train, k))

        # 添加到准确度集合中
        score = accuracy_score(y_test, predictions)
        print("k = {}，accuracy = {}".format(k, score))
        accuracy.append(score)

    # 可视化K值与精度的关系
    Plot(accuracy)

    print(classification_report(y_test, predictions))
