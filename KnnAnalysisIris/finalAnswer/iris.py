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
def Plot(x, accuracy):
    figure = plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    plt.xlabel('k values')
    plt.ylabel('Accuracy')
    plt.plot(x, accuracy, c='blue')
    plt.show()

# 主函数
if __name__ == '__main__':
    # 加载数据集
    iris = load_iris()
    datasets = iris.data
    labels = iris.target
    test_Length = 40

    # 手动划分数据集
    x_train, y_train, x_test, y_test = SplitDatasets(datasets, labels, test_Length)
    # 设置精确度集合以及其对应的引索集合
    index = []
    accuracy = list()
    # 采用每一次K值遍历计算精确度之后的平均值进行可视化，所以这里定义最终的平均精确度
    accuracy_average = []
    # 创建一个零矩阵
    accuracy_sum = np.zeros(98)

    for l in range(10):
        # 手动划分数据集
        x_train, y_train, x_test, y_test = SplitDatasets(datasets, labels, test_Length)

        # K从2-100一次遍历计算精确度
        for k in range(2, 100, 1):
            # accuracy对应的引索值
            if len(index) < 98:
                index.append(k)
            # 定义一个预测结果集
            predictions = []

            # 预测
            for i in range(x_test.shape[0]):
                predictions.append(KNN(x_test[i], x_train, y_train, k))

            # 添加到准确度集合中
            score = accuracy_score(y_test, predictions)
            accuracy.append(score)
        # 将每一次对应位置上的精确度进行求和
        accuracy_sum = np.array(accuracy) + accuracy_sum
        # 找出accuracy中精确度最大的值
        max_accuracy = max(accuracy)
        # 每一轮k值变化过程结束后最大精确度对应的K值集合
        K_ = []
        # 筛选出最大精确度对应的K值
        for g in range(len(accuracy)):
            if accuracy[g] == max_accuracy:
                # K_.append(accuracy[g])
                K_.append(index[g])
        # 打印
        print("第{}轮运行过程中，最优K值集合为{}".format(l+1, K_))

        # Plot(accuracy)  输出每一次运行的子图
        # 重置精确度和引索
        accuracy = []

    # 通过最后的平均值进行绘制，更能体现算法性能
    # 可视化K值与精度的关系
    Plot(index, accuracy_sum/10)

    # 均值计算之后找到最大精度对应的K值
    K__ = []
    # 对每一个accuracy_sum里的元素进行求平均值
    accuracy_avg = accuracy_sum / 10
    # 找出accuracy_avg中的最大值
    accuracy_avg_max = max(accuracy_avg)
    # 遍历accuracy_avg集合，寻找平均图的最优K值集合
    for q in range(len(accuracy_avg)):
        if accuracy_avg[q] == accuracy_avg_max:
            K__.append(index[q])
    print("平均图的最优K值集合为：{}".format(K__))
