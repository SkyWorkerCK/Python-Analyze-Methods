import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import operator
import random
import matplotlib.pyplot as plt


# 手动划分数据集
def SplitDatasets(datasets, labels, testLength):

    """打乱数据"""
    row, col = datasets.shape      # row为其中的数据条数
    """为每一条数据建立引索集合"""
    index = [i for i in range(row)]
    """随机打乱引索顺序"""
    random.shuffle(index)
    """按照打乱之后的引索进行重建数据集"""
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


# KNN 算法模块
def KNN(NewVec, datasets, labels, k):
    # 计算NewVec这个点到数据集datasets其他所有点的欧式距离
    distances = ComputingEuroDistance(datasets, labels, NewVec)
    # 对距离进行从小到大排序，返回排序后的索引号
    sort = distances.argsort()
    # 统计前k个键值对的数量
    classCount = {}
    for i in range(k):
        label = labels[sort[i]]
        classCount[label] =classCount.get(label, 0) + 1

    # 投票机制  ——————  少数服从多数
    # 对各个分类字典进行分类排序
    # sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作
    # list 的 sort 方法返回的是对已经存在的列表进行操作，无返回值，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作
    # print(classCount.items())      dict_items([('A', 1), ('B', 2)])
    Count = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return Count[0][0]

# 进行不同k值下的准确率的可视化
def Plot(accuracy):
    figure = plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    x = [i for i in range(len(accuracy))]
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.plot(x, accuracy, c='red')
    plt.show()


if __name__ == '__main__':
    iris = load_iris()

    # 调用函数进行划分训练集和数据集
    # 分割数据0.2为测试数据，0.8为训练数据
    # x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

    # 手动划分数据集
    datasets = iris.data
    labels = iris.target
    test_Length = 40
    x_train, y_train, x_test, y_test = SplitDatasets(datasets, labels, test_Length)

    # 为每一个accuracy对应的k值建立引索
    index = []
    # 建立准确率集合
    accuracy = list()

    for k in range(2, 100, 1):
        index.append(k)
        # 建立预测结果数据集合
        predictions = list()
        accuracy_each = list()

        # 同一个K值通过划分不同的训练集和测试集进行预测10次
        for l in range(10):
            # 随机划分训练集和测试集
            x_train, y_train, x_test, y_test = SplitDatasets(datasets, labels, test_Length)
            # 预测过程
            for i in range(x_test.shape[0]):
                predictions.append(KNN(x_test[i], x_train, y_train, k))
            score_each = accuracy_score(y_test, predictions)
            # 清空预测结果集
            predictions = []
            accuracy_each.append(score_each)
        if k <= 10 :
            print("当K为{}时，随机划分10次训练集和测试集进行测试的精度分别为{},最优K值为{}".format(k, accuracy_each, max(accuracy_each)))

        # 添加到准确率集合中
        # score = accuracy_score(y_test, predictions)
    #     print("k = {}，accuracy = {}".format(k, score))
    #     accuracy.append(score)
    #
    # print(accuracy)
    # Plot(accuracy)

    # sklearn中的classification_report函数用于显示主要分类指标的文本报告．
    # 在报告中显示每个类的精确度，召回率，F1值等信息
    # print(classification_report(y_test, predictions))

        # 用于显示主要分类指标的文本报告．在报告中显示每个类的精确度，召回率，F1值等信息。

        # print("Score : {}".format(accuracy_score(y_test, predictions)))