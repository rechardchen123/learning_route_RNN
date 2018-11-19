import numpy as np
import matplotlib.pyplot as plt
import h5py  # 加载训练数据集，将数据保存成HDF格式，Hierarchical Data Format对大数据进行组织和存储的
# 文件格式，常用于保存数据
import skimage.transform as tf  # 用于缩放图片


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', 'r')  # 加载训练数据集
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # 从训练数据中提取图片的特征数据
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # 从训练数据中提取出图片的标签数据

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', 'r')  # 加载测试数据集
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])

    classes = np.array(test_dataset["list_classes"][:])  # 加载标签类别数据，这里的类别只有两种，1代表有猫，0代表无猫

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # 把数组的维度从(209,)变成(1, 209)，这样
    # 好方便后面进行计算
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))  # 从(50,)变成(1, 50)

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def sigmoid(z):
    '''

    :param z: 数值或者numpy数组
    :return: 范围[0，1]的值
    '''
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    '''
    用于初始化权重数组w和偏置b
    :param dim: w的大小，dim在当前时12288
    :return: w-权重数组 b-偏置
    '''
    w = np.zeros((dim, 1))
    b = 0
    return w, b


def propagate(w, b, X, Y):
    '''

    :param w: 权重数组 维度是(12288,1)
    :param b: 偏置
    :param X: 图片的特征数据，维度是(12288,209)
    :param Y: 图片对应的标签 0或1 维度是(1,209)
    :return:
    cost --成本
    dw -- w的梯度
    db -- b的梯度
    '''
    m = X.shape[1]
    # 前向传播
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
    # 反向传播
    dZ = A - Y
    dw = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m
    # 将dw和db保存在字典里
    grads = {'dw': dw, 'db': db}
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    '''

    :param w:权重数组，维度是 (12288, 1)
    :param b:偏置bias
    :param X:图片的特征数据，维度是 (12288, 209)
    :param Y:图片对应的标签，0或1，0是无猫，1是有猫，维度是(1,209)
    :param num_iterations:指定要优化多少次
    :param learning_rate:学习步进，是我们用来控制优化步进的参数
    :param print_cost:为True时，每优化100次就把成本cost打印出来,以便我们观察成本的变化
    :return:
    params -- 优化后的w和b
     costs -- 每优化100次，将成本记录下来，成本越小，表示参数越优化
    '''
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)  # 计算得出梯度和成本
        # 从字典中取出梯度
        dw = grads['dw']
        db = grads['db']

        # 进行梯度下降，更新参数，使其越来越优化，使成本越来越小
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # 将成本记录下来
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("优化%i后的成本是：%f" % (i, cost))

    params = {'w': w, 'b': b}
    return params, costs


def predict(w, b, X):
    '''

    :param w:权重数组，维度是 (12288, 1)
    :param b:偏置bias
    :param X:图片的特征数据，维度是 (12288, 图片张数)
    :return:
    Y_prediction -- 对每张图片的预测结果
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))

    A = sigmoid(np.dot(w.T, X) + b)  # 通过这行代码来对图片进行预测

    # 上面得出的预测结果是小数的形式，为了方便后面显示，我们将其转换成0和1的形式（大于等于0.5就是1/有猫，小于0.5就是0/无猫）
    for i in range(A.shape[1]):
        if A[0, i] >= 0.5:
            Y_prediction[0, i] = 1
    return Y_prediction



# 调用上面定义的函数将数据加载到各个变量中
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

index = 8
plt.imshow(train_set_x_orig[index])
print("标签为" + str(train_set_y[:, index]) + ", 这是一个'" + classes[np.squeeze(train_set_y[:, index])].decode(
    "utf-8") + "' 图片.")

# 我们要清楚变量的维度，否则后面会出很多问题。下面我把他们的维度打印出来
print("train_set_x_orig shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_orig shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))

# 上面train_set_x_orig的各维度的含义分别是(样本数，图片宽，图片长，3个RGB通道)

# 后面要用到样本数和长宽像素值，下面我分别把它们提取出来
m_train = train_set_x_orig.shape[0]
m_test = train_set_x_orig.shape[0]
num_px = test_set_x_orig.shape[1]  # 由于我们的图片是正方形的，所以长宽相等

print("训练样本数: m_train = " + str(m_train))
print("测试样本数：m_test = " + str(m_test))
print("每张图片的宽/高：num_px = " + str(num_px))

# 为了方便后面进行矩阵运算，我们需要将样本数据进行扁平化和转置
# 处理后的数组各维度的含义是（图片数据，样本数）
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))

# 下面我们对特征数据进行了简单的标准化处理（除以255，使所有值都在[0，1]范围内）
# 为什么要对数据进行标准化处理呢？简单来说就是为了方便后面进行计算，详情以后再给大家解释
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

def model(X_train,Y_train,X_test,Y_test,num_iterations = 2000,learning_rate=0.5,print_cost =False):
    '''

    :param X_train: 训练图片,维度是(12288, 209)
    :param Y_train:训练图片对应的标签,维度是 (1, 209)
    :param X_test:测试图片,维度是(12288, 50)
    :param Y_test:测试图片对应的标签,维度是 (1, 50)
    :param num_iterations:需要训练/优化多少次
    :param learning_rate:学习步进，是我们用来控制优化步进的参数
    :param print_cost:为True时，每优化100次就把成本cost打印出来,以便我们观察成本的变化
    :return:
    d
    '''
    # 初始化待训练的参数
    w, b = initialize_with_zeros(X_train.shape[0])
    # 使用训练数据来训练/优化参数
    parameters, costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)

    #从字典中分别取出训练好的w和b
    w = parameters['w']
    b = parameters['b']

    #使用训练好的w和b来分别对训练图片和测试图片进行预测
    Y_prediction_train = predict(w,b,X_train)
    Y_prediction_test = predict(w,b,X_test)

    #打印出预测结果
    print("对训练图片的预测准确率为: {}%".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("对测试图片的预测准确率为: {}%".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d

d = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations=2000,learning_rate=0.005,print_cost=True)

costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost') # 成本
plt.xlabel('iterations (per hundreds)') # 横坐标为训练次数，以100为单位
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()