from deep_neural_network import *
import h5py
import numpy as np
import matplotlib.pyplot as plt


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


train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

m_train = train_x_orig.shape[0]  # 训练样本的数量
m_test = test_x_orig.shape[0]  # 测试样本的数量
num_px = test_x_orig.shape[1]  # 每张图片的宽/高

# 为了方便后面进行矩阵运算，我们需要将样本数据进行扁平化和转置
# 处理后的数组各维度的含义是（图片数据，样本数）
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# 下面我们对特征数据进行了简单的标准化处理（除以255，使所有值都在[0，1]范围内）
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.


# 利用上面的工具函数构建一个深度神经网络训练模型
def dnn_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    '''

    :param X:数据集
    :param Y:数据集标签
    :param layers_dims:指示该深度神经网络用多少层，每层有多少个神经元
    :param learning_rate:学习率
    :param num_iterations:指示需要训练多少次
    :param print_cost:指示是否需要在将训练过程中的成本信息打印出来，好知道训练的进度好坏。
    :return:
    :parameters:  返回训练好的参数。以后就可以用这些参数来识别新的陌生的图片
    '''
    np.random.seed(1)
    costs = []

    # 初始化每层的参数w和b
    parameters = initialize_parameters_deep(layers_dims)

    # 按照指示的次数来训练深度神经网络
    for i in range(0, num_iterations):
        # 进行前向传播
        AL, caches = L_model_forward(X, parameters)
        # 计算成本
        cost = compute_cost(AL, Y)
        # 进行反向传播
        grads = L_model_backward(AL, Y, caches)
        # 更新参数，进行下一轮传播
        parameters = update_parameters(parameters, grads, learning_rate)

        # 打印成本
        if i % 100 == 0:
            if print_cost and i > 0:
                print("训练%i次后成本是: %f" % (i, cost))
            costs.append(cost)

    # 画出成本曲线图
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


# 设置好深度神经网络的层次信息——下面代表了一个4层的神经网络（12288是输入层），
# 第一层有20个神经元，第二层有7个神经元。。。
# 你也可以构建任意层任意神经元数量的神经网络，只需要更改下面这个数组就可以了
layers_dims = [12288, 20, 7, 5, 1]
# 根据上面的层次信息来构建一个深度神经网络，并且用之前加载的数据集来训练这个神经网络，得出训练后的参数
parameters = dnn_model(train_x, train_y, layers_dims, num_iterations=2000, print_cost=True)


def predict(X, parameters):
    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # 进行一次前向传播，得到预测结果
    probas, caches = L_model_forward(X, parameters)

    # 将预测结果转化成0和1的形式，即大于0.5的就是1，否则就是0
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    return p


# 对训练数据集进行预测
pred_train = predict(train_x, parameters)
print("预测准确率是: " + str(np.sum((pred_train == train_y) / train_x.shape[1])))

# 对测试数据集进行预测
pred_test = predict(test_x, parameters)
print("预测准确率是: " + str(np.sum((pred_test == test_y) / test_x.shape[1])))
