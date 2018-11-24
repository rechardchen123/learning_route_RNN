import numpy as np
import h5py
import matplotlib.pyplot as plt

# 加载我们自定义的工具函数
from testCases import *
from dnn_utils import *

# 设置一些画图相关的参数
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)


# 该函数用于初始化所有层的参数w和b
def initialize_parameters_deep(layer_dims):
    '''

    :param layer_dims:这个list列表里面，包含了每层的神经元个数。
     例如，layer_dims=[5,4,3]，表示第一层有5个神经元，第二层有4个，最后一层有3个神经元
    :return:
    parameters -- 这个字典里面包含了每层对应的已经初始化了的W和b。
    例如，parameters['W1']装载了第一层的w，parameters['b1']装载了第一层的b
    '''
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)  # 获取神经网络总共有几层
    # 遍历每一层，为每一层的W和b进行初始化
    for l in range(1, L):
        # 构建并随机初始化该层的W。
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        # 构建并初始化b
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        # 核对一下W和b的维度是我们预期的维度
        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b'] + str(l).shape == (layer_dims[l], 1))
    # 就是利用上面的循环，我们就可以为任意层数的神经网络进行参数初始化，只要我们提供每一层的神经元个数就可以了。
    return parameters


parameters = initialize_parameters_deep([5, 4, 3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# 线性前向传播
def linear_forward(A, W, b):
    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)  # 保存变量A，W，b
    return Z, cache


A, W, b = linear_forward_test_case()

Z, linear_cache = linear_forward()
print("Z = " + str(Z))


# linear_activation_forward 激活函数
def linear_activation_forward(A_prev, W, b, activation):
    '''

    :param A_prev: 上一层得到的A，输入到本层来计算Z和本层的A，第一层时A_prev就是输入特征x
    :param W: 本层相关的W
    :param b: 本层相关的b
    :param activation: 两个字符串，‘sigmoid’和'relu'，指示本层该用哪种激活函数
    :return: 返回本层计算的A和cache
    '''
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation == 'sigmoid':
        A = sigmoid(Z)
    elif activation == 'relu':
        A = relu(Z)

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, Z)
    return A, cache


A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = linear_activation_forward(A, prev, W, b, activation='sigmoid')
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="relu")
print("With ReLU: A = " + str(A))


# 这个函数构建了一个完整的前向传播过程。这个前向传播一共有L层，前面的L-1层用的激活函数是relu，最后一层使用sigmoid。
def L_model_forward(X, parameters):
    '''

    :param X: 输入特征数据
    :param parameters: 这个list列表包含了每一层的参数w和b
    :return:
    '''
    cache = []
    A = X

    # 获取参数列表的长度，这个长度的一半就是神经网络的层数
    # 为什么时一半呢？因为列表是这样的[w1,b1,w2,b2...wl,bl],里面的w1和b1代表了一层
    L = len(parameters) // 2

    #循环L-1次，即进入L-1步前向传播，每一步使用的激活函数都是relu
    for l in range(1,L):
        A_prev = A
        A,caches = linear_activation_forward(A_prev,
                                            parameters['W'+str(l)],
                                            parameters['b'+str(l)],
                                            activation='relu')
        caches.append(cache)# 把一些变量数据保存起来，以便后面的反向传播使用

    # 进行最后一层的前向传播，这一层的激活函数是sigmoid。得出的AL就是y'预测值
    AL, cache = linear_activation_forward(A,
                                          parameters['W'+str(L)],
                                          parameters['b'+str(L)],
                                          activation='sigmoid')
    caches.append(cache)

    assert(AL.shape==(1,X.shape[1]))

    return AL,caches

X,parameters = L_model_forward_test_case()
AL,caches = L_model_forward(X,parameters)
print('AL = '+str(AL))
print('length of caches list = '+str(len(caches)))

# 上面已经完成了前向传播了。下面这个函数用于计算成本（单个样本时是损失，多个样本时是成本）。
# 通过每次训练的成本我们就可以知道当前神经网络学习的程度好坏。
def compute_cost(AL,Y):
    m = Y.shape[1]
    cost = (-1/m)*np.sum(np.multiply(Y,np.log(AL))+np.multiply(1-Y,np.log(1-AL)))

    cost = np.squeeze(cost)# 确保cost是一个数值而不是一个数组的形式
    assert(cost.shape==())

    return cost

Y, AL = compute_cost_test_case()

print("cost = " + str(compute_cost(AL, Y)))

#反向传播，根据后一层的dZ来计算前一层的dW,db和dA

def linear_backward(dZ,cahce):
    '''

    :param dZ: 后面一层的dZ
    :param cahce: 前向传播时保留下来的关于本层的一些变量
    :return:
    '''
    A_prev, W,b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ,cache[0].T)/m
    db = np.sum(dZ,axis = 1,keepdims=True)/m
    dA_prev = np.dot(cahce[1].T,dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev,dW,db

dZ, linear_cache = linear_backward_test_case()

dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))

# linear_activation_backward 用于根据本层的dA计算出本层的dZ. dZ[l]=dA[l]∗g′(Z[l]),g'()表示求Z相当于本层的激活函数的偏导数。
#所以不同的激活函数也有不同的求导公式, sigmoid_backward和relu_backward 在 dnn_utils.py中
def linear_activation_backward(dA,cache,activation):
    '''

    :param dA:本层的dA
    :param cache:前向传播时保存的本层的相关变量
    :param activation:指示该层使用的是什么激活函数: "sigmoid" 或 "relu"
    :return:
    '''
    linear_cache,activation_cache = cache

    if activation == 'relu':
        dZ = relu_backward(dA,activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA,activation_cache)

    # 这里我们又顺带根据本层的dZ算出本层的dW和db以及前一层的dA
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev,dW,db

dAL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))


# 下面这个函数构建出整个反向传播。
def L_model_backward(AL,Y,caches):
    '''

    :param AL:最后一层的A，也就是y'，预测出的标签
    :param Y:真实标签
    :param caches:前向传播时保存的每一层的相关变量，用于辅助计算反向传播
    :return:
    '''
    grads = {}
    L =len(caches) # 获取神经网络层数。caches列表的长度就等于神经网络的层数
    Y = Y.reshape(AL.shape)# 让真实标签的维度和预测标签的维度一致

    # 计算出最后一层的dA，前面文章我们以及解释过，最后一层的dA与前面各层的dA的计算公式不同，
    # 因为最后一个A是直接作为参数传递到成本函数的，所以不需要链式法则而直接就可以求dA（A相当于成本函数的偏导数）
    dAL = -(np.divide(Y,AL)-np.divide(1-Y,1-AL))

    # 计算最后一层的dW和db，因为最后一层使用的激活函数是sigmoid
    current_cache = caches[-1]
    grads['dA'+str(L-1)],grads['dW'+str(L)],grads['db'+str(L)] = linear_activation_backward(dAL,
                                                                                            current_cache,
                                                                                            activation='sigmoid')

    # 计算前面L-1层到第一层的每层的梯度，这些层都使用relu激活函数
    for c in reversed(range(1,L)): # reversed(range(1,L))的结果是L-1,L-2...1。是不包括L的。第0层是输入层，不必计算。
        grads['dA'+str(c-1)],grads['dW'+str(c)],grads['db'+str(c)] = linear_activation_backward(
            grads['dA'+str(c)],
            caches[c-1],
            # 这里我们也是需要当前层的caches，但是为什么是c-1呢？因为grads是字典，我们从1开始计数，而caches是列表，
            # 是从0开始计数。所以c-1就代表了c层的caches。数组的索引很容易引起莫名其妙的问题，大家编程时一定要留意。
            activation='relu'
        )

    return grads

AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dA1 = "+ str(grads["dA1"]))

#通过上面的反向传播，我们得到了每一层的梯度（每一层w和b相当于成本函数的偏导数）。
# 下面的update_parameters函数将利用这些梯度来更新/优化每一层的w和b，也就是进行梯度下降。
def upgrade_parameters(parameters,grads,learning_rate):
    '''

    :param parameters:每一层的参数w和b
    :param grads:每一层的梯度
    :param learning_rate:是学习率，学习步进
    :return:
    '''
    L =len(parameters)//2  # 获取层数。//除法可以得到整数

    for l in range(1,L+1):
        parameters['W'+str(l)] = parameters['W'+str(l)] - learning_rate*grads['dW'+str(l)]
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return parameters

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = " + str(parameters["W1"]))
print ("b1 = " + str(parameters["b1"]))
print ("W2 = " + str(parameters["W2"]))
print ("b2 = " + str(parameters["b2"]))


























