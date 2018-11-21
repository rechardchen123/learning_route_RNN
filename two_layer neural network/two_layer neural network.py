import numpy as np
import matplotlib.pyplot as plt
import sklearn  # 这个库是用于数据挖掘，数据分析和机器学习的库，例如它里面就内置了很多人工智能函数
import sklearn.datasets
import sklearn.linear_model

from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from testCases import *

np.random.seed(1)# 设置一个随机数种子，来保证后面我的代码产生的随机数与你们电脑上产生的一样,这样我的结果才能和你们运行的结果一样

# 加载训练数据。这一次我们没有用真实的数据集，而是在planar_utils文件中用代码生成了一些虚拟的数据集。
# 这个数据集由400个样本组成。这400个样本是400个颜色点。输入特征X是点的横纵坐标，标签Y是点的颜色标签（0表示红色，1表示蓝色）
# 下面的图片展示了这400个颜色点。
# 我们的目标就是通过训练一个神经网络来通过坐标值判断在下图坐标系中某一点可能的颜色，
# 例如坐标（-4，2）的点可能是什么颜色，（-4，3）最可能是什么颜色。将红色和蓝色的点区分出来。

X, Y = load_planar_dataset()

# 下面用scatter来将数据集中的400个点画出来。
# X[0, :]表示400点的横坐标，X[1, :]表示纵坐标，c=Y.ravel()是指定400个点的颜色，s=40指定点的大小，
# cmap指定调色板，如果用不同的调色板，那么Y的值对应的颜色也会不同。用plt.cm.Spectral这个调色板时，Y等于0指代红色，1指代蓝色。
# 你可能会有疑问，为什么不直接用c=Y,而用c=Y.ravel()，它们只是维度表示方式不同，
# Y的维度是(1,400),Y.ravel()的维度是(400,)，scatter这个库函数需要后面的形式。
plt.scatter(X[0,:],X[1,:],c = Y.ravel(),s=40,cmap=plt.cm.Spectral)

shape_X = X.shape
shape_Y = Y.shape
m = Y.shape[1]   #样本数量

print('X的维度是： '+str(shape_X))
print('Y的维度是: '+str(shape_Y))
print('训练样本的个数是： '+str(m))

#生成LogisticRegressionCV类的一个对象，LogisticRegressionCV类实现了一个单神经元网络
clf = sklearn.linear_model.LogisticRegressionCV()

# 将数据集传入对象中进行训练。像学习率和训练次数等超参数都有默认值，所以我们只需要简单地传入数据集就可以了。
# 这个方法会根据数据集进行训练，并将训练好的w和b参数保存在对象clf中，后面就可以用这些参数进行预测。
clf.fit(X.T,Y.T.ravel())

# 用clf对象来对数据集进行预测。我们为了简便起见，只用了训练数据集，其实应该用测试数据集的。
# 返回的结果LR_predictions中包含的是400个0或1，表示对400个点颜色的预测结果。
LR_predictions = clf.predict(X.T)

# 打印出预测准确度。下面是用了自定义的一个算法来求准确度，其实也可以简单地用clf的方法来求——clf.score(X.T,Y.T.ravel())
print ('预测准确度是: %d ' % float((np.dot(Y, LR_predictions) + np.dot(1 - Y,1 - LR_predictions)) / float(Y.size) * 100) + '% ')

# 画出预测结果图。
# 可以看到，这个单神经元网络只是简单地认为在坐标上面部分的点大致就是红色的，位于坐标下半部分的就是蓝色。
# 还是那句话，如果python功底不深的话，没有必要进去看plot_decision_boundary的实现代码。
# 我简单直观地描述下这个函数。首先，在这个函数里面，会将整个坐标的点（不仅仅是待预测花形点集X）传入到clf.predict中，
# 来得出坐标中每一个点的颜色预测值，然后根据它们画出底图（就是坐标上半部是红色下半部是蓝色的图）。
# 最后，再将待预测点集X画到底图上。通过底图和待预测点集的重叠，就可以很直观地看出神经网络预测的精准度。
# 例如，本例中预测得就很不准确，单神经元网络只是简单地认为坐标上部的点是红色，但实际上上面还有很多蓝色点，而且下面也有很多红色点。
plot_decision_boundary(lambda x: clf.predict(x), X, Y.ravel())

# 初始化参数w和b。
# 这个在单神经元实战时也有，但是那时是全都初始化为0，在多神经网络中是万万不可全初始化为0的
def initialize_parameters(n_x,n_h,n_y):
    '''
    参数
    :param n_x:输入层的神经元个数
    :param n_h: 隐藏层的神经元个数
    :param n_y: 输出层的神经元个数
    :return:返回一个神经元的参数字典
    '''
    np.random.seed(2)
    # 随机初始化第一层（隐藏层）相关的参数w.
    # 每一个隐藏层神经元都与输入层的每一个神经元相连。每一个相连都会有一个对应的参数w。
    # 所以W1的维度是（n_h, n_x）,表示（隐藏层的神经元个数，输入层神经元个数）
    W1 = np.random.randn(n_h,n_x)*0.01

    # 将第一层的参数b赋值为0，因为w已经非0了，所以b可以为0
    # 因为每一个神经元只有一个对应的b，所以b1的维度是(n_h, 1)，表示（隐藏层神经元个数，1）
    b1 = np.zeros(shape=(n_h,1))

    #初始化第二层参数
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros(shape=(n_y,1))

    #将初始化好的参数放入字典中
    parameters = {'W1':W1,
                  'b1':b1,
                  'W2':W2,
                  'b2':b2}
    return parameters

# 这是针对于initialize_parameters函数的单元测试
#在实现期间，出现了问题，但是又很难定位问题出在哪里~~ 所以在每个函数后面都会加入一个单元测试
# 如果实现时，发现某个函数的单元测试的输出结果不一致，那么问题应该就出现在那个函数里面。
n_x, n_h, n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x,n_h,n_y)
print('W1= '+str(parameters['W1']))
print('b1= '+str(parameters['b1']))
print('W2= '+str(parameters['W2']))
print('b2= '+str(parameters['b2']))

# 初始化了参数后，前向传播算法
def forward_propagation(X, parameters):
    '''
    参数
    :param X: 输入特征，维度（横纵坐标，样本数）
    :param parameters: 参数W和b
    :return: A2-The sigmoid output of the second activation
            cache- a dictionary containing 'Z1','A1','Z2','A2'
    '''
    #从字典中取出参数
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    #实现前向传播
    Z1 = np.dot(W1,X) +b1
    A1 = np.tanh(Z1) # 第一层的激活函数我们使用tanh。numpy库里面已经帮我们实现了tanh工具函数
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2) #第二层我们使用sigmoid，因为我们要解决的这个问题属于二分问题。这个函数在planar_utils里面实现

    # 将这些前向传播时得出的值保存起来，因为在后面进行反向传播计算时会用
    cache = {'Z1':Z1,
             'A1':A1,
             'Z2':Z2,
             'A2':A2}
    return A2, cache

# 单元测试
X_assess, parameters = forward_propagation_test_case()
A2,cache = forward_propagation(X_assess,parameters)
print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

# 这个函数被用来计算成本
def compute_cost(A2,Y,parameters):
    '''
    参数：
    :param A2:神经网络最后一层的输出结果
    :param Y:数据的颜色标签
    :param parameters:神经网络的参数
    :return:返回计算成本
    '''
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2),Y)+np.multiply((1-Y),np.log(1-A2))
    cost = - np.sum(logprobs)/m

    return cost

A2, Y_assess, parameters = compute_cost_test_case()
print('cost = '+str(compute_cost(A2, Y_assess, parameters)))

#反向传播
def back_propagation(parameters,cache,X,Y):
    '''
    参数:
    :param parameters:参数W和b
    :param cache: 前向传播时保存的数据
    :param X: 输入特征
    :param Y: 标签
    :return: 计算的梯度值
    '''
    m = X.shape[1] #获取样本个数
    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache['A1']
    A2 = cache['A2']

    dz2 = A2 - Y
    dW2 = (1/m)*np.dot(dz2,A1.T)
    db2 = (1/m)*np.sum(dz2,axis=1,keepdims=True)
    dz1 = np.multiply(np.dot(W2.T,dz2),1-np.power(A1,2))
    dW1 = (1/m)*np.dot(dz1,X.T)
    db1 = (1/m)*np.sum(dz1,axis =1,keepdims=True)

    grads = {'dW1':dw1,
             'db1':db1,
             'dW2':dw2,
             'db2':db2}
    return grads

# 单元测试
parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
grads = back_propagation(parameters, cache, X_assess, Y_assess)
print ("dw1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dw2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))

# 用上面得到的梯度来进行梯度下降（更新参数w和b，使其更优化）
def update_parameters(parameters, grads, learning_rate=1.2):
    '''
    参数
    :param parameters: w和b
    :param grads: 梯度
    :param learning_rate:学习率
    :return: 参数
    '''
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    # 根据梯度和学习率来更新参数，使其更优
    W1 = W1 -learning_rate*dW1
    b1 = b1 -learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 -learning_rate*db2

    parameters = {'W1':W1,
                  'b1':b1,
                  'W2':W2,
                  'b2':b2}
    return parameters

# 单元测试
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# 上面已经将各个所需的功能函数都编写好了。现在我们将它们组合在一个大函数中来构建出一个训练模型。
def nn_model(X,Y,n_h,num_iterations = 10000, print_cost = False):
    '''

    :param X: 输入特征
    :param Y: 标签
    :param n_h: 隐藏层神经元个数
    :param num_iterations: 训练次数
    :param print_cost: 打印出成本
    :return:
    '''
    np.random.seed(3)
    n_x = X.shape[0] # 根据输入特征的维度得出输入层的神经元个数
    n_y = Y.shape[0] # 根据标签的维度得出输出层的神经元个数

    #初始化参数
    parameters = initialize_parameters(n_x,n_h,n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # 在这个循环里进行训练，一次一次地对参数进行优化
    for i in range(0,num_iterations):
        #前向传播
        A2, cache = forward_propagation(X,parameters)

        #计算出本次的成本
        cost = compute_cost(A2,Y,parameters)

        #进行反向传播
        grads = back_propagation(parameters,cache,X,Y)

        #根据梯度对参数进行一次优化
        parameters = update_parameters(parameters,grads)

        #将本次训练的成本打印出来
        if print_cost and i % 1000 ==0:
            print('在悬链%i次后，成本是：%f'%(i,cost))

    return parameters

# 单元测试
X_assess, Y_assess = nn_model_test_case()

parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# 我们已经可以通过上面的函数来进行参数训练。
# 这个函数可以利用上面学习到的参数来对新数据进行预测
def predict(parameters,X):
    '''

    :param parameters: 训练得出的参数
    :param X: 预测数据
    :return:
    '''
    #预测其实是执行一次前向传播
    A2, cache = forward_propagation(X,parameters)
    predictions = np.round(A2)#对结果进行四舍五入,小于0.5就是0，否则是1
    return predictions

# 单元测试
parameters, X_assess = predict_test_case()

predictions = predict(parameters, X_assess)
print("predictions mean = " + str(np.mean(predictions)))

# 首先是根据训练数据来进行参数学习（训练数据是与单神经元网络一样一样的）
parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)

# 然后用训练得出的参数进行预测。
predictions = predict(parameters, X)
print ('预测准确率是: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

# 将预测结果画出来。
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y.ravel())





