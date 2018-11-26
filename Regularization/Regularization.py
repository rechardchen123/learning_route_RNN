'''
前面的文章已经说了，如果神经网络过拟合了，就会出现神经网络对训练数据集的预测效果很好，但是到实际使用时，效果就很差。
为了解决过拟合问题，添加数据量是肯定有效的，但是数据太难获取也太贵，所以首选方案就是正则化了。
所以说，正则化是很重要的。本次实战编程就带领大家对正则化进行进一步地直观认识！
'''
import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid,relu,plot_decision_boundary,initialize_parameters,load_2D_dataset,predict_dec
from reg_utils import compute_cost,predict,forward_propagation,backward_propagation,update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_2D_dataset()

'''
1 - 不加正则化的模型
下面就是我们将使用的神经网络模型函数。这个函数会根据输入的参数不同而选择是否为模型添加正则化。

如果要添加 L2正则化 -- 那么就将参数lambd设置为非0的值。
如果要使用 dropout -- 那么就设置keep_prob为小于1的数。
这个函数里面的大部分工具函数已经在reg_utils.py里面实现好了，通过前面的实战编程，我们已经很熟悉它们了，无需再浪费篇幅展示它们。

里面的compute_cost_with_regularization()和backward_propagation_with_regularization()这个两个函数是用于实现L2正则化的；
forward_propagation_with_dropout()和backward_propagation_with_dropout()是用于实现dropout的。这4个函数的实现是我们本次实战编程的重点。
'''

def model(X,Y,learning_rate=0.3,num_iterations = 30000,print_cost=True,lambd=0,keep_prob=1):
    grads = {}
    costs = []
    m =X.shape[1]
    layers_dims = [X.shape[0],20,3,1]

    parameters = initialize_parameters(layers_dims)

    for i in range(0,num_iterations):
        if keep_prob ==1:
            a3,cache = forward_propagation(X,parameters)
        elif keep_prob<1:
            a3, cache = forward_propagation_with_dropout(X,parameters,keep_prob)

        if lambd ==0:
            cost = compute_cost(a3,Y)
        else:
            cost = compute_cost_with_regularization(a3,Y,parameters,lambd)

        assert(lambd==0 or keep_prob==1)

        if lambd==0 and keep_prob==1:
            grads = backward_propagation(X,Y,cache)
        elif lambd!=0:
            grads = backward_propagation_with_regularization(X,Y,cache,lambd)
        elif keep_prob<1:
            grads = backward_propagation_with_dropout(X,Y,cache,keep_prob)

        parameters = update_parameters(parameters,grads,learning_rate)

        if print_cost and i %10000==0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000==0:
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

#首先，我们对不加正则化的模型进行训练。然后观察训练好的模型对训练数据集和测试数据集的预测精准度。
parameters = model(train_X,train_Y)
print('On the training set:')
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())

'''
2 - L2正则化
L2正则化是解决过拟合的常用方法之一.
'''
def compute_cost_with_regularization(A3,Y,parameters,lambd):
    m = Y.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    #获得常规的成本
    cross_entropy_cost = compute_cost(A3,Y)
    #计算L2的尾巴
    L2_regularization_cost = lambd *(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))/(2*m)
    cost = cross_entropy_cost + L2_regularization_cost

    return cost

#单元测试
A3,Y_assess,parameters = compute_cost_with_regularization_test_case()
print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd = 0.1)))

#第二步，在反向传播计算偏导数时在dW后加上L2尾巴lambda/m W.
def backward_propagation_with_regularization(X,Y,cache,lambd):
    m = X.shape[1]
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3) = cache

    dZ3 = A3 -Y

    dW3 = 1./m*np.dot(dZ3,A2.T)+(lambd *W3)/m
    db3 = 1./m*np.sum(dZ3,axis =1,keepdims=True)

    dA2 = np.dot(W3.T,dZ3)
    dZ2 = np.multiply(dA2,np.int64(A2>0))
    dW2 = 1./m*np.dot(dZ2,A1.T)+(lambd*W2)/m
    db2 = 1./m*np.sum(dZ2,axis=1,keepdims=True)

    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(dA1,np.int64(A1>0))
    dW1 = 1./m*np.dot(dZ1,X.T)+(lambd*W1)/m
    db1 = 1./m*np.sum(dZ1,axis=1,keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
    return gradients

X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()

grads = backward_propagation_with_regularization(X_assess,Y_assess,cache,lambd=0.7)
print ("dW1 = " + str(grads["dW1"]))
print ("dW2 = " + str(grads["dW2"]))
print ("dW3 = " + str(grads["dW3"]))

'''
下面我们通过设置模型函数model()的参数 λλ 来构建添加了L2正则化的神经网络模型.(λ=0.7)model() 函数里面会上面我们实现的L2函数:
用compute_cost_with_regularization取代compute_cost
用backward_propagation_with_regularization取代backward_propagation
'''
parameters = model(train_X,train_Y,lambd=0.7)
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())

'''
3 - Dropout
dropout也是一个被深度学习领域经常用到的解决过拟合的方法。
'''

#dropout的forward_propagation
def forward_propagation_with_dropout(X,parameters,keep_prob=0.5):
    np.random.seed(1)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1,X)+b1
    A1 = relu(Z1)

    D1 = np.random.rand(A1.shape[0],A1.shape[1]) #dropout第一步，创建一个与A1相同的矩阵D1
    D1 = D1 < keep_prob                          #第二步,设置阈值
    A1 = A1 *D1                                  #第三步,重置A1
    A1 = A1 / keep_prob                          #第四步，反向失活

    Z2 = np.dot(W2,A1)+b2
    A2 = relu(Z2)

    D2 = np.random.rand(A2.shape[0],A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 *D2
    A2 = A2/keep_prob

    Z3 = np.dot(W3,A2)+b3
    A3 = sigmoid(Z3)

    cache =(Z1,D1,A1,W1,b1,Z2,D2,A2,W2,b2,Z3,A3,W3,b3)

    return A3,cache

X_assess, parameters = forward_propagation_with_dropout_test_case()

A3,cache = forward_propagation_with_dropout(X_assess,parameters,keep_prob=0.7)

#drop_out的backward_propagation
def backward_propagation_with_dropout(X,Y,cache,keep_prob):
    m = X.shape[1]
    (Z1,D1,A1,W1,b1,Z2,D2,A2,W2,b2,Z3,A3,W3,b3) = cache

    dZ3 = A3 -Y
    dW3 = 1./m*np.dot(dZ3,A2.T)
    db3 = 1./m*np.sum(dZ3,axis=1,keepdims=True)
    dA2 = np.dot(W3.T,dZ3)

    dA2 = dA2*D2
    dA2 = dA2/keep_prob

    dZ2 = np.multiply(dA2,np.int64(A2>0))
    dW2 = 1./m*np.dot(dZ2,A1.T)
    db2 = 1./m*np.sum(dZ2,axis=1,keepdims=True)

    dA1 = np.dot(W2.T,dZ2)

    dA1 = dA1*D1
    dA1 = dA1/keep_prob

    dZ1 = np.multiply(dA1,np.int64(A1>0))
    dW1 = 1./m*np.dot(dZ1,X.T)
    db1 = 1./m*np.sum(dZ1,axis=1,keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()


gradients = backward_propagation_with_dropout(X_assess, Y_assess, cache, keep_prob=0.8)

print ("dA1 = " + str(gradients["dA1"]))
print ("dA2 = " + str(gradients["dA2"]))

parameters = model(train_X, train_Y, keep_prob=0.86, learning_rate=0.3)

print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y.ravel())






















