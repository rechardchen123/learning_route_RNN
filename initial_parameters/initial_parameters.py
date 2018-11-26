'''
训练一个神经网络模型就是要找到一组特殊的参数。这个模型配上这组参数后，就拥有了某项能力，例如可以识别猫。
所以，找参数才是训练的最终目的。所以，参数的初始化非常重要。
如果参数被初始化得离理想参数很远很远，那么就需要很长很长的时间来进行梯度下降才能到达理想参数。
打比方说，w的理想值是2，而你将w初始化为1000，每次梯度下降又只能使w靠近理想值1个单位，那么要进行998次梯度下降才能找到理想参数；
如果将w初始化为10，那么就只需要8次。更坏的情况是，如果参数初始化得不合理，那么有可能会导致无论怎么样训练都无法找到理想值，
你的模型永远不可能被训练成功.本次实战编程向大家展示了3种不同的初始化方法，只有初始化不同而已，其它都是一样的，
但结果却有3种：一种是无法找到理想值，另外一种是很长时间才能找到理想值，最后一种却很快就找到理想值了。
'''
#加载系统工具
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

#加载自定义工具
from init_utils import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 加载我们用算法生成的假数据并把它们画出来（只画了训练数据，没有画测试数据）。
# 我们的目的就是训练一个模型，使其能够将红点和蓝点区分开。
train_X, train_Y, test_X, test_Y = load_dataset()

# 构建一个模型，实现细节很多都在我们自定义的工具库init_utils.py里面。因为那些细节我们之前已经学过，
# 所以为了突出重点，就把它们隐藏在工具库里面了。
# 这个模型的特点是，它可以指定3种不同的初始化方法，通过参数initialization来控制
def model(X,Y,learning_rate=0.01,num_iterations=15000,print_cost=True,initialization='he'):

    grads={}
    costs = []
    m =X.shape[1]
    layers_dims = [X.shape[0],10,5,1] #构建一个三层的神经网络

    #3中初始化的方法
    if initialization =='zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization=='random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialization=='he':
        parameters = initialize_parameters_he(layers_dims)

    #梯度下降训练参数
    for i in range(0,num_iterations):
        a3,cache = forward_propagation(X,parameters)
        cost = compute_loss(a3,Y)
        grads = backward_propagation(X,Y,cache)
        parameters = update_parameters(parameters,grads,learning_rate)

        if print_cost and i % 1000 == 0:
            print('cost after iteration {}:{}'.format(i,cost))
            costs.append(cost)

    #画出成本函数
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iteration (per hundreds)')
    plt.title('Learning_rate = '+str(learning_rate))
    plt.show()

    return parameters

# 第一种方法:也是我们学习过的第一种方法——全部初始化为0(参数最差的方法)
def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)

    for l in range(1,L):
        parameters['W'+str(l)] = np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters['b'+str(l)] = np.zeros((layers_dims[l],1))

    return parameters

# 单元测试
parameters = initialize_parameters_zeros([3,2,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

#用全0的初始化方法进行训练
parameters = model(train_X,train_Y,initialization='zeros')
print('On the train set: ')
predictions_train = predict(train_X,train_Y,parameters)
print('On the test set: ')
predictions_test = predict(test_X,test_Y,parameters)

'''
从上面的图表我们可以看出，成本完全没有下降，说明根本就一点都没有优化到。0.5的精确度就像赌单双一样，完全没有预测的能力。
'''

print('prediction_train = '+str(predictions_train))
print('prediction_test = '+str(predictions_test))
#如果初始化参数为0，那么神经网络的每一层都只会学习到同样的东西，也就是说，一万层的神经网络和一层的单神经网络一样一样了。

'''
为了使每一层每一个神经元都能学到不同的东西，我们需要将参数进行随机初始化。下面这个方法就是对神经网络进行参数随机初始化。
'''
def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    parameters={}
    L = len(layers_dims)

    for l in range(1,L):
        parameters['W'+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])*10
        parameters['b'+str(l)]= np.zeros((layers_dims[l],1))

    return parameters

parameters = initialize_parameters_random([3,2,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

parameters = model(train_X, train_Y, initialization = "random")
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

print(predictions_train)
print(predictions_test)

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x:predict_dec(parameters,x.T),train_X,train_Y)

'''
从上面的成本图可以看出，成本开始时特别大。这是因为我们将参数初始化成了很大的值，
这就会导致神经网络在前期对预测太绝对了，不是0就是1，如果预测错了，就会导致成本很大。
参数初始化得不对会导致训练效率很差，需要训练很长时间才能靠近理想值。
下面的代码中，你可以将训练次数改大一些，你会看到，训练得越久，成本会越来越小，预测精准度越来越高。
参数初始化得不对，还会导致梯度消失和爆炸。
'''

def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters={}
    L = len(layers_dims)-1

    for l in range(1,L+1):
        parameters['W'+str(l)]= np.random.randn(layers_dims[l],layers_dims[l-1])*np.sqrt(2/layers_dims[l-1])
        parameters['b'+str(l)]=np.zeros((layers_dims[l],1))

    return parameters

parameters = initialize_parameters_he([2,4,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

parameters = model(train_X,train_Y,initialization='he')
print('On the train set: ')
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)














