# -*- coding: utf-8 -*-
# @Author: south
# @Date:   2017-08-31 10:16:28
# @Last Modified by:   south
# @Last Modified time: 2017-08-31 11:24:09
from perceptron import Perceptron

# define activation function f

f = lambda x: x

class linear_unit(Perceptron):
	"""docstring for linear_unit"""
	def __init__(self, input_num):
		Perceptron.__init__(self, input_num, f)
	

def get_training_dataset():
	# creative five persons' data
	# 构建训练数据
    # 输入向量列表，每一项是工作年限
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    # 期望的输出列表，月薪，注意要与输入一一对应
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels 

		
def train_linear_unit():
    '''
    使用数据训练线性单元
    '''
    # 创建感知器，输入参数的特征数为1（工作年限）
    lu = linear_unit(1)
    # 训练，迭代10轮, 学习速率为0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    #返回训练好的线性单元
    return lu

def plot(linear_unit):
    import matplotlib.pyplot as plt
    input_vecs, labels = get_training_dataset()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(list(map(lambda x: x[0], input_vecs)), labels)
    weights = linear_unit.weights
    bias = linear_unit.bias
    x = range(0,12,1)
    y = list(map(lambda x:weights[0] * x + bias, x))
    ax.plot(x, y)
    plt.show()

if __name__ == '__main__': 
    '''训练线性单元'''
    LinearUnit = train_linear_unit()
    # 打印训练获得的权重
    print (LinearUnit)
    # 测试
    print ('Work 3.4 years, monthly salary = %.2f' % LinearUnit.predict([3.4]))
    print ('Work 15 years, monthly salary = %.2f' % LinearUnit.predict([15]))
    print ('Work 1.5 years, monthly salary = %.2f' % LinearUnit.predict([1.5]))
    print ('Work 6.3 years, monthly salary = %.2f' % LinearUnit.predict([6.3]))
    plot(LinearUnit)

