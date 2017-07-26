#!/usr/bin/env python
# encoding: utf-8
# /**
#  * @Author:      south
#  * @DateTime:    2017-07-24 17:29:04
#  * @Description: 利用感知器实现and
#  */
from functools import reduce

class Perceptron(object):
    """docstring for Perceptron"""
    def __init__(self, input_num, activator):
        '''
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为 double -> double
        '''
        #super(Perceptron, self).__initinput_num, activator
        self.arg = input_num, activator
        self.activator = activator
	    # 权重向量初始化为0
        self.weights = [0.0 for _ in range(input_num)]
	    # 偏置项初始化为0
        self.bias = 0.0

    def __str__(self):
        '''
        打印学习到的权重、偏置项
        '''
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)
        

    def predict(self, input_vec):
        '''
            输入向量，输出感知器计算结果
        '''
        # 把 input_vec[x1,x2,...] 和 weights[w1,w2,...] 打包在一起
        # 变成 [(x1,w1),(x2,w2),...]
        # 然后利用 map 函数计算[x1*w1, x2*w2, ...]
        # 最后利用 reduce 求和
        # return self.activator(
        #     reduce(lambda a, b: a + b,
        #            map(lambda (x, w): x * w,  
        #                zip(input_vec, self.weights))
        #         , 0.0) + self.bias)
        return self.activator(
            reduce(lambda a, b: a + b,
                    map(lambda x_w: x_w[0] * x_w[1],
                        zip(input_vec, self.weights))
                , 0.0) + self.bias)
        
    def train(self, input_vecs, labels, iteration, rate):
        """
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        """
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)
        
    def _one_iteration(self, input_vecs, labels, rate):
        """docstring for _one_iteration
        一次迭代，把所有的训练数据过一遍
        """
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label),...]
        # 而每个训练样本是 (input_vec, label)
        samples = zip(input_vecs, labels)
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)
        
    def _update_weights(self, input_vec, output, label, rate):
        """docstring for _update_weights
            按照感知器规则更新权重
        """
        # 把 input_vec[x1,x2,...] 和 weights[w1, w2, ...]打包在一起
        # 变成 [(x1,w1),(x2,w2),...]
        # 然后利用感知器规则更新权重
        delta = label - output
        # self.weights = map(
        #         lambda (x, w): w + rate * delta *x,
        #         zip(input_vec, self.weights))
        self.weights = list(map(
                lambda x_w: x_w[1] + rate * delta *x_w[0],
                zip(input_vec,self.weights)))
        # 更新bias
        self.bias += rate * delta
        

def f(x):
    """docstring for f
    定义激活函数f"""
    return 1 if x > 0 else 0
    

def get_training_dataset():
    """docstring for get_training_dataset
    基于 and 真值表构建训练数据"""
    # 构建训练数据
    # 输入向量列表
    input_vecs = [[1,1], [0,0], [1,0], [0,1]]
    # 期望的输出列表，注意一定要一一对应
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    labels = [1, 0, 0, 0]
    return input_vecs, labels
    

def train_and_perceotron():
    """docstring for train_and_perceotron
    使用 and 真值表训练感知器"""
    # 创建感知器，输入参数个数为2（因为 and 是二元函数），激活函数为f
    p = Perceptron(2, f)
    # 训练，迭代10轮，学习速率0.1
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    # 返回训练好的感知器
    return p
    

if __name__ == '__main__':
    # 训练 and 感知器
    and_perceptron = train_and_perceotron()
    # 打印训练获得的权重
    print (and_perceptron)
    # 测试
    print ('1 and 1 = %d' % and_perceptron.predict([1, 1])) 
    print ('0 and 0 = %d' % and_perceptron.predict([0, 0]))
    print ('1 and 0 = %d' % and_perceptron.predict([1, 0]))
    print ('0 and 1 = %d' % and_perceptron.predict([0, 1]))
