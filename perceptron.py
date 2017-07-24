# /**
#  * @Author:      south
#  * @DateTime:    2017-07-24 17:29:04
#  * @Description: 利用感知器实现and
#  */

class Perceptron(object):
	"""docstring for Perceptron"""
	def __init__(self, input_num, activator):
		'''
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为 double -> double
        '''
		super(Perceptron, self).__initinput_num, activator)
		self.arg = ainput_num, activator

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
		pass

	def predict(self, input_vec):
		'''
		输入向量，输出感知器计算结果
		'''
		# 把 input_vec[x1,x2,...] 和 weights[w1,w2,...] 打包在一起
		# 变成 [(x1,w1),(x2,w2),...]
		# 然后利用 map 函数计算[x1*w1, x2*w2, ...]
		# 最后利用 reduce 求和
                return self.activator(
                        reduce(lambda a,b:a+b,
                                 map(lambda(x,w):x*w,
                                     zip(input_vec,self.weights))
                            ,  0.0) + self.bias )
		pass
        def train(self, input_vecs, labels, iteration, rate):
            """
            输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
            """
            for i in range(iteration):
                self._one_iteration(input_vecs, labels, rate)
            pass
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
            pass
        def _update_weights(self, input_vec, output, label, rate):
            """docstring for _update_weights
            按照感知器规则更新权重
            """
            # 把 input_vec[x1,x2,...] 和 weights[w1, w2, ...]打包在一起
            # 变成 [(x1,w1),(x2,w2),...]
            # 然后利用感知器规则更新权重
            delta = label - output
            self.weights = map(
                    lambda (x,w): w + rate * delta *x,
                    zip(input_vec, selg.weights))
            # 更新bias
            self.bias += rate * delta
            pass
