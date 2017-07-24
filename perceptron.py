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
		pass