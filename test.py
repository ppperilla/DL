#!/usr/bin/env python
# encoding: utf-8
# /**
#  * @Author:      south
#  * @DateTime:    2017-07-25 13:51:39
#  * @Description: Description
#  */

import matplotlib.pyplot as plt
from numpy import *

class Connection(object):
	def __init__(self, upstream_node, downstream_node):
		'''
		初始化连接，权重初始化为是一个很小的随机数
		upstream_node: 连接的上游节点
		downstream_node: 连接的下游节点
		'''
		self.upstream_node = upstream_node
		self.downstream_node = downstream_node
		self.weight = random.uniform(-0.1, 0.1)
		self.gradient = 0.0 #梯度
	pass

from functools import reduce

def f():
	input_vec = [1,2,3]
	weights = [4,5,6]
	test = map(
				lambda x_w: x_w[1] + 0.1 * x_w[0],
				zip(input_vec,weights))
	#print(list(test))
	connections = [Connection(upstream_node, downstream_node) 
		for upstream_node in self.layers[layer].nodes
		for downstream_node in self.layers[layer + 1].nodes[:-1]]
	pass

def mat():
	fig = plt.figure()
	ax = fig.add_subplot(111)
	x = [1,2,3]
	y = [2,3,6]
	ax.plot(x,y)
	plt.show()
	pass
if __name__ == '__main__':
	mat()


