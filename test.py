#!/usr/bin/env python
# encoding: utf-8
# /**
#  * @Author:      south
#  * @DateTime:    2017-07-25 13:51:39
#  * @Description: Description
#  */

from functools import reduce

def f():
	input_vec = [1,2,3]
	weights = [4,5,6]
	
	test = reduce(lambda a, b: a + b,
				map(lambda x_w: x_w[1] + 0.1 * x_w[0],
					zip(input_vec,weights))
				, 0.0)
	print(test)
	pass
if __name__ == '__main__':
	f()


