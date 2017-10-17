from miniflow import *
import numpy as np

# x, y, z = Input(), Input(), Input()
#
# add = Add(Mul(x, y), z)
#
# feed_dict = {x: 5, y: 4, z: 3}
#
# sorted_nodes = topological_sort(feed_dict=feed_dict)
# output = forward_pass(add, sorted_nodes)
#
# print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x],
#                                                          feed_dict[y],
#                                                          feed_dict[z],
#                                                          output))


# inputs, weights, bias = Input(), Input(), Input()
#
# f = Linear(inputs, weights, bias)
#
# feed_dict = {
#     inputs: [6, 14, 3],
#     weights: [0.5, 0.25, 1.4],
#     bias: 2
# }
#
# graph = topological_sort(feed_dict)
# output = forward_pass(f, graph)
#
# print(output) # should be 12.7 with this example

# X, W, b = Input(), Input(), Input()
#
# f = Linear(X, W, b)
#
# X_ = np.array([[-1., -2.], [-1, -2]])
# W_ = np.array([[2., -3], [2., -3]])
# b_ = np.array([-3., -5])
#
# feed_dict = {X: X_, W: W_, b: b_}
#
# graph = topological_sort(feed_dict)
# output = forward_pass(f, graph)
#
# """
# Output should be:
# [[-9., 4.],
# [-9., 4.]]
# """


X, W, b = Input(), Input(), Input()

f = Linear(X, W, b)
g = Sigmoid(f)

X_ = np.array([[-1., -2.], [-1, -2]])
W_ = np.array([[2., -3], [2., -3]])
b_ = np.array([-3., -5])

feed_dict = {X: X_, W: W_, b: b_}

graph = topological_sort(feed_dict)
output = forward_pass(g, graph)

"""
Output should be:
[[  1.23394576e-04   9.82013790e-01]
 [  1.23394576e-04   9.82013790e-01]]
"""
print(output)

