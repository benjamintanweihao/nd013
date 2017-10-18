from functools import reduce
import numpy as np


class Node(object):
    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []
        self.value = None

        # Keys are the inputs to this node and their values
        # are the partials of this node with respect to that
        # input
        self.gradients = {}

        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

    def forward(self):
        """
        Forward propagation.

        Compute the output value based on `inbound_nodes` and
        store the result in self.value.
        """
        raise NotImplemented

    def backward(self):
        raise NotImplemented


class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes
        # so no need to pass anything to the Node instantiator
        Node.__init__(self)

    # NOTE: Input node is the only node where the value
    # may be passed as an argument to forward()
    #
    # All other implementations should get the value
    # of the previous node from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in
        if value is not None:
            self.value = value

    def backward(self):
        # An Input node has no inputs so the gradient (derivative)
        # is zero
        self.gradients = {self: 0}
        # Weights and biases may be inputs, so you need to sum the
        # gradients from output gradients
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] += grad_cost * 1


class Add(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = sum([x.value for x in self.inbound_nodes])


class Mul(Node):
    def __init__(self, *inputs):
        Node.__init__(self, inputs)

    def forward(self):
        self.value = reduce(lambda x, y: x * y,
                            [x.value for x in self.inbound_nodes])


class Linear(Node):
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def forward(self):
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value

        self.value = np.dot(X, W) + b

    def backward(self):
        """
        Calculates the gradients based on the output values
        """

        # Initialize a partial for each of the in bound nodes
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Cycle through the outputs. The gradient will change depending on each
        # output. so the gradients are summed over all outputs
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this node's inputs
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            # Set the partial of the loss with respect to this node's weights
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this node's bias
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def forward(self):
        self.value = self._sigmoid(self.inbound_nodes[0].value)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self.inbound_nodes[0]] = grad_cost * (self.value * (1 - self.value))


class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):
        """
        Calculates the mean squared error

        Note: We reshape these to avoid possible matrix/vector
        broadcast errors.
        """
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        self.m = self.inbound_nodes[0].value.shape[0]
        self.diff = y - a
        self.value = np.mean(np.square(self.diff))

    def backward(self):
        """
        Calculates the gradient of the cost

        This is the final node of the network so outbound nodes
        are not a concern
        """
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = -(2 / self.m) * self.diff


def forward_and_backward(graph):
    """
    Performs a forward pass and backward pass through a list of sorted nodes

    Arguments:

        `graph`: The result of calling `topological_sort`
    """

    for n in graph:
        n.forward()

    for n in graph[::-1]:
        n.backward()


def topological_sort(feed_dict):
    """
    Sort generic nodes in topological order using Kahn's Algorithm.

    `feed_dict`: A dictionary where the key is a `Input` node and the value is the respective value feed to that node.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L
