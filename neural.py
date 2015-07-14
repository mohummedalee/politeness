__author__ = 'raza'
from Node import Node
from Tree import Tree
import numpy as np
from Model import Model


dim = 5
w = np.random.rand(dim, 2*dim)
ws = np.random.rand(2, dim)

delta_w = np.zeros(w.shape)
delta_ws = np.zeros(ws.shape)
reg_cost = 0.001
l_rate = 0.1

predictions = None
target = np.array([1, 0])
target = target[:, np.newaxis]


tree = Tree()
tree.root = Node()
node = Node(np.random.rand(dim, 1))
tree.root.add_child(node)
node = Node()
tree.root.add_child(node)
node1 = Node(np.random.rand(dim, 1))
node.add_child(node1)
node1 = Node(np.random.rand(dim, 1))
node.add_child(node1)


tree1 = Tree()
tree1.root = Node()
node = Node(np.random.rand(dim, 1))
tree1.root.add_child(node)
node = Node()
tree1.root.add_child(node)
node1 = Node(np.random.rand(dim, 1))
node.add_child(node1)
node1 = Node(np.random.rand(dim, 1))
node.add_child(node1)

RNN = Model(dim)
RNN.trees.append(tree)
RNN.trees.append(tree1)
RNN.train()
