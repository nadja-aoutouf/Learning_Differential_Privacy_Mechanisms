import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from keras.models import model_from_json


class InterfaceDecisionTree(DecisionTreeRegressor):
    def __init__(self, hypercube_dim, num_attribute):
        self.hypercube_dim = hypercube_dim
        self.num_attribute = num_attribute

    def number_of_nodes(self):
        if self.hypercube_dim == 0:
            number_of_nodes = 0
        else:
            num_of_nodes = 1
            for i in range(self.hypercube_dim):
                for j in range(self.num_attribute):
                    num_of_nodes *= len(thresholds[i][j])
        return num_of_nodes

    def number_of_dimensions(self):  # minimal number of subtrees
        return self.hypercube_dim

    def nodes_are_neighbours(self, node_id_a, node_id_b):
        assert node_id_a != node_id_b  # cubes must no overlap
        threshold_a = nodes[node_id_a]["threshold"]
        threshold_b = nodes[node_id_b]["threshold"]
        equal_thresholds = np.zeros(iface.hypercube_dim)

        for i in range(iface.hypercube_dim):
            equal_thresholds[i] = threshold_a[i] == threshold_b[i]

        non_equal_dimensions = np.arange(iface.hypercube_dim)[
            np.logical_not(equal_thresholds)
        ]

        return list(non_equal_dimensions) if len(non_equal_dimensions) == 1 else []

    # Approach for get_node_id()
    # 1. given the data_set, find the thresholds which describe the
    # corresponding node
    # 2. given the thresholds, find the corresponding node

    def get_node_id(self, data_set):
        node_of_data_set = [
            [1 for _ in range(iface.num_attribute)].copy()
            for _ in range(iface.hypercube_dim)
        ]

        for dim in range(self.hypercube_dim):
            for attribute in range(self.num_attribute):
                i = 0
                while (
                    thresholds[dim][attribute][i] <= data_set[dim][attribute]
                    and i <= len(thresholds[dim][attribute]) - 1
                ):
                    i += 1

                node_of_data_set[dim][attribute] = thresholds[dim][attribute][i]

        for id, node in enumerate(nodes):
            if node_of_data_set == node["threshold"]:
                node_id = id

        return node_id

    def evaluate_f(self, data_set):
        # this is something like a circular wave originating from 0, with minimum
        # -1 and maximum 1,
        # and multiple up and downs, characterised by SPEED
        SPEED = 4
        return np.sin(np.pi * np.linalg.norm(data_set, ord=2) * SPEED)

    def generate_data(self, length, seed):
        # helper function
        np.random.seed(seed)
        return np.random.uniform(
            low=0, high=1, size=(length, self.hypercube_dim, self.num_attribute)
        )

    # note dim = 3 is dimension 4. In general dim = n-1 represents the tree for
    # dimension n
    def build_tree(self, data_X):
        X = self.bring_data_in_valid_form(data_X)
        y = [self.evaluate_f(data_set) for data_set in data_X]

        clf = DecisionTreeRegressor(random_state=0)
        # max_leaf_nodes = 2 ** self.num_attribute, random_state=0)
        fitted_tree = clf.fit(X, y)
        return fitted_tree

    def bring_data_in_valid_form(self, data_X):
        X = np.array([data_set for data_set in data_X])
        nsample, n_dim, n_attribute = X.shape
        X = X.reshape((nsample, n_dim * n_attribute))
        return X


iface = InterfaceDecisionTree(hypercube_dim=3, num_attribute=2)
data = iface.generate_data(length=10, seed=10)

# creating the tree
decisiontree = iface.build_tree(data)


# return the set of leaves of a given binary tree
def classified_node_ids(decisiontree):
    set_of_leaves = []
    splitting_nodes = []
    path_to_leaf = [0]

    n_nodes_in_tree = decisiontree.tree_.node_count
    children_left = decisiontree.tree_.children_left
    children_right = decisiontree.tree_.children_right
    node_depth = np.zeros(shape=n_nodes_in_tree, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes_in_tree, dtype=bool)

    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)

    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]

        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    for i in range(n_nodes_in_tree):
        if is_leaves[i]:
            set_of_leaves.append(i)
        else:
            splitting_nodes.append(i)

    # check if all leaves are contained in set_of_leaves
    # assert len(set_of_leaves) == tree.get_n_leaves()
    return set_of_leaves, splitting_nodes


# assign all splitting nodes to one dimension
# IMPORTANT to find all neighbour nodes

leave_nodes, splitting_nodes = classified_node_ids(decisiontree)

dim_of_splitting_node = []
attribute_of_splitting_node = []
threshold_of_splitting_node = []

# note: dimension n is stored as n-1 ind dim_of_splitting_node
# (same for attribute_of_splitting_node)

for id in splitting_nodes:
    # find cooresponding threshold which is relevant for the splitting node
    threshold = decisiontree.tree_.threshold[id]
    threshold_of_splitting_node.append(threshold)

    # find corresponding dimension which is relevant for the splitting node
    dim = int(decisiontree.tree_.feature[id] / iface.num_attribute)
    dim_of_splitting_node.append(dim)

    # find corresponding attribute which is relevant for the splitting node
    attribute = (
        decisiontree.tree_.feature[id] - ((dim + 1) * iface.num_attribute)
    ) % iface.num_attribute
    attribute_of_splitting_node.append(attribute)

# find all thresholds in each dimension for each attribute
# assume that each attribute takes a value between [0,1]
# thresholds is a matrix with column = #attributes and rows = #dimensions

thresholds = [
    [None for _ in range(iface.num_attribute)].copy()
    for _ in range(iface.hypercube_dim)
]

for dim in range(iface.hypercube_dim):
    for attribute in range(iface.num_attribute):
        threshold_in_dim_and_attribute = [1]

        for i, node_id in enumerate(splitting_nodes):
            if (
                dim == dim_of_splitting_node[i]
                and attribute == attribute_of_splitting_node[i]
            ):
                threshold_in_dim_and_attribute.append(threshold_of_splitting_node[i])

        thresholds[dim][attribute] = sorted(threshold_in_dim_and_attribute)


# Given a leaf id i, get_path(i) returns path (from root to leaf i)
# Output is a list with tree_node_ids of increasing depth (starts with root and
# ends with leaf node i)
# if k is the id of an internal node, get_path(k) will return None


def allpaths(start_node):
    # a class method which finds all the tree paths from the root to the leaf
    if start_node == None:
        return []
    elif start_node in leave_nodes:  # in set of leaf nodes
        return [start_node]

    else:
        return [
            str(start_node) + "-->" + str(l)
            for l in allpaths(decisiontree.tree_.children_left[start_node])
            + allpaths(decisiontree.tree_.children_right[start_node])
        ]


def get_path(k):
    """tree method to get the desired path from root to leaf"""
    l = allpaths(0)
    for x in l:
        my_list = [int(y) for y in x.split("-->")]
        if my_list[-1] == k:
            return my_list
        else:
            pass


# returns if the direction in which the given path continues after each
# splitting node
# True : left child - splitting threshold is respected
# False: right child - splitting threshold is exceeded
# note: this function works only for binary trees


def splitting_node_direction(path):
    assert path != None  # input path is not valid
    decisions = []
    for i, id in enumerate(path[: len(path) - 1]):
        if decisiontree.tree_.children_left[id] == path[i + 1]:
            decisions.append(True)
        else:
            decisions.append(False)
    return decisions
