import numpy as np


def find_hierarchy_path(node, hierarchy):
    path = []
    while node is not None:
        path.append(node)
        node = hierarchy.get(node)
    path.reverse()  # Reverse to get path from root to the node
    return path


def convert_labels_to_hierarchy(y_train, y_test, hierarchy):
    y_hier_train = np.array([find_hierarchy_path(label, hierarchy) for label in y_train], dtype=object)
    y_hier_test = np.array([find_hierarchy_path(label, hierarchy) for label in y_test], dtype=object)
    return y_hier_train, y_hier_test

def convert_label_to_hierarchy(y_train, hierarchy):
    y_hier_train = np.array([find_hierarchy_path(label, hierarchy) for label in y_train], dtype=object)
    return y_hier_train

def convert_hierarchy_to_label(y_hier_train):
    while y_hier_train[-1] == ',' or y_hier_train[-1] == ' ' or y_hier_train[-1] == '' or y_hier_train[-1] == "":
        y_hier_train = y_hier_train[:-1]
    y_train = y_hier_train[-1]
    return y_train