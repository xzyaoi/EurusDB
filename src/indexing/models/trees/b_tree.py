# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from timeit import default_timer as timer
from typing import List, Tuple

import numpy as np

import src.indexing.utilities.metrics as metrics
from src.indexing.models import BaseModel
from src.indexing.models.trees.item import Item


class BTreeNode():
    def __init__(self, degree) -> None:
        self.degree = degree
        self.items: List[Item] = [None] * (2 * degree - 1)
        self.children: List[BTreeNode] = [None] * (2 * degree)
        self.num_of_keys = 0
        self.is_leaf = True
        self.index = None

    def is_full(self) -> bool:
        return self.num_of_keys == 2 * self.degree - 1


class BTree():
    def __init__(self, degree) -> None:
        self.degree = degree
        self.root = BTreeNode(self.degree)
        self.root.index = 1
        self.next = self.root.index + 1
        self.nodes = {}
        self.nodes[self.root.index] = self.root

    def allocate_node(self):
        new_node = BTreeNode(self.degree)
        new_node.index = self.next
        self.nodes[self.next] = new_node
        self.next = self.next + 1
        return new_node

    def get_node_at(self, index) -> BTreeNode:
        return self.nodes[index]

    def search(self, key) -> Tuple[bool, int, Item]:
        '''
        Search tries to find the key in the current node, 
        Parameters
        -----------
            key: [any] the requested key value
        Returns:
            has_found: [boolean] if the key is found in this node
            position:  [int] the smallest index i in the sorted array such that
                             key <= self.items[i]
            val: [Item] the value found in the items
            * if there's no such key equals to the look-up key 
                reurn False, smallest higher key and None
        '''
        current = self.root
        while True:
            i = 0
            while i < current.num_of_keys and key > current.items[i].key:
                i += 1
            if i < current.num_of_keys and key == current.items[i].key:
                # found the result, stop and break the loop
                return True, current.index, current.items[i]
            if current.is_leaf:
                # reached leaf node, stop and break the loop
                if i > 0:
                    return False, current.index, current.items[i - 1]
                else:
                    return False, current.index, current.items[0]
            else:
                # now move to children[i]
                current = self.get_node_at(current.children[i])

    def split_child(self, parent: BTreeNode, index: int, child: BTreeNode):
        new_node = self.allocate_node()
        new_node.is_leaf = child.is_leaf
        new_node.num_of_keys = self.degree - 1
        for j in range(self.degree - 1):
            new_node.items[j] = child.items[j + self.degree]
        if not child.is_leaf:
            for j in range(self.degree):
                new_node.children[j] = child.children[j + self.degree]

        child.num_of_keys = self.degree - 1
        j = parent.num_of_keys + 1
        while j > index + 1:
            parent.children[j + 1] = parent.children[j]
            j -= 1
        parent.children[j] = new_node.index
        j = parent.num_of_keys
        while j > index:
            parent.items[j + 1] = parent.items[j]
            j -= 1
        parent.items[index] = child.items[self.degree - 1]
        parent.num_of_keys += 1

    def insert(self, item: Item):
        has_found, position, val = self.search(item.key)
        if has_found:
            return None
        current = self.root
        if current.is_full():
            # root is full, create a new node as the new root
            new_root = self.allocate_node()
            self.root = new_root
            self.root.index = new_root.index
            # now the new root cannot be leaf
            new_root.is_leaf = False
            # now the new root does not contain any keys
            new_root.num_of_keys = 0
            new_root.children[0] = current.index
            self.split_child(new_root, 0, current)
            self.insert_nonfull(new_root, item)
        else:
            # root is not full, simply insert
            self.insert_nonfull(current, item)
        return item

    def insert_nonfull(self, target_node: BTreeNode, item: Item):
        # i is the current number of items
        i = target_node.num_of_keys - 1
        if target_node.is_leaf:
            while i >= 0 and item < target_node.items[i]:
                # rearrange items that are larger than the given item
                # to make an empty space to locate the new item
                target_node.items[i + 1] = target_node.items[i]
                i -= 1
            target_node.items[i + 1] = item
            target_node.num_of_keys += 1
        else:
            # if it is not the leaf
            # find the median first
            # i the median
            while i >= 0 and item < target_node.items[i]:
                i -= 1
            i += 1
            if self.get_node_at(target_node.children[i]).is_full():
                # then we need to split into two children
                self.split_child(target_node, i,
                                 self.get_node_at(target_node.children[i]))
                if item > target_node.items[i]:
                    i += 1
            self.insert_nonfull(self.get_node_at(target_node.children[i]),
                                item)


class BTreeModel(BaseModel):
    def __init__(self, page_size, degree=0):
        super().__init__("B-Tree (d={})".format(degree), page_size)
        self.btree = BTree(degree)

    def train(self, x_train, y_train, x_test, y_test):
        self.total_data_size = x_train.shape[0]
        x, y = (list(t) for t in zip(*sorted(zip(x_train, y_train))))
        start_time = timer()
        for i in range(self.total_data_size):
            self.btree.insert(Item(x[i], y[i]))
        end_time = timer()
        test_data_size = x_test.shape[0]
        pred_y = []
        for i in range(test_data_size):
            pred_y.append(
                self.btree.search(x_test[i])[2].value // self.page_size)
        pred_y = np.array(pred_y)
        mse = metrics.mean_squared_error(y_test, pred_y)
        return mse, end_time - start_time

    def fit(self, x_train, y_train):
        data_size = len(x_train)
        for i in range(data_size):
            self.btree.insert(Item(x_train[i], y_train[i]))

    def __str__(self):
        return self.btree.__str__()

    def predict(self, x):
        has_found, position, item = self.btree.search(int(x))
        if not has_found:
            return item.value
        return item.value
