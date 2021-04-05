# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


class Item():
    '''
    Each Node in B-Tree contains an Item.
    '''
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __repr__(self):
        return "[key:{}, val: {}]".format(self.key, self.value)

    def __eq__(self, other):
        return self.key == other.key

    def __gt__(self, other):
        return self.key > other.key

    def __ge__(self, other):
        return self.key >= other.key

    def __lt__(self, other):
        return self.key < other.key

    def __le__(self, other):
        return self.key <= other.key

    @property
    def val(self):
        return self.value
