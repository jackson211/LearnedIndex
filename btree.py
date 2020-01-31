import pandas as pd
import sys
import datetime
import argparse
import random


class Item():

    def __init__(self, key, pos):
        self.k = key
        self.p = pos

    def __str__(self):
        return "[k:" + str(self.k) + " p:" + str(self.p) + ']'

    def __eq__(self, other):
        return self.k == other.k

    def __gt__(self, other):
        return self.k > other.k

    def __ge__(self, other):
        return self.k >= other.k

    def __lt__(self, other):
        return self.k < other.k

    def __le__(self, other):
        return self.k <= other.k


class BTreeNode(object):
    __slots__ = ["isLeaf", "keys", "children", "size", "_index"]

    def __init__(self, t, isLeaf=False):
        self.isLeaf = isLeaf
        self.keys = []
        self.children = []
        self.size = t

    @property
    def is_full(self):
        return len(self.keys) == 2 * self.size - 1

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, i):
        self._index = i


class BTree(object):
    __slots__ = ["root", "t", "nodes", "index_count", "key_count"]

    def __init__(self, t):
        self.root = BTreeNode(t, isLeaf=True)
        self.root.index = 0
        self.t = t
        self.nodes = {}
        self.index_count = 0
        self.key_count = 0

    def alloc_node(self, isLeaf=False):
        new_node = BTreeNode(self.t, isLeaf=isLeaf)
        self.index_count += 1
        new_node.index = self.index_count
        self.nodes[self.index_count] = new_node
        return new_node

    def search(self, k, node=None):
        if node is None:
            node = self.root
        if not isinstance(k, Item):
            k = Item(k, 0)
        i = 0
        while i < len(node.keys) and k > node.keys[i]:
            i += 1
        if i < len(node.keys) and k == node.keys[i]:  # found key
            return node.keys[i].p
        elif node.isLeaf:  # If the node is leaf, there are no more key to look for
            return None
        else:
            return self.search(k, node.children[i])

    def insert(self, k):
        search_result = self.search(k)
        if search_result is not None:
            return None
        r = self.root
        if r.is_full:
            s = self.alloc_node()
            self.root = s
            s.children.insert(0, r)
            self._split_child(s, 0)
            self._insert_nonfull(s, k)
        else:
            self._insert_nonfull(r, k)

    def _insert_nonfull(self, x, k):
        i = len(x.keys) - 1
        if x.isLeaf:
            # insert a key
            x.keys.append(Item(-1, -1))
            while i >= 0 and k < x.keys[i]:
                x.keys[i + 1] = x.keys[i]
                i -= 1
            x.keys[i + 1] = k
            self.key_count += 1
        else:
            # insert a child
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            if x.children[i].is_full:
                self._split_child(x, i)
                if k > x.keys[i]:
                    i += 1
            self._insert_nonfull(x.children[i], k)

    def _split_child(self, p_node, i):
        mid = self.t - 1
        r_node = p_node.children[i]
        l_node = self.alloc_node(isLeaf=r_node.isLeaf)

        # insert left node to parent node at i + 1
        # insert mid key of right node to its parent node
        p_node.children.insert(i + 1, l_node)
        p_node.keys.insert(i, r_node.keys[mid])

        # keys after middle key store in l node
        # keys before middle key store in r node
        l_node.keys = r_node.keys[(mid + 1):]
        r_node.keys = r_node.keys[0:(mid - 1)]

        if not r_node.isLeaf:
            l_node.children = r_node.children[(mid + 1):]
            r_node.children = r_node.children[0:mid]

    def __str__(self):
        this_level = [self.root]
        s = ""
        i = 0
        while this_level:
            next_level = []
            output = ""
            for node in this_level:
                if node is None:
                    output += " - "
                    continue
                if node.children:
                    next_level.extend(node.children)
                    next_level.append(None)

                output += '(' + ' '.join([str(k) for k in node.keys]) + ')'
                # output += str(node.keys) + " "
            s += "Level {}: ".format(i) + output + '\n'
            this_level = next_level
            i += 1

        return s


def main(path, t):
    # Build BTree
    print("Loading data file from: {}".format(str(path)))
    data = pd.read_csv(path, header=None)
    total_data_size = data.shape[0]
    test_data_size = int(0.1 * total_data_size)

    print("Total data size: ", total_data_size)
    print("Test data size: ", test_data_size)

    btree = BTree(t)

    start = datetime.datetime.now()
    for index, row in data.iterrows():
        btree.insert(Item(row[0], row[1]))
    end = datetime.datetime.now()
    # print(btree)
    print("BTree degree: {}".format(t))
    print("BTree total keys: {}".format(btree.key_count))
    print("BTree build time: " + str(end - start) + "\n")

    # Search Key
    print("Search for keys")
    test_data = data.sample(n=test_data_size)
    search_result = {}
    search_start = datetime.datetime.now()
    for index, row in test_data.iterrows():
        search_key = row[0]
        search_result[search_key] = btree.search(search_key)
    search_end = datetime.datetime.now()
    print(list(k for k, v in search_result.items() if v is None))
    # print(data.loc[data[0] == 699])

    print("Keys' search time: " + str(search_end - search_start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build Btree')
    parser.add_argument('-p',
                        '--path',
                        type=str,
                        required=True,
                        help='data path')
    parser.add_argument('-t',
                        '--degree',
                        type=int,
                        required=True,
                        dest='degree',
                        help='degree of BTree')

    args = parser.parse_args()

    path = args.path
    t = args.degree
    main(path, t)