import datetime
import random


class BTreeNode(object):

    def __init__(self, t, isLeaf=False):
        self.isLeaf = isLeaf
        self.index = None
        self.keys = []
        self.children = []
        self.size = t

    @property
    def is_full(self):
        return len(self.keys) == 2 * self.size - 1

    @property
    def index(self):
        return self.__index

    @index.setter
    def index(self, i):
        assert (i == None) or isinstance(i, int)
        self.__index = i


class BTree(object):
    __slots__ = ["root", "t", "nodes", "index_count"]

    def __init__(self, t):
        self.root = BTreeNode(t, isLeaf=True)
        self.root.index = 0
        self.t = t
        self.nodes = {0: self.root}
        self.index_count = 0

    def alloc_node(self, isLeaf=False):
        new_node = BTreeNode(self.t, isLeaf=isLeaf)
        self.index_count += 1
        new_node.index = self.index_count
        self.nodes[self.index_count] = new_node
        return new_node

    def search(self, k, node=None):
        if node is None:
            node = self.root
        i = 0
        while i < len(node.keys) and k > node.keys[i]:
            i += 1
        if i < len(node.keys) and k == node.keys[i]:  # found key
            return (node.index, i)
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
            x.keys.append(-1)
            while i >= 0 and k < x.keys[i]:
                x.keys[i + 1] = x.keys[i]
                i -= 1
            x.keys[i + 1] = k
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
        t = self.t
        r_node = p_node.children[i]
        l_node = self.alloc_node(isLeaf=r_node.isLeaf)

        # slide all children of node to the right and insert z at i+1.
        p_node.children.insert(i + 1, l_node)
        p_node.keys.insert(i, r_node.keys[t - 1])

        # keys of z are t to 2t - 1,
        # y is then 0 to t-2
        l_node.keys = r_node.keys[t:]
        r_node.keys = r_node.keys[0:(t - 2)]

        # children of z are t to 2t els of y.children
        if not r_node.isLeaf:
            l_node.children = r_node.children[t:]
            r_node.children = r_node.children[0:(t - 1)]

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
                output += str(node.keys) + " "
            s += "Level {}: ".format(i) + output + '\n'
            this_level = next_level
            i += 1

        return s


# def binary_search(array, x):
#     lo, hi = 0, len(array)
#     while lo <= hi:
#         m = (lo + hi) // 2
#         if array[m] == x:
#             return m
#         elif x > array[m]:
#             lo = m + 1
#         else:
#             hi = m - 1
#     return None


def main():
    # Build BTree
    b = BTree(3)

    start = datetime.datetime.now()
    for i in range(100):
        key = random.randint(0, 100)
        print(key)
        b.insert(key)
        print(b)
    end = datetime.datetime.now()

    print("BTree build time: " + str(end - start))

    # Search Key
    search_key = 8

    search_start = datetime.datetime.now()
    result = b.search(search_key)
    search_end = datetime.datetime.now()

    if result is not None:
        ix, pos = result
        result_key = b.nodes[ix].keys[pos]
        print("Search Key : {0}, Result pos: {1}, Correct: {2}".format(
            search_key, result, (result_key == search_key)))
        print("Key search time: " + str(search_end - search_start))
    else:
        print("Key {} not found".format(search_key))


if __name__ == "__main__":
    main()