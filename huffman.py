__author__ = 'Matias'

from collections import Counter
from heapq import heappop, heappush, heapify
from itertools import count


class Node(object):
    _ids = count(0)

    def __init__(self, freq, word=None, left=None, right=None):
        self.id = self._ids.next() if word is None else None
        self.word = word
        self.left = left
        self.right = right
        self.freq = freq

    def __cmp__(self, other):
        if self.freq > other.freq:
            return 1
        elif self.freq < other.freq:
            return -1
        else:
            return 0


class HuffmanEncoder(object):

    def __init__(self, word_counts):
        self.tree = self.build_huffman_tree(word_counts)
        self.encodings = self.huffman_codes(self.tree)

    def build_huffman_tree(self, counts):
        nodes = [Node(freq=b, word=a) for a, b in counts.iteritems()]
        heapify(nodes)
        while len(nodes) > 2:
            left = heappop(nodes)
            right = heappop(nodes)
            new_node = Node(freq=left.freq+right.freq, left=left, right=right)
            heappush(nodes, new_node)
        left = nodes[0]
        right = nodes[1]
        root = Node(freq=left.freq+right.freq, left=left, right=right)
        return root

    def huffman_codes(self, tree, encodings={}, prefix=""):
        # depth-first walk
        if tree.word is not None:
            encodings[tree.word] = prefix
        else:
            left_encodings = self.huffman_codes(tree.left, encodings, prefix=prefix+"1")
            right_encodings = self.huffman_codes(tree.right, encodings, prefix=prefix+"0")
            # merge left- and right-child encodings
            encodings = left_encodings.copy()
            encodings.update(right_encodings)
        return encodings

    def get_internal_node_ids(self, word):
        code = self.encodings[word]
        ids = []
        tree = self.tree
        for bit in code:
            ids.append(tree.id)
            tree = tree.left if bit == "1" else tree.right
        return ids

    def get_code(self, word):
        return self.encodings[word]


if __name__ == "__main__":
    word_counts = Counter([1,2,4,2,1,4,3,1,3,5,3,2,4,2,2,2,2,2,2,2,9])
    encoder = HuffmanEncoder(word_counts)
    print encoder.get_internal_node_ids(1)