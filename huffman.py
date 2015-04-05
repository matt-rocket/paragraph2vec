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


def build_huffman_tree(counts):
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


def huffman_codes(tree):
    pass


if __name__ == "__main__":
    word_counts = Counter([1,2,4,2,1,4,3,1,3,5,3,2,4,2,2,2,2,2,2,2,9])
    tree = build_huffman_tree(word_counts)