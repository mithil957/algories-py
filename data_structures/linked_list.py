import unittest
from dataclasses import dataclass


class Node:
    def __init__(self, value, next_node):
        self.value = value
        self.next_node = next_node


class LinkedList:
    def __init__(self, value):
        self.root = Node(value, None)

    def is_empty(self):
        return self.root is None

    def append_left(self, value):
        if self.is_empty():
            self.root = Node(value, None)
        else:
            new_root = Node(value, self.root)
            self.root = new_root

    def append_right(self, value):
        if self.is_empty():
            self.root = Node(value, None)
        else:
            current = self.root
            while current.next_node is not None:
                current = current.next_node

            current.next_node = Node(value, None)

    def pop_left(self):
        if self.is_empty():
            return
        else:
            new_root = self.root.next_node
            del self.root
            self.root = new_root

    def pop_right(self):
        if self.is_empty():
            return
        else:
            current = self.root
            while current.next_node.next_node is not None:
                current = current.next_node

            del current.next_node
            current.next_node = None

    def to_list(self):
        if self.is_empty():
            return []
        else:
            res = []
            current = self.root
            while current is not None:
                res.append(current.value)
                current = current.next_node

            return res


class TestLinkedList(unittest.TestCase):
    def test_case_1(self):
        ll = LinkedList(0)

        for i in range(1, 5):
            ll.append_right(i)

        self.assertEqual([0, 1, 2, 3, 4], ll.to_list())

        for _ in range(1, 5):
            ll.pop_right()

        self.assertEqual([0], ll.to_list())


