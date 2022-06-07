import unittest
from collections import deque

"""
    Reference: HackerRank - Find the nearest clone
"""


class BFSIterator:
    """
        Helper class which helps with stepping through BFS from node one step at a time
    """

    def __init__(self, color_map, node):
        self.distance = 0
        self.exhausted = False
        self.color_of_interest = color_map[node]
        self.visited = set()
        self.to_visit = deque()
        self.to_visit.append(node)

    def step(self, graph, node_distances):
        number_of_nodes_in_q = len(self.to_visit)
        if number_of_nodes_in_q == 0:
            self.exhausted = True

        for _ in range(number_of_nodes_in_q):
            current = self.to_visit.popleft()
            self.visited.add(current)

            if current in node_distances:
                return self.distance + node_distances[current]
            else:
                node_distances[current] = self.distance

            for neighbor in graph[current]:
                if neighbor not in self.visited:
                    self.to_visit.append(neighbor)

        self.distance += 1

        return None


def find_shortest(graph_nodes, graph_from, graph_to, ids, val):
    """

    :param graph_nodes: number of nodes in graph
    :param graph_from: represents connection FROM a node
    :param graph_to: represents connection TO a node
    :param ids: ids[i] is the color of node i (1-indexed)
    :param val: the color value we are looking for
    :return: the shortest distance between nodes of same val, if not found then -1

    Create graph of type [int, set[int]]
    Create color map which maps node number to its color
    Use color map to find all nodes with color equal to val
    Create "BFS searchers" from those points by calling the BFSIteration class
    Create node_distances map which will hold the distance to a node that a searcher has traveled
        - each searcher will discover a node then record the distance from node to the search's origin

    Loop until searchers run out of nodes to search:
        Step each searcher
        if searcher comes across a node in node_distances
            (meaning it has been discovered by other searchers)
            then return the searcher's distance plus
                the distance the other searcher traveled (lookup in node_distances)

    if searchers cannot search anymore then return -1
    """

    # Create graph
    graph = {}
    for origin, destination in zip(graph_from, graph_to):
        if origin in graph:
            graph[origin].add(destination)
        else:
            graph[origin] = {destination}

        if destination in graph:
            graph[destination].add(origin)
        else:
            graph[destination] = {origin}

    # create hashtable where (key -> value) is mapping between node number and node color
    color_map = {node_number: color for node_number, color in enumerate(ids, 1)}

    bfs_iterators = []
    for node_number, color in color_map.items():
        if color == val:
            bfs_iterators.append(BFSIterator(color_map, node_number))

    node_distances = {}
    all_exhausted = False

    while not all_exhausted:
        for bfs_searcher in bfs_iterators:
            search_results = bfs_searcher.step(graph, node_distances)
            if search_results is not None:
                return search_results

        all_exhausted = all(bfs_searcher.exhausted for bfs_searcher in bfs_iterators)

    return -1


class NearestCloneTest(unittest.TestCase):
    def test_case_1(self):
        result = find_shortest(5, [1, 1, 2, 3], [2, 3, 4, 5], [1, 2, 3, 3, 2], 2)
        self.assertEqual(result, 3)

    def test_case_2(self):
        result = find_shortest(7, [1, 1, 2, 3, 5, 6], [2, 3, 4, 5, 6, 7], [1, 2, 3, 3, 2, 8, 2], 2)
        self.assertEqual(result, 2)

    def test_case_3(self):
        result = find_shortest(7,
                               [1, 1, 2, 3, 5, 6, 2],
                               [2, 3, 4, 5, 6, 7, 7],
                               [1, 2, 3, 3, 2, 8, 2], 2)
        self.assertEqual(result, 1)

    def test_case_4(self):
        result = find_shortest(10,
                               [1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9],
                               [2, 3, 4, 7, 10, 5, 6, 7, 8, 10, 9, 10],
                               [2, 1, 1, 1, 2, 1, 1, 1, 2, 1], 2)
        self.assertEqual(result, 3)

    def test_case_5(self):
        result = find_shortest(9,
                               [1, 4, 2, 5, 3, 6],
                               [4, 7, 5, 8, 6, 9],
                               [2, 2, 2, 1, 1, 1, 1, 1, 1], 2)
        self.assertEqual(result, -1)
