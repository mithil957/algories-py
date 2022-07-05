import unittest
from collections import defaultdict


def walk_forward(pair, pairs, path):
    if len(pairs) == 0:
        return path

    end = pair[1]
    for index, edge in enumerate(pairs):
        if edge[0] == end:
            result = walk_forward(edge, pairs[:index] + pairs[index + 1:], path + [edge])
            if result is not None:
                return result

    return walk_backward(path[0], pairs, path)


def walk_backward(pair, pairs, path):
    if len(pairs) == 0:
        return path

    start = pair[0]
    for index, edge in enumerate(pairs):
        if edge[1] == start:
            result = walk_backward(edge, pairs[:index] + pairs[index + 1:], [edge] + path)
            if result is not None:
                return result

    return None


def valid_arrangement(pairs):
    return walk_forward(pairs[0], pairs[1:], [pairs[0]])


def valid_arrangement_euler_path(pairs):
    """
        Copied from https://leetcode.com/problems/valid-arrangement-of-pairs/discuss/1616288/Python-O(V%2BE)-by-Euler-path-w-Visualization
        b/c their comments made it very easy to understand
    """
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    adj_matrix = defaultdict(list)

    for src, dst in pairs:
        in_degree[dst] += 1
        out_degree[src] += 1
        adj_matrix[src].append(dst)

    # Case 1 -> Euler circuit in graph, so any node can be starting value
    # Euler circuit starts at and ends at the SAME vertex, and uses every edge once
    start_node_idx = pairs[0][0]

    # Case 2 -> Not a circuit but an Euler path. So, we find
    # Euler path starts at and ends at a DIFFERENT vertex, and uses every edge once
    for node in adj_matrix:

        # find node whose out-degree is one more than in-degree
        if out_degree[node] - in_degree[node] == 1:
            start_node_idx = node
            break

    def euler_path(adj_matrix, path, curr_node):

        # DFS until all edges of current node are visited
        while adj_matrix[curr_node]:
            # pop one edge and get next visit node
            next_visit_node = adj_matrix[curr_node].pop()
            euler_path(adj_matrix, path, next_visit_node)

            # post-order style
            # current explorer is finished, record current edge pair
            path.append([curr_node, next_visit_node])

    record = []
    euler_path(adj_matrix, record, start_node_idx)

    # reversed of post-order is the euler path
    return list(reversed(record))


class ValidArrangementTest(unittest.TestCase):
    def test_case_1(self):
        result = valid_arrangement([[5, 1], [4, 5], [11, 9], [9, 4]])
        self.assertEqual([[11, 9], [9, 4], [4, 5], [5, 1]], result)

        result = valid_arrangement_euler_path([[5, 1], [4, 5], [11, 9], [9, 4]])
        self.assertEqual([[11, 9], [9, 4], [4, 5], [5, 1]], result)

    def test_case_2(self):
        result = valid_arrangement([[1, 3], [3, 2], [2, 1]])
        self.assertEqual([[1, 3], [3, 2], [2, 1]], result)

    def test_case_3(self):
        result = valid_arrangement([[8, 5], [5, 0], [0, 9], [0, 7], [7, 0]])
        self.assertEqual([[8, 5], [5, 0], [0, 7], [7, 0], [0, 9]], result)
