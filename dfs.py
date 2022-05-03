from collections import deque
import unittest


def depth_first_search(graph: dict[str, set[str]],
                       starting_node: str):
    """
    Search through the graph by prioritizing what we see last
    :param graph: hashmap of str to hashset of str
    :param starting_node: where to begin search from
    :return: list of nodes in the order they were searched
    :raises: Exception if starting_node not in graph
    """

    if starting_node not in graph:
        raise Exception(f'{starting_node} not in graph')

    plan_to_visit: deque[str] = deque()
    visited: set[str] = set()
    trace: list[str] = []

    plan_to_visit.append(starting_node)

    while plan_to_visit:
        visit_now = plan_to_visit.pop()
        if visit_now in visited:
            continue

        visited.add(visit_now)
        trace.append(visit_now)

        connections = (c for c in graph.get(visit_now, set())
                       if c not in visited)

        for c in connections:
            plan_to_visit.append(c)

    return trace


class TestDFS(unittest.TestCase):
    def test_empty_case(self):
        with self.assertRaises(Exception):
            depth_first_search({}, "Z")

    def test_sample_graph(self):
        self.assertCountEqual(
            ['A', 'B', 'D', 'E', 'F', 'G', 'C'],
            depth_first_search(
                {
                    "A": {"B", "C", "D"},
                    "B": {"A", "D", "E"},
                    "C": {"A", "F"},
                    "D": {"B", "D"},
                    "E": {"B", "F"},
                    "F": {"C", "E", "G"},
                    "G": {"F"},
                }, "A"
            ))


if __name__ == '__main__':
    unittest.main()
