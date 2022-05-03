from queue import Queue
import unittest


def breath_first_search(graph: dict[str, set[str]],
                        starting_node: str) -> list[str]:
    """
    Search through graph by prioritizing elements we see first

    Have a graph you would like to search
    Have a starting point to search form
    Have a function which decides when to stop searching
    Have a function for calculating other spots to search given current position
    Have a data structure to store positions

    :param graph: hashmap of string to hashset of strings
    :param starting_node: node from which to begin search
    :return: a list of nodes in the order they were searched
    :raises: Exception if starting node not in graph
    """

    if starting_node not in graph:
        raise Exception(f'{starting_node} not in graph')

    plan_to_visit: Queue[str] = Queue()
    visited: set[str] = set()
    trace: list[str] = []

    plan_to_visit.put(starting_node)

    while not plan_to_visit.empty():
        visit_now = plan_to_visit.get()
        if visit_now in visited:
            continue

        visited.add(visit_now)
        trace.append(visit_now)

        connections = (c for c in graph.get(visit_now, set())
                       if c not in visited)

        for c in connections:
            plan_to_visit.put(c)

    return trace


class TestBFS(unittest.TestCase):
    def test_empty_graph(self):
        with self.assertRaises(Exception):
            breath_first_search({}, "Z")

    def test_sample_graph(self):
        self.assertCountEqual(
            ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
            breath_first_search(
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
