import math
import unittest
from copy import deepcopy
from typing import TypeAlias, Callable
from queue import PriorityQueue

GCost: TypeAlias = int
HCost: TypeAlias = int
FCost: TypeAlias = int


class Node:
    def __init__(self, x: int, y: int, value: str):
        self.x = x
        self.y = y
        self.value = value
        self.g_cost = None
        self.h_cost = None
        self.f_cost = None

    def set_cost(self, g_cost: GCost, h_cost: HCost, f_cost: FCost) -> None:
        self.g_cost: GCost = g_cost
        self.h_cost: HCost = h_cost
        self.f_cost: FCost = f_cost

    def __key(self):
        return self.x, self.y

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other_node):
        return self.__key() == other_node.__key()

    def __lt__(self, other_node):
        if self.g_cost is not None and other_node.g_cost is not None:
            return (self.g_cost < other_node.g_cost) or (
                    self.g_cost == other_node.g_cost and self.h_cost < other_node.h_cost)
        else:
            raise Exception("one or both node do not have cost calculated")


Grid: TypeAlias = list[list[Node]]
TraceGrid: TypeAlias = list[list[Node]]


def euclidean_cost(
        starting: Node,
        ending: Node,
        candidate: Node
) -> tuple[GCost, HCost, FCost]:
    diff_x: int = abs(starting.x - candidate.x)
    diff_y: int = abs(starting.y - candidate.y)
    g_cost: GCost = math.ceil((diff_x ** 2 + diff_y ** 2) ** .5)

    diff_x: int = abs(ending.x - candidate.x)
    diff_y: int = abs(ending.y - candidate.y)
    h_cost: HCost = math.ceil((diff_x ** 2 + diff_y ** 2) ** .5)

    f_cost: FCost = g_cost + h_cost
    return g_cost, h_cost, f_cost


def a_star_search(
        grid: Grid,
        start_node: Node,
        end_node: Node,
        cost_func: Callable[[Node, Node, Node], tuple[GCost, HCost, FCost]]
) -> TraceGrid:
    """
    Thinking beyond a 2d map and instead as a graph where movement is limited by
    what edges a node has

    A-star is the same as BFS and DFS in that
        - can be used to search a graph
        - has a starting point
        - has an ending point
        - to get to end
            - pull a move from data structure
            - make a move if that move has not been made before
            - from our current move, look at all possible moves
            - add those moves to data structure
        - keep going until you hit end or run out of moves

    A-star differs in that
        - moves in the data structure are ordered using a function/heuristic
        - in this case the heuristic we wanted to minimize was the sum
            of (distance from current position to new position) +
            (distance from new position to end position)

    :param grid: a 2d map where O represents passable terrain
        and X represents impassable terrain
    :param start_node: where to start the searching form
    :param end_node: where to end the search
    :param cost_func: used to calculate best move
    :return: 2d map with path taken by a_star
    """
    trace_grid: TraceGrid = deepcopy(grid)
    plan_to_visit: PriorityQueue[Node] = PriorityQueue()
    visited: set[Node] = set()

    x_dim: int = len(grid[0])
    y_dim: int = len(grid)

    start_node.set_cost(*cost_func(start_node, end_node, start_node))
    plan_to_visit.put(start_node)
    while not plan_to_visit.empty():
        # finds best move by finding the lowest cost move
        # since moves are kept in order using PriorityQueue, we just get the first element
        lowest_cost_node = plan_to_visit.get()

        # mark node in visited, so we don't visit it again
        visited.add(lowest_cost_node)

        # if the node we are on is end node, we made it
        if lowest_cost_node == end_node:
            return trace_grid

        # this is just for making the trace look pretty, can be ignored
        if lowest_cost_node != start_node:
            trace_grid[lowest_cost_node.y][lowest_cost_node.x].value = "*"

        """
        find neighbors, since its 2d grid, we look in 8 directions
        for neighbor, we filter if ... 
            it's out of bounds
            impassable
            already visited
        if neighbor passes filter -> calculate cost and put in queue
        """
        for delta_x in [-1, 0, 1]:
            for delta_y in [-1, 0, 1]:
                if delta_y == 0 and delta_x == 0:
                    continue

                neighbor_x = lowest_cost_node.x + delta_x
                neighbor_y = lowest_cost_node.y + delta_y
                out_of_bounds = neighbor_x >= x_dim or neighbor_x < 0 or neighbor_y >= y_dim or neighbor_y < 0
                if out_of_bounds:
                    continue

                neighbor_node = grid[neighbor_y][neighbor_x]
                if neighbor_node.value != "X" and neighbor_node not in visited:
                    neighbor_node.set_cost(*(cost_func(lowest_cost_node, end_node, neighbor_node)))
                    plan_to_visit.put(neighbor_node)

    return trace_grid


def node_grid_as_str(node_grid: list[list[Node]]) -> str:
    return '\n'.join(['\t'.join([i.value for i in j]) for j in node_grid])


class TestAStar(unittest.TestCase):
    def test_sample_grid(self):
        base_grid = [
            ['O', 'O', 'O', 'O', 'O', 'O'],
            ['O', 'O', 'X', 'O', 'O', 'O'],
            ['O', 'O', 'X', 'O', 'O', 'O'],
            ['O', 'O', 'X', 'A', 'O', 'O'],
            ['O', 'O', 'X', 'O', 'X', 'O'],
            ['O', 'O', 'O', 'X', 'X', 'O'],
            ['B', 'O', 'O', 'O', 'O', 'O']
        ]

        base_grid[3][3] = 'A'
        base_grid[6][0] = 'B'

        node_grid = []
        for j in range(len(base_grid)):
            temp = []
            for i in range(len(base_grid[0])):
                temp.append(Node(i, j, base_grid[j][i]))
            node_grid.append(temp)

        self.assertEqual(len(base_grid), len(node_grid))
        self.assertEqual(len(base_grid[0]), len(node_grid[0]))

        trace = a_star_search(node_grid,
                              node_grid[3][3],
                              node_grid[6][0],
                              euclidean_cost)

        print(node_grid_as_str(trace))
