from abc import ABC, abstractmethod


class Solver(ABC):

    @abstractmethod
    def solve(self, board: list[list[int]], visualize: bool):
        pass
