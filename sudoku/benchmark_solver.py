import time

from sudoku_solver_wfc import WFCSolver
from sudoku_solver_brute import BruteSolver
from sudoku_solver import Solver
import random


class Benchmark:
    def __init__(self, solver: Solver):
        self.solver = solver

    def benchmark_against_17_clue(self):
        times_taken = []

        # all 9 by 9 puzzles
        with open('all_17_clue_sudokus.txt') as f:
            num_puzzles = int(f.readline().replace('\n', ''))

            for puzzle_count in range(num_puzzles):
                if random.random() <= .015:  # run on 15% of cases
                    current_puzzle = f.readline().replace('\n', '')
                    board = [[0 for _ in range(9)] for _ in range(9)]

                    for cell_number, val in enumerate(current_puzzle):
                        board[cell_number // 9][cell_number % 9] = int(val)

                    # solving board
                    start_time = time.time()
                    self.solver.solve(board, visualize=False)
                    print(f'Puzzle {puzzle_count} solved in {(time.time() - start_time) * 1000} milliseconds')
                    times_taken.append(time.time() - start_time)

        print('---------------------')
        print(f'Total time taken: {sum(times_taken)} seconds')
        print(f'Avg time taken: {sum(times_taken) / len(times_taken)} seconds')


if __name__ == '__main__':
    wfc_solver = WFCSolver()
    benchmark = Benchmark(wfc_solver)

    benchmark.benchmark_against_17_clue()
