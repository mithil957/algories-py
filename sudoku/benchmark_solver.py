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

        with open('all_17_clue_sudokus.txt') as f:
            num_puzzles = f.readline().replace('\n', '')
            puzzle_counter = 0
            can_read = True
            while can_read:
                puzzle_counter += 1
                if random.random() >= .15:  # run on 15% of cases
                    continue

                # TODO break loop check
                current_puzzle = f.readline().replace('\n', '')
                board = [[0 for _ in range(9)] for _ in range(9)]

                for cell_number, val in enumerate(current_puzzle):
                    board[cell_number // 9][cell_number % 9] = int(val)

                # solving board
                start_time = time.time()
                self.solver.solve(board)
                print(f'Puzzle {puzzle_counter} solved in {(time.time() - start_time) * 1000} milliseconds')
                times_taken.append(time.time() - start_time)

        print('---------------------')
        print(f'Total time taken: {sum(times_taken)} seconds')
        print(f'Avg time taken: {sum(times_taken) / len(times_taken)} seconds')


if __name__ == '__main__':
    wfc_solver = WFCSolver()
    wfc_solver.visualize = False
    benchmark = Benchmark(wfc_solver)

    benchmark.benchmark_against_17_clue()
