from collections import namedtuple
from collections import deque
from dataclasses import dataclass

import os
import time

from sudoku_solver import Solver


def clear() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')


"""
    Sudoku Solver 
    - brute force approach
    - board is a 2d array of size N by N 
    - start at the top left cell
        - calculate possible values for that cell using Sudoku rules
        - if there is at least one value we can place in cell
            - *pick a value 
            - update the board
            - push our "choice" represented as Move object pushed onto the stack
            - *move to the next cell (left to right, top to bottom)
        - if there is no possible value for the cell
            - loop
            - pop from stack, we get our previous "choice"
            - when choice was made, were there any other values we could have picked?
            - Move object has other_possible_values property
            - if other_possible_values is empty (no other choice could have been made)
                - loop again
            - if other_possible_values is not empty (another choice could have been made)
                - *pick a value from other_possible_values
                - update the board
                - push "choice" onto the pack with updated other_possible_values
                - *move to the next cell
        
    
    * pick a value
        - possible values are represented as a set
        - picks the first value in set
        - Is there a better way of picking a value from the possibilities?
         
    * move to next cell
        - from our current cell, we always move to left to right and top to bottom
        - Is there a better way of deciding which cell to go to next?
"""

Cell = namedtuple('Cell', ['row', 'col'])


@dataclass
class Move:
    cell: Cell
    value: int
    other_possible_values: set[int]


BoardSolved = namedtuple('BoardSolved', [])
Backtrack = namedtuple('Backtrack', [])
SkipCell = namedtuple('SkipCell', [])


class BruteSolver(Solver):
    def __init__(self):
        self.board: list[list[int]] | None = None
        self.n = 0
        self.cell_size = 0
        self.moves: deque[Move] | None = None
        self.current_cell: Cell | None = None
        self.possibilities: set[int] | None = None

        # for visualizing
        self.visualize = False
        self.steps = 0

    def set_board(self, board: list[list[int]]):
        self.board = board
        self.n = len(board)
        self.cell_size = int(self.n ** .5)
        self.moves: deque[Move] = deque()
        self.current_cell = Cell(0, 0)
        self.possibilities: set[int] = set()

    def solve(self, board: list[list[int]]):
        self.set_board(board)

        while True:
            match self.next_action():
                case BoardSolved():
                    break

                case SkipCell():
                    self.move_forward()

                case Cell(_, _):
                    self.make_move_on_board()
                    self.move_forward()
                    self.possibilities = set()

                case Backtrack():
                    self.backtrack()

    def next_action(self) -> BoardSolved | SkipCell | Cell | Backtrack:
        if self.current_cell == Cell(self.n, 0):
            return BoardSolved()

        if self.board[self.current_cell.row][self.current_cell.col] != 0:
            return SkipCell()

        if len(self.possibilities) != 0:
            return self.current_cell

        self.possibilities = self.calculate_possible_values(self.current_cell)
        if len(self.possibilities) != 0:
            return self.current_cell

        else:
            return Backtrack()

    def move_forward(self) -> None:
        if self.current_cell.col + 1 < self.n:
            self.current_cell = Cell(self.current_cell.row, self.current_cell.col + 1)
        else:
            self.current_cell = Cell(self.current_cell.row + 1, 0)

        self.visualize_process()

    def calculate_possible_values(self, cell: Cell) -> set[int]:
        row_values = {i for i in self.board[cell.row]}
        col_values = {self.board[row][cell.col] for row in range(len(self.board))}

        cell_row = (cell.row // self.cell_size) * self.cell_size
        cell_col = (cell.col // self.cell_size) * self.cell_size

        cell_values = {
            self.board[cell_row + dx][cell_col + dy]
            for dy in range(0, self.cell_size)
            for dx in range(0, self.cell_size)
        }

        all_values = {i for i in range(1, self.n + 1)}

        return all_values - row_values - col_values - cell_values

    def make_move_on_board(self):
        current_cell = self.current_cell
        move = Move(current_cell, self.possibilities.pop(), self.possibilities)
        self.moves.append(move)
        self.board[current_cell.row][current_cell.col] = move.value

        self.visualize_process()

    def backtrack(self):
        cannot_make_move = True

        while cannot_make_move:
            last_move = self.moves.pop()
            self.current_cell = last_move.cell
            self.board[last_move.cell.row][last_move.cell.col] = 0

            if len(last_move.other_possible_values) > 0:
                cannot_make_move = False
                self.possibilities = last_move.other_possible_values
            else:
                self.possibilities = set()

            self.visualize_process()

    def visualize_process(self):
        if not self.visualize:
            return

        clear()
        self.steps += 1
        print(f'Steps: {self.steps}', end='\n\n')
        print(self.board_repr(self.current_cell))
        time.sleep(1 / 60)

    # why is pretty printing a board so HARD
    def board_repr(self, cell: Cell):
        s: list[list[int | str]] = [[self.board[row_ind][col_ind]
                                     for col_ind in range(self.n)]
                                    for row_ind in range(self.n)]

        for row_ind in range(self.n):
            for col_ind in range(self.n):
                val = s[row_ind][col_ind]
                p1 = ' ' * (self.cell_size + 2)
                p2 = ' ' * (self.n - self.cell_size - 2)
                s[row_ind][col_ind] = p1 + str(val) + p2

        bs = []

        cell_number = (cell.row * self.n) + cell.col
        spacing = len(str(self.n)) * len(str(self.cell_size))
        for row_ind, row in enumerate(s):
            str_row = ['|' for _ in range(self.cell_size)]

            for col_ind, val_str in enumerate(row):
                for ind, i in enumerate(range(0, self.n, self.cell_size)):
                    if self.board[row_ind][col_ind] != 0:
                        str_row[ind] += f'\033[1;37;42m{val_str[i: i + self.cell_size]:{spacing}}\033[0;0m'
                    elif row_ind * self.n + col_ind == cell_number:
                        str_row[ind] += f'\033[1;37;46m{val_str[i: i + self.cell_size]:{spacing}}\033[0;0m'
                    else:
                        str_row[ind] += f'{val_str[i: i + self.cell_size]:{spacing}}'

                    str_row[ind] += '|'

                if col_ind % self.cell_size == self.cell_size - 1:
                    for i in range(len(str_row)):
                        # size 4 gap
                        str_row[i] += '    '

            if row_ind % self.cell_size == 0:
                bs.append(' ' * len(str_row[0]))

            bs.extend(str_row)
            bs.append('-' * ((spacing * self.n) + (self.n + 1) + (4 * (self.cell_size - 1))))

        return '\n'.join(bs)

    def __repr__(self):
        return self.board_repr(self.current_cell)


if __name__ == '__main__':
    # 9 by 9
    example_board_1 = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, 0, 3, 4, 5, 6, 7],
        [0, 3, 4, 5, 0, 6, 1, 8, 2],

        [0, 0, 1, 0, 5, 8, 2, 0, 6],
        [0, 0, 8, 6, 0, 0, 0, 0, 1],
        [0, 2, 0, 0, 0, 7, 0, 5, 0],

        [0, 0, 3, 7, 0, 5, 0, 2, 8],
        [0, 8, 0, 0, 6, 0, 7, 0, 0],
        [2, 0, 7, 0, 8, 3, 6, 1, 5],
    ]

    solver = BruteSolver()
    solver.visualize = True
    solver.solve(example_board_1)

    # ---------------------------------------------
    # 16 by 16

    example_board_2_str = """
    0 15 0 1 0 2 10 14 12 0 0 0 0 0 0 0
    0 6 3 16 12 0 8 4 14 15 1 0 2 0 0 0
    14 0 9 7 11 3 15 0 0 0 0 0 0 0 0 0
    4 13 2 12 0 0 0 0 6 0 0 0 0 15 0 0
    0 0 0 0 14 1 11 7 3 5 10 0 0 8 0 12
    3 16 0 0 2 4 0 0 0 14 7 13 0 0 5 15
    11 0 5 0 0 0 0 0 0 9 4 0 0 6 0 0
    0 0 0 0 13 0 16 5 15 0 0 12 0 0 0 0
    0 0 0 0 9 0 1 12 0 8 3 10 11 0 15 0
    2 12 0 11 0 0 14 3 5 4 0 0 0 0 9 0
    6 3 0 4 0 0 13 0 0 11 9 1 0 12 16 2
    0 0 10 9 0 0 0 0 0 0 12 0 8 0 6 7
    12 8 0 0 16 0 0 10 0 13 0 0 0 5 0 0
    5 0 0 0 3 0 4 6 0 1 15 0 0 0 0 0
    0 9 1 6 0 14 0 11 0 0 2 0 0 0 10 8
    0 14 0 0 0 13 9 0 4 12 11 8 0 0 2 0
    """

    # some transformations to get turn it into 2d array of ints
    example_board_2 = [
        list(map(int, row.strip().split(' ')))
        for row in example_board_2_str.split('\n')[1:-1]
    ]

    solver.solve(example_board_2)
