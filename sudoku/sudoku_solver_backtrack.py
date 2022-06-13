from collections import namedtuple
from collections import deque
import os
import time


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

Move = namedtuple('Move', ['row', 'col', 'value', 'other_possible_values'])


class SudokuSolver:
    def __init__(self, board: list[list[int]]):
        self.board = board
        self.n = len(board)
        self.cell_size = int(self.n ** .5)

    def solve(self):
        moves: deque[Move] = deque()
        row, col = 0, 0
        iterations = 0

        while True:
            # for visualizing algo., not needed for solving
            clear()
            iterations += 1
            print(f'Iterations: {iterations}')
            print()
            print(self.board_repr(row, col))
            time.sleep(1 / 240)

            # if at end, then we completed the board
            if row == 9 and col == 0:
                break

            # if there is value set, ignore it
            # for handling pre-filled values of a board
            if self.board[row][col] != 0:
                row, col = self.move_forward(row, col)
                continue

            # calculate what values can go in current cell
            remaining_values = self.calculate_remaining_values(row, col)

            # if we can make a move, make it
            if len(remaining_values) != 0:
                move = Move(row, col, remaining_values.pop(), remaining_values)
                moves.append(move)
                self.board[row][col] = move.value
                self.move_forward(row, col)

            # if we cannot make a move, backtrack
            else:
                cannot_make_move = True

                # keep popping from stack until we can make a move
                while cannot_make_move:
                    iterations += 1  # for visualizing algo

                    last_move = moves.pop()
                    self.board[last_move.row][last_move.col] = 0

                    # we can make move
                    if len(last_move.other_possible_values) > 0:
                        # break loop
                        cannot_make_move = False
                        new_move = Move(last_move.row, last_move.col,
                                        last_move.other_possible_values.pop(), last_move.other_possible_values)
                        moves.append(new_move)
                        self.board[new_move.row][new_move.col] = new_move.value
                        row, col = self.move_forward(new_move.row, new_move.col)

    def calculate_remaining_values(self, current_row: int, current_col: int) -> set[int]:
        row_values = {i for i in self.board[current_row]}
        col_values = {self.board[row][current_col] for row in range(len(self.board))}

        cell_row = (current_row // self.cell_size) * self.cell_size
        cell_col = (current_col // self.cell_size) * self.cell_size

        cell_values = {
            self.board[cell_row + dx][cell_col + dy]
            for dy in range(0, self.cell_size)
            for dx in range(0, self.cell_size)
        }

        all_values = {i for i in range(1, self.n + 1)}

        return all_values - row_values - col_values - cell_values

    def move_forward(self, row, col) -> tuple[int, int]:
        if col + 1 < self.n:
            return row, col + 1
        else:
            return row + 1, 0

    # why is pretty printing a board so HARD
    def board_repr(self, current_row, current_col):
        board_str = [['' for _ in range(len(self.board[0]))]
                     for _ in range(len(self.board))]

        current_cell = (current_row * self.n) + current_col

        for cell_number in range(self.n ** 2):
            row_ind = cell_number // self.n
            col_ind = cell_number % self.n

            if cell_number <= current_cell:
                pretty_cell = f'\033[1;37;46m{self.board[row_ind][col_ind]:3}\033[0;0m'
            else:
                pretty_cell = f'\033[1;37;40m{self.board[row_ind][col_ind]:3}\033[0;0m'

            board_str[row_ind][col_ind] = pretty_cell

        for row in board_str:
            for ind, modifier in enumerate(range(0, self.n + 1, self.cell_size)):
                # vertical divider
                row.insert(ind + modifier, '|')

        s = []
        for row in board_str:
            row_str = ''.join([val for val in row])
            s.append(row_str)

        divider = '-' * (1 + self.cell_size + (3 * self.n))
        for ind, modifier in enumerate(range(0, self.n + 1, self.cell_size)):
            s.insert(ind + modifier, f'{divider}')

        return '\n'.join(s)

    def __repr__(self):
        return self.board_repr(-1, -1)


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

    solver = SudokuSolver(example_board_1)
    solver.solve()

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

    solver = SudokuSolver(example_board_2)
    # solver.solve()
