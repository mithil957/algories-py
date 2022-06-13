from dataclasses import dataclass

import os
import time
from collections import deque, namedtuple, Counter

from sudoku.sudoku_solver import Solver


def clear() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')


"""
    Sudoku Solver
    - wave function collapse approach (constraint propagation + backtracking)
    - board size is a 2d array of size N by N
    - create lookup table T where 
        - key is (i,j), coordinate of cell
        - value is set of possible values for cell at ith row and jth col
    
    - loop
        - cell -> find key in T 
            - is unmarked on board
            - *whose set of possible values is closest to 1 but not less than 1
            - we will work on this cell
            - b/c we have the highest chances of getting it right
            - if key is unmarked and set of possible values is 0 -> backtrack
            - if all keys are marked then board is solved -> break loop
        
        - looking up key in T gives us possible values P
        
        - R_v = list of sets where each set is the possible values along cell's row
        - C_v = list of sets where each set is the possible values along cell's col
        - S_v = list of sets where each set is the possible values along cell's square
        
        - counter
            - for each set in R_v, C_v, S_v 
                - we count the number of times a possible value has shown up
        
        - possible_choices is a table
            - where key is choice (values in P and counter)
            - where value is the number of counts
        
        - at this point we have a cell (i, j) and possible_choices
        - *we mark the cell with choice (value) that has the highest count
        - create object Move(i, j, choice, possible_choices)
        - push onto stack
            - we want to push all the information we used to make a choice
            - so when we backtrack
                - we know where we were
                - we know what choice we made
                - we know what choices could have been made
        
        - remove choice from R_v, C_v, S_v
    
        
    * whose set of possible value is closest to 1
        - we want to pick a cell whose possibilities are the smallest
    
    * we mark the cell with choice (value) that has the highest count
        - from the possibilities
            - we pick one that effects the most cells
"""

Cell = namedtuple('Cell', ['row', 'col'])


@dataclass
class Move:
    cell: Cell
    value: int
    other_possible_values: list[int]
    effected_cells: set[Cell]


BoardSolved = namedtuple('BoardSolved', [])
Backtrack = namedtuple('Backtrack', [])


class WFCSolver(Solver):
    def __init__(self, board: list[list[int]]):
        self.board = board
        self.n = len(board)
        self.cell_size = int(self.n ** .5)
        self.board_state = self.scan_board()
        self.moves: deque[Move] = deque()
        self.current_cell: Cell = Cell(0, 0)

        # for visualizing
        self.visualize = False
        self.steps = 0

    def solve(self):
        while True:
            match self.look_for_cell():
                case BoardSolved():
                    break

                case Backtrack():
                    self.backtrack()

                case Cell(_, _) as cell:
                    self.current_cell = cell
                    self.make_move_on_board(self.current_cell)

    def scan_board(self) -> dict[Cell, set[int]]:
        board_possibilities = {}

        for cell_number in range(self.n ** 2):
            current_cell = Cell(cell_number // self.n, cell_number % self.n)

            match self.board[current_cell.row][current_cell.col]:
                case 0:
                    possibilities_for_current_cell = self.calculate_possible_values(current_cell)
                    board_possibilities[current_cell] = possibilities_for_current_cell
                case _ as val:
                    board_possibilities[current_cell] = {val}

        return board_possibilities

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

    def look_for_cell(self) -> Cell | BoardSolved | Backtrack:
        smallest_set_of_possibilities = float('inf')
        result_cell: Cell | None = None

        for cell, possibilities in self.board_state.items():
            # cell is marked
            if self.board[cell.row][cell.col] != 0:
                continue

            # cell is unmarked and no possibilities
            elif len(possibilities) == 0:
                return Backtrack()

            # cell is unmarked and its possibilities are smaller than current smallest set
            elif len(possibilities) < smallest_set_of_possibilities:
                result_cell = cell
                smallest_set_of_possibilities = len(possibilities)

        if result_cell is None:
            return BoardSolved()
        else:
            return result_cell

    def make_move_on_board(self, current_cell: Cell) -> None:
        if self.visualize:
            self.visualize_process()

        possibilities = self.board_state[current_cell]

        row_cells = self.get_cells_along_row(current_cell)
        col_cells = self.get_cells_along_col(current_cell)
        square_cells = self.get_cells_along_square(current_cell)

        neighbor_cells = (row_cells | col_cells | square_cells) - {current_cell}

        # if we only have one choice
        # append move and mark board
        if len(possibilities) == 1:
            choice = possibilities.pop()
            effected_cells = self.propagate_choice(neighbor_cells, choice) | {current_cell}

            move = Move(current_cell, choice, [], effected_cells)
            self.moves.append(move)
            self.board[current_cell.row][current_cell.col] = move.value
            self.board_state[current_cell] = {move.value}

            return

        possibility_counts = self.get_possibility_counts(neighbor_cells)
        choices = Counter({p: possibility_counts[p] for p in possibilities})

        # most_common returns list[tuple] -> tuple is (key, value) -> pull key
        choices_sorted_by_count = [c[0] for c in choices.most_common()]

        # TODO how to pick the value
        choice_with_lowest_count = choices_sorted_by_count.pop(0)

        effected_cells = self.propagate_choice(neighbor_cells, choice_with_lowest_count) | {current_cell}

        move = Move(current_cell, choice_with_lowest_count, choices_sorted_by_count, effected_cells)
        self.moves.append(move)
        self.board[current_cell.row][current_cell.col] = choice_with_lowest_count
        self.board_state[current_cell] = {choice_with_lowest_count}

    def get_cells_along_row(self, cell: Cell):
        return {Cell(cell.row, col) for col in range(self.n)}

    def get_cells_along_col(self, cell: Cell):
        return {Cell(row, cell.col) for row in range(self.n)}

    def get_cells_along_square(self, cell):
        square_row = (cell.row // self.cell_size) * self.cell_size
        square_col = (cell.col // self.cell_size) * self.cell_size

        return {
            Cell(square_row + dx, square_col + dy)
            for dy in range(0, self.cell_size)
            for dx in range(0, self.cell_size)
        }

    def get_possibility_counts(self, cells: set[Cell]) -> Counter:
        counter = Counter()
        for cell in cells:
            # count only unmarked cells
            if self.board[cell.row][cell.col] == 0:
                p = self.board_state[cell]
                counter.update(p)

        return counter

    def propagate_choice(self, cells: set[Cell], choice: int) -> set[Cell]:
        effected_cells = set()

        for cell in cells:
            if self.board[cell.row][cell.col] == 0 and choice in self.board_state[cell]:
                self.board_state[cell].remove(choice)
                effected_cells.add(cell)

        return effected_cells

    def undo_choice(self, cells: set[Cell], choice: int) -> None:
        for cell in cells:
            if self.board[cell.row][cell.col] == 0:
                self.board_state[cell].add(choice)

    def backtrack(self):
        cannot_make_move = True

        while cannot_make_move:

            if self.visualize:
                self.visualize_process()

            last_move = self.moves.pop()
            self.current_cell = last_move.cell
            self.board[last_move.cell.row][last_move.cell.col] = 0
            self.undo_choice(last_move.effected_cells, last_move.value)

            if len(last_move.other_possible_values) > 0:
                cannot_make_move = False
                self.board_state[last_move.cell] = set(last_move.other_possible_values)
            else:
                possibilities = self.calculate_possible_values(last_move.cell)
                self.board_state[last_move.cell] = possibilities

    def visualize_process(self) -> None:
        clear()
        self.steps += 1
        print(f'Steps: {self.steps}', end='\n\n')
        print(self.board_repr(self.current_cell))
        time.sleep(1 / 60)

    def board_repr(self, cell: Cell):
        s: list[list[set[int] | str]] = [[self.board_state[Cell(row_ind, col_ind)]
                                          for col_ind in range(self.n)]
                                         for row_ind in range(self.n)]

        for row_ind in range(self.n):
            for col_ind in range(self.n):
                val = s[row_ind][col_ind]
                if len(val) == 1 and self.board[row_ind][col_ind] != 0:
                    p1 = ' ' * (self.cell_size + 2)
                    p2 = ' ' * (self.n - self.cell_size - 2)
                    s[row_ind][col_ind] = p1 + str(list(val)[0]) + p2
                    continue

                cell_s = ''
                for i in range(1, self.n + 1):
                    if i in val:
                        cell_s += str(i)
                    else:
                        cell_s += ' '
                s[row_ind][col_ind] = cell_s

        bs = []

        cell_number = (cell.row * self.n) + cell.col
        for row_ind, row in enumerate(s):
            str_row = ['|' for _ in range(self.cell_size)]

            for col_ind, val_str in enumerate(row):
                for ind, i in enumerate(range(0, self.n, self.cell_size)):
                    if self.board[row_ind][col_ind] != 0:
                        str_row[ind] += f'\033[1;37;42m{val_str[i: i + self.cell_size]:5}\033[0;0m'
                    elif row_ind * self.n + col_ind == cell_number:
                        str_row[ind] += f'\033[1;37;46m{val_str[i: i + self.cell_size]:5}\033[0;0m'
                    else:
                        str_row[ind] += f'{val_str[i: i + self.cell_size]:5}'

                    str_row[ind] += '|'

            if row_ind % self.cell_size == 0:
                bs.append(' ' * len(str_row[0]))

            bs.extend(str_row)
            bs.append('-' * (6 * self.n + 1))

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

    solver = WFCSolver(example_board_1)
    solver.visualize = True
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

    solver = WFCSolver(example_board_2)
    solver.solve()
    print(solver)
