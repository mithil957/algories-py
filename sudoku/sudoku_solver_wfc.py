from dataclasses import dataclass
from collections.abc import Iterator
from enum import Enum

import os
import time
from collections import deque, namedtuple, Counter
from operator import itemgetter

from sudoku_solver import Solver


def clear() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')


"""
    Sudoku Solver
    - wave function collapse approach (constraint propagation + backtracking)
    - Given a 2d array of size N by N
    
    - board is a list[list[int]]
        - where each cell have 0 or some value between 1 and N
        - 0 represents an unfilled cell
        
    - board_state is dict[Cell, set[int]] 
        - for each cell, we calculate a set of possibilities
    
    Solving a board
    - loop
    - check if board is solved
        - keep track of number of filled cells
        - if number of cells == n^2 -> BOARD_SOLVED
    
    - check for invalidity along all cells
        - a cell is invalid if it is unfilled (equal to 0) and set of possibilities is size 0
        - if invalid -> BACKTRACK
    
    - check for invalidity along rows, cols, and squares
        - a (row|col|square) of cells is invalid 
            - if there exists two cells that can only be the same value
        - if invalid -> BACKTRACK
    
    - then MOVE_FORWARD
    
    - MOVE_FORWARD
        - pattern is (look for cell -> look for value)
        - look for a cell to work on
            - find cell with the LOWEST number of possibilities
            - keep track of what cell is being worked on
            
        - look for what value to fill cell with
            - find value such that it has the HIGHEST reduction in the possibilities of neighboring cells
            - create a Move object (cell, value, other values that could have picked, effected cells)
                - save other values that could have been picked in sorted order
                    - order such that the first value has the HIGHEST reduction and last value has LOWEST reduction
            - place Move object on to moves 
                - moves is a deque[Move], we use it as a stack
            - update board
            - update board_state -> propagate move
    
    - BACKTRACK
        - check if moves is empty
            - if empty -> BOARD_NOT_SOLVABLE
            
        - pattern is (look for value -> look for cell)
        - pop a move from moves, call this last_move
        - undo move by giving back value of move to all effected cells
        - if last_move has other values that could have been picked
            - for each cell in empty_cells
                - calculate possible values for cell and update board_state
            - set board to a value 
            - update board_state
            - find all cells that were effected
            - create Move object and place on moves stack
            - break backtrack
        - if last_move has no other values that could have been picked
            - add it to stack of empty cells -> empty_cells
"""


@dataclass(frozen=True)
class Cell:
    row: int
    col: int


@dataclass
class Move:
    cell: Cell
    value: int
    other_possible_values: list[int]
    effected_cells: set[Cell]


BoardSolved = namedtuple('BoardSolved', [])
BackTrack = namedtuple('BackTrack', [])
MoveForward = namedtuple('MoveForward', [])


class WFCSolver(Solver):
    def __init__(self):
        self.board: list[list[int]] = [[]]
        self.n: int = 0
        self.cell_size: int = 0
        self.moves: deque[Move] = deque()
        self.current_cell: Cell = Cell(0, 0)

        self.number_of_filled_cells = 0
        self.board_state: dict[Cell, set[int]] = {}

        # for visualizing
        self.visualize: bool = False
        self.steps: int = 0

    def set_board(self, board: list[list[int]], visualize: bool):
        self.board = board
        self.n = len(board)
        self.cell_size = int(self.n ** .5)
        self.moves: deque[Move] = deque()
        self.current_cell: Cell = Cell(0, 0)

        self.number_of_filled_cells = self.get_filled_cells()
        self.board_state = self.scan_board()

        self.visualize = visualize
        self.steps = 0

    def solve(self, board: list[list[int]], visualize: bool = False):
        self.set_board(board, visualize)

        while True:
            self.visualize_process()
            match self.get_action():
                case BoardSolved():
                    return
                case MoveForward():
                    cell = self.look_for_cell_to_work_on()
                    move = self.get_move(cell)
                    self.make_move_forward(move)

                case BackTrack():
                    match self.backtrack():
                        case Move() as move:
                            self.make_move_forward(move)
                        case None:
                            raise Exception('board not solvable')

    def get_action(self) -> BoardSolved | BackTrack | MoveForward:
        if self.number_of_filled_cells == (self.n ** 2):
            return BoardSolved()

        all_cells = (Cell(i // self.n, i % self.n) for i in range(self.n ** 2))
        for cell in all_cells:
            if len(self.board_state[cell]) == 0:
                return BackTrack()

        if self.look_for_invalidity_along_rows():
            return BackTrack()

        if self.look_for_invalidity_along_cols():
            return BackTrack()

        if self.look_for_invalidity_along_squares():
            return BackTrack()

        return MoveForward()

    def get_filled_cells(self) -> int:
        count = 0
        all_cells = (Cell(i // self.n, i % self.n) for i in range(self.n ** 2))
        for cell in all_cells:
            if self.board[cell.row][cell.col] != 0:
                count += 1
        return count

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

    def get_cells_along_row(self, cell: Cell) -> set[Cell]:
        return {Cell(cell.row, col) for col in range(self.n)}

    def get_cells_along_col(self, cell: Cell) -> set[Cell]:
        return {Cell(row, cell.col) for row in range(self.n)}

    def get_cells_along_square(self, cell) -> set[Cell]:
        square_row = (cell.row // self.cell_size) * self.cell_size
        square_col = (cell.col // self.cell_size) * self.cell_size

        return {
            Cell(square_row + dx, square_col + dy)
            for dy in range(0, self.cell_size)
            for dx in range(0, self.cell_size)
        }

    def look_for_invalidity_among_cells(self, cells: Iterator[Cell]) -> bool:
        store = set()
        for cell in cells:
            possibility = list(self.board_state[cell])[0]
            if possibility in store:
                return True
            store.add(possibility)
        return False

    def look_for_invalidity_along_rows(self) -> bool:
        for row_ind in range(self.n):
            cells = self.get_cells_along_row(Cell(row_ind, 0))
            cells = filter(lambda c: len(self.board_state[c]) == 1, cells)
            cells = filter(lambda c: self.board[c.row][c.col] == 0, cells)
            if self.look_for_invalidity_among_cells(cells):
                return True
        return False

    def look_for_invalidity_along_cols(self) -> bool:
        for col_ind in range(self.n):
            cells = self.get_cells_along_col(Cell(0, col_ind))
            cells = filter(lambda c: len(self.board_state[c]) == 1, cells)
            cells = filter(lambda c: self.board[c.row][c.col] == 0, cells)
            if self.look_for_invalidity_among_cells(cells):
                return True
        return False

    def look_for_invalidity_along_squares(self) -> bool:
        for row_ind in range(0, self.n, self.cell_size):
            for col_ind in range(0, self.n, self.cell_size):
                cells = self.get_cells_along_square(Cell(row_ind, col_ind))
                cells = filter(lambda c: len(self.board_state[c]) == 1, cells)
                cells = filter(lambda c: self.board[c.row][c.col] == 0, cells)
                if self.look_for_invalidity_among_cells(cells):
                    return True
        return False

    def make_move_forward(self, move: Move) -> None:
        self.moves.append(move)
        self.current_cell = move.cell
        self.visualize_process()

        self.board[move.cell.row][move.cell.col] = move.value
        self.board_state[move.cell] = {move.value}
        # propagate choice, remove that value from effected cells
        for cell in move.effected_cells:
            self.board_state[cell].remove(move.value)

        self.number_of_filled_cells += 1

    def look_for_cell_to_work_on(self) -> Cell:
        # TODO how to decide which cell to work on
        # pick cell with the lowest amount of possibilities
        lowest_possibility_count = float('inf')
        cell_to_work_on: Cell = Cell(0, 0)

        for cell, possibilities in self.board_state.items():
            if self.board[cell.row][cell.col] == 0 and len(possibilities) < lowest_possibility_count:
                lowest_possibility_count = len(possibilities)
                cell_to_work_on = cell

        return cell_to_work_on

    def get_neighbor_cells(self, cell: Cell) -> set[Cell]:
        row_cells = self.get_cells_along_row(cell)
        col_cells = self.get_cells_along_col(cell)
        square_cells = self.get_cells_along_square(cell)

        return (row_cells | col_cells | square_cells) - {cell}

    def get_move(self, cell: Cell) -> Move:
        possibilities = self.board_state[cell]
        neighbor_cells = self.get_neighbor_cells(cell)

        # if only one choice left
        if len(possibilities) == 1:
            choice = next(iter(possibilities))
            effected_cells = self.get_effected_cells(neighbor_cells, choice)
            return Move(cell, choice, [], effected_cells)

        possibility_counts = self.get_possibility_counts(neighbor_cells)
        choices = Counter({p: possibility_counts[p] for p in possibilities})

        # sorted from THE HIGHEST reduction to the LOWEST reduction
        choices_sorted_by_count = [c[0] for c in choices.most_common()]

        # TODO how to pick choice/value for cell
        # pick the choice/value that leads the highest reduction of choices
        choice = choices_sorted_by_count.pop(0)

        effected_cells = self.get_effected_cells(neighbor_cells, choice)

        return Move(cell, choice, choices_sorted_by_count, effected_cells)

    def get_possibility_counts(self, cells: set[Cell]) -> Counter:
        counter = Counter()
        for cell in cells:
            # count only unmarked cells
            if self.board[cell.row][cell.col] == 0:
                p = self.board_state[cell]
                counter.update(p)

        return counter

    def get_effected_cells(self, neighbor_cells: set[Cell], choice: int) -> set[Cell]:
        # cells whose possibilities contain this choice
        # so if that choice were picked, it would affect these cells
        effected_cells = set()

        for cell in neighbor_cells:
            if self.board[cell.row][cell.col] == 0 and choice in self.board_state[cell]:
                effected_cells.add(cell)

        return effected_cells

    def backtrack(self) -> Move | None:
        while len(self.moves) != 0:
            last_move = self.moves.pop()
            self.number_of_filled_cells -= 1

            # undo our choice, give back value to effected cells
            for cell in last_move.effected_cells:
                self.board_state[cell].add(last_move.value)

            if len(last_move.other_possible_values) > 0:
                choice = last_move.other_possible_values.pop(0)
                neighbor_cells = self.get_neighbor_cells(last_move.cell)
                effected_cells = self.get_effected_cells(neighbor_cells, choice)
                return Move(last_move.cell, choice, last_move.other_possible_values, effected_cells)

            else:
                # make cell empty, set value to 0 and possibilities to 0
                self.board[last_move.cell.row][last_move.cell.col] = 0
                self.board_state[last_move.cell] = self.calculate_possible_values(last_move.cell)

            self.visualize_process()

        return None

    def visualize_process(self) -> None:
        if not self.visualize:
            return

        clear()
        self.steps += 1
        print(f'Steps: {self.steps}', end='\n\n')
        print(self)
        time.sleep(1 / 60)

    def __repr__(self):
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

        cell_number = (self.current_cell.row * self.n) + self.current_cell.col
        spacing = 5 + len(str(self.n))

        for row_ind, row in enumerate(s):
            str_row = ['|' for _ in range(self.cell_size)]

            for col_ind, val_str in enumerate(row):
                for ind, i in enumerate(range(0, self.n, self.cell_size)):
                    if row_ind * self.n + col_ind == cell_number:
                        str_row[ind] += f'\033[1;37;46m{val_str[i: i + self.cell_size]:{spacing}}\033[0;0m'
                    elif self.board[row_ind][col_ind] != 0:
                        str_row[ind] += f'\033[1;37;42m{val_str[i: i + self.cell_size]:{spacing}}\033[0;0m'
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

    def __str__(self):
        return self.__repr__()


if __name__ == '__main__':
    # 4 by 4
    example_board_0 = [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [2, 0, 0, 1],
        [0, 0, 4, 0]
    ]
    solver = WFCSolver()
    solver.solve(example_board_0, visualize=True)

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

    solver.solve(example_board_1, visualize=True)

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

    solver.solve(example_board_2, visualize=True)
