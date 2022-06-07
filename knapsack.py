import unittest
from dataclasses import dataclass


@dataclass
class Jewel:
    value: int
    weight: int


class KnapSack:
    @staticmethod
    def solve_brute_force(jewels: list[Jewel], capacity: int, profit: int) -> int:
        if len(jewels) == 0:
            return profit

        if jewels[0].weight > capacity:
            return KnapSack.solve_brute_force(jewels[1:], capacity, profit)

        value_from_keeping_current_jewel = KnapSack.solve_brute_force(
            jewels[1:], capacity - jewels[0].weight, profit + jewels[0].value)

        value_from_discarding_current_jewel = KnapSack.solve_brute_force(
            jewels[1:], capacity, profit)

        return max(value_from_keeping_current_jewel, value_from_discarding_current_jewel)

    @staticmethod
    def solve_dynamically(jewels: list[Jewel], capacity: int) -> int:

        def dynamic_recursion(current_capacity: int, jewel_index: int,
                              lookup_table: list[list[int]]):

            if 0 <= jewel_index < len(table) and 0 <= current_capacity < len(table[0]):

                if lookup_table[jewel_index][current_capacity] != -1:
                    return lookup_table[jewel_index][current_capacity]

                elif jewels[jewel_index].weight <= current_capacity:
                    take_jewel = dynamic_recursion(current_capacity - jewels[jewel_index].weight,
                                                   jewel_index - 1, lookup_table) + jewels[jewel_index].value

                    do_not_take_jewel = dynamic_recursion(current_capacity,
                                                          jewel_index - 1, lookup_table)

                    lookup_table[jewel_index][current_capacity] = max(take_jewel, do_not_take_jewel)
                else:
                    lookup_table[jewel_index][current_capacity] = dynamic_recursion(current_capacity,
                                                                                    jewel_index - 1,
                                                                                    lookup_table)

                # Move forward
                if (current_capacity + 1) < len(lookup_table[0]):
                    dynamic_recursion(current_capacity + 1, jewel_index, lookup_table)
                else:
                    dynamic_recursion(0, jewel_index + 1, lookup_table)
            else:
                return 0

        # table of size number of jewels by capacity
        table = [[-1 for _ in range(capacity + 1)]
                 for _ in range(len(jewels))]

        dynamic_recursion(0, 0, table)
        return table[-1][-1]


class TestKnapSack(unittest.TestCase):

    def test_case_1(self):
        jewel_values = [60, 50, 70, 30]
        jewel_weights = [5, 3, 4, 2]
        max_weight = 5

        jewels = [Jewel(v, w) for v, w in zip(jewel_values, jewel_weights)]

        res = KnapSack.solve_brute_force(jewels, max_weight, 0)
        self.assertEqual(80, res)

        res = KnapSack.solve_dynamically(jewels, max_weight)
        self.assertEqual(80, res)
