import unittest


class Staircase:
    """
        We want to climb a staircase. We can take steps of certain sizes.
        How many ways are there to climb this staircase?
    """

    @staticmethod
    def solve_brute_force(steps: list[int], staircase_height: int) -> int:
        """
        :param steps:  a list of the size of steps you can take
        :param staircase_height: the number we want to reach by taking steps
        :return: the number of ways we can reach staircase_height
        """

        def recursive(current_height: int, counter: int) -> int:
            if current_height == staircase_height:
                return counter + 1
            elif current_height > staircase_height:
                return counter

            possible_ways = 0
            for step in steps:
                possible_ways += recursive(current_height + step, counter)

            return possible_ways

        return recursive(0, 0)

    @staticmethod
    def solve_dynamically(steps: list[int], staircase_height: int) -> int:
        """
        :param steps:  a list of the size of steps you can take
        :param staircase_height: the number we want to reach by taking steps
        :return: the number of ways we can reach staircase_height

        s_i is the ith element in steps
        f(k) = f(k - s_1) + f(k - s_2) + ... f(k - s_n)

        Ex. steps = [1, 2, 3] and staircase_height = 4
        f(4) = f(3) + f(2) + f(1)
        f(3) = f(2) + f(1) + f(0)
        f(2) = f(1) + f(0) + f(-1)

        There are repeated function calls, those value can be stored in a table.
        We start by calculating f(0) then f(1) up to f(k) and storing the values.
        """

        def recursive(current_height: int, lookup_table: dict[int, int]) -> int:
            if 0 <= current_height <= staircase_height:
                if current_height in lookup_table:
                    return lookup_table[current_height]
                else:
                    possible_ways = 0
                    for step in steps:
                        possible_ways += recursive(current_height - step, lookup_table)
                    lookup_table[current_height] = possible_ways

                # Move forward
                recursive(current_height + 1, lookup_table)
            else:
                return 0

        table = {0: 1}
        recursive(1, table)
        return table[staircase_height]


class TestStaircase(unittest.TestCase):
    def test_case_1(self):
        steps = [1, 2, 3, 4]
        height = 5

        res = Staircase.solve_brute_force(steps, height)
        self.assertEqual(15, res)

        res = Staircase.solve_dynamically(steps, height)
        self.assertEqual(15, res)

    def test_case_2(self):
        steps = [1, 2, 3]
        height = 4

        res = Staircase.solve_brute_force(steps, height)
        self.assertEqual(7, res)

        res = Staircase.solve_dynamically(steps, height)
        self.assertEqual(7, res)
