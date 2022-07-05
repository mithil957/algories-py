import unittest


class Partition:
    """
        Given a list of ints, we want to know if we can split the list such that
        the sub-lists have the same sum
    """

    @staticmethod
    def solve_brute_force(elements: list[int]) -> bool:
        """
        :param elements: list of elements
        :return: True if there exists a way to partition elements
            such that sub-lists have same sum
        """

        def recursive(current_set: list[int], index: int, target_sum: int) -> bool:
            if sum(current_set) == target_sum:
                return True

            if index >= len(elements):
                return False

            if recursive(current_set + [elements[index]], index + 1, target_sum):
                return True

            if recursive(current_set, index + 1, target_sum):
                return True

            return False

        # if the sum is odd, then it is not possible to divide the list into
        # sub-lists whose sum is the same. If the sum of sub-list is n then 2n can
        # never be odd
        if sum(elements) % 2:
            return False

        return recursive([], 0, sum(elements) // 2)

    @staticmethod
    def solve_dynamically(elements: list[int]) -> bool:
        """
        :param elements: list of elements
        :return: True if there exists a way to partition elements
            such that sub-lists have same sum

        Same as Knapsack
        Construct lookup table where
         we treat elements as jewels with equal weight and value
         and the target sum as capacity of knapsack
        """

        def recursive(index: int, current_sum: int, lookup_table: list[list[int]], target_sum: int) -> int:

            if 0 <= index < len(elements) and 0 <= current_sum < len(lookup_table[0]):
                if lookup_table[index][current_sum] != -1:
                    return lookup_table[index][current_sum]

                elif elements[index] <= current_sum:
                    sum_if_element_is_included = recursive(index - 1, current_sum - elements[index],
                                                           lookup_table, target_sum) + elements[index]

                    sum_if_element_is_not_included = recursive(index - 1, current_sum, lookup_table, target_sum)

                    lookup_table[index][current_sum] = max(sum_if_element_is_included, sum_if_element_is_not_included)

                else:
                    lookup_table[index][current_sum] = recursive(index - 1, current_sum, lookup_table, target_sum)

                # Move forward
                if current_sum + 1 < len(lookup_table[0]):
                    recursive(index, current_sum + 1, lookup_table, target_sum)
                else:
                    recursive(index + 1, 0, lookup_table, target_sum)
            else:
                return 0

        elements_sum = sum(elements)
        if elements_sum % 2:
            return False

        table = [[-1 for _ in range(elements_sum + 1)]
                 for _ in range(len(elements))]

        recursive(0, 0, table, elements_sum)
        return table[-1][-1] == elements_sum


class TestPartition(unittest.TestCase):
    def test_case_1(self):
        elements = [4, 2, 1, 3]
        res = Partition.solve_brute_force(elements)
        self.assertEqual(True, res)

        res = Partition.solve_dynamically(elements)
        self.assertEqual(True, res)
