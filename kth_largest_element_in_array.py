import unittest
from collections import deque


def search_helper(nums, target):

    lower = 0
    upper = len(nums) - 1
    middle = (lower + upper) // 2
    pivot = nums[middle]

    index_list = deque()
    index_list.append(middle)

    pivot_list = [pivot]

    for index in range(len(nums)):
        if index == middle:
            continue

        # numbers greater than pivot on left
        # numbers lower than pivot on right
        if nums[index] >= pivot:
            index_list.appendleft(index)
            pivot_list.insert(0, nums[index])
        else:
            index_list.append(index)
            pivot_list.insert(-1, nums[index])

    middle_loc = None
    for index, i in enumerate(index_list, 1):
        if i == middle:
            middle_loc = index
            break

    if middle_loc is None:
        return None

    if middle_loc == target:
        return pivot

    if middle_loc > target:
        return search_helper(pivot_list[:middle_loc], target)

    if middle_loc < target:
        return search_helper(pivot_list[middle_loc:], target - middle_loc)


class KthLargestElementTest(unittest.TestCase):
    def test_case_1(self):
        result = search_helper([3, 2, 1, 5, 6, 4], 2)
        self.assertEqual(6, result)

    def test_case_2(self):
        result = search_helper([3, 2, 3, 1, 2, 4, 5, 5, 6], 1)
        self.assertEqual(4, result)

    def test_case_3(self):
        result = search_helper([2, 1], 2)
        self.assertEqual(1, result)
