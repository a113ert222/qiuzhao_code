from collections import defaultdict


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        record = defaultdict()
        ans = []
        for i in range(len(nums)):
            tmp = target - nums[i]
            if tmp in record:
                return [i, record[tmp]]
            record[nums[i]] = i
        return []