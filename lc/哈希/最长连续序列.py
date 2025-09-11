class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        st = set(nums)
        ans = 0
        for x in st:
            if x-1 in st:
                continue
            y = x + 1
            while y in st:
                y += 1
            ans = max(ans, y-x)
        return ans