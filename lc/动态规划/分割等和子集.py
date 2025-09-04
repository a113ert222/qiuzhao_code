class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        if sum(nums) % 2 == 1:
            return False
        s = sum(nums)
        @cache
        def dfs(i, j):
            if i < 0:
                return j == 0
            if j < nums[i]:
                return dfs(i-1, j)
            return dfs(i-1, j) or dfs(i-1, j-nums[i])
        return dfs(n-1, s//2)