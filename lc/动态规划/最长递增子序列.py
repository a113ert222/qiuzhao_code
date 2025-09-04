class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        @cache
        def dfs(i):
            max_len = 0
            for j in range(i-1, -1, -1):
                if nums[j] < nums[i]:
                    max_len = max(max_len, dfs(j))
            return max_len+1
        return max(dfs(i) for i in range(len(nums)))