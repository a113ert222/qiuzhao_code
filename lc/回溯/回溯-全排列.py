class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        ans = []
        s = []
        path = [0] * n
        def dfs(i, s):
            if i == n:
                ans.append(path)
                return
            for x in s:
                path[i] = x
                dfs(i+1, s-{x})
        dfs(0, set(nums))
        return ans