class Solution:
    def climbStairs(self, n: int) -> int:
        @cache
        def dfs(i):
            if i <= 1:
                return 1
            return dfs(i-1) + dfs(i-2)
        return dfs(n)