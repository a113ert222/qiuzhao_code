class Solution:
    def longestCommonSubsequence(self, s: str, t: str) -> int:
        m, n = len(s), len(t)
        @cache
        def dfs(i, j):
            if i < 0 or j < 0:
                return 0
            if s[i] == t[j]:
                return dfs(i-1, j-1) + 1
            return max(dfs(i-1, j), dfs(i, j-1))
        return dfs(m-1, n-1)