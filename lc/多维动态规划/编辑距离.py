class Solution:
    def minDistance(self, s: str, t: str) -> int:
        m, n = len(s), len(t)
        @cache
        def dfs(i, j):
            if i < 0:
                return j + 1
            if j < 0:
                return i + 1
            if s[i] == t[j]:
                return dfs(i-1, j-1)
            return min(dfs(i-1, j), dfs(i, j-1), dfs(i-1, j-1)) + 1
        return dfs(m-1, n-1)