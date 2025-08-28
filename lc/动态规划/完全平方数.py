@cache
def dfs(i, n):
    if i == 0:
        return inf if n else 0
    if n < i*i:
        return dfs(i-1, n)
    return min(dfs(i-1, n), dfs(i, n-i*i)+1)

class Solution:
    def numSquares(self, n: int) -> int:
        return dfs(isqrt(n), n)
