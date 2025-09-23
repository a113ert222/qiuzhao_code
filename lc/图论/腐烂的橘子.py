class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        fresh = 0
        q = []
        for i, row in enumerate(grid):
            for j, c in enumerate(row):
                if c == 1:
                    fresh += 1
                elif c == 2:
                    q.append((i, j))
        ans = 0
        while q and fresh:
            ans += 1
            temp = q
            q = []
            for x, y in temp:
                for i, j in (x-1, y), (x+1, y), (x, y-1), (x, y+1):
                    if 0 <= i < m and 0 <= j < n and grid[i][j] == 1:
                        grid[i][j] = 2
                        q.append((i, j))
                        fresh -= 1
        return ans if fresh == 0 else -1