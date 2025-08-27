class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        ans = []
        col = [0] * n # 记录每一行的皇后在哪一列（下标代表所在行数，值代表所在列数）

        def valid(r, c):
            for R in range(r):
                C = col[R]
                if r + c == R + C or r - c == R - C:
                    return False
            return True
        
        def dfs(r, s):
            if r == n:
                ans.append(['.'*c + 'Q' + '.'*(n-c-1) for c in col]) # 注意得有一个中括号
                return
            for c in s:
                if valid(r, c):
                    col[r] = c
                    dfs(r+1, s-{c})
        dfs(0, set(range(n)))
        return ans