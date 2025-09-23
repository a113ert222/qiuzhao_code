class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        g = [[] for _ in range(numCourses)]
        for a, b in prerequisites:
            g[b].append(a)
        colors = [0] * numCourses
        def dfs(i):
            colors[i] = 1
            for y in g[i]:
                if colors[y] == 1 or colors[y]== 0 and dfs(y):
                    return True
            colors[i] = 2
            return False
        for i, c in enumerate(colors):
            if c == 0 and dfs(i):
                return False
        return True