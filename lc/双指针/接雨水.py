class Solution:
    def trap(self, height: List[int]) -> int:
        ans = 0
        n = len(height)
        pre = [0] * n
        suf = [0] * n
        pre[0] = height[0]
        for i in range(1, n):
            pre[i] = max(pre[i-1], height[i])

        suf[-1] = height[-1]
        for j in range(n-2, -1, -1):
            suf[j] = max(suf[j+1], height[j])

        for k in range(n):
            ans += min(pre[k], suf[k]) - height[k]
        return ans