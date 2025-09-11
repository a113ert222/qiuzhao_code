class Solution:
    def maxArea(self, height: List[int]) -> int:
        n = len(height)
        l, r = 0, n-1
        max_w = 0
        while l < r:
            tmp = (r-l) * min(height[l], height[r])
            max_w = max(tmp, max_w)
            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1
        return max_w