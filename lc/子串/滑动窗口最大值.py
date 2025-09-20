from collections import deque


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        ans = []
        q = deque()
        for i, num in enumerate(nums):
            while q and nums[q[-1]] < num:
                q.pop()
            q.append(i)
            left = i - k + 1
            if q[0] < left:
                q.popleft()
            if left >= 0:
                ans.append(nums[q[0]])
        return ans