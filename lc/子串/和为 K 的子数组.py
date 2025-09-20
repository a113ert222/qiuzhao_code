from collections import defaultdict


class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        ans = 0
        n = len(nums)
        pre = [0] * (n+1)
        for i in range(n):
            pre[i+1] = pre[i] + nums[i]
        ans = 0
        cnt = defaultdict(int)
        for i, num in enumerate(pre):
            ans += cnt[num-k]
            cnt[num] += 1
        return ans