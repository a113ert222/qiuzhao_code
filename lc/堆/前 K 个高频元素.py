import collections

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        cnt = collections.Counter(nums)
        max_val = max(cnt.values())
        
        buckets = [[] for _ in range(max_val+1)]
        for x, c in cnt.items():
            buckets[c].append(x)
        
        ans = []
        for bucket in reversed(buckets):
            ans += bucket
            if len(ans) == k:
                return ans