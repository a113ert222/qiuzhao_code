import heapq
import random

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # heap = []
        # for num in nums:
        #     heapq.heappush(heap, num)
        #     if len(heap) > k:
        #         heapq.heappop()
        # return heap[0]
        def quick_select(nums, k):
            big, small, equal = [], [], []
            privot = random.choice(nums)
            for c in nums:
                if c > privot:
                    big.append(c)
                elif c < privot:
                    small.append(c)
                else:
                    equal.append(c)
            if len(big) >= k:
                return quick_select(big, k)
            elif len(nums) - len(small) < k:
                return quick_select(small, k - len(nums) + len(small))
            else:
                return privot
        return quick_select(nums, k)