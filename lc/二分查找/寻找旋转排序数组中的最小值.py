class Solution:
    def findMin(self, nums: List[int]) -> int:
        n = len(nums)
        l, r = 0, n-2
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] < nums[-1]:
                r = mid - 1
            else:
                l = mid + 1
        return l