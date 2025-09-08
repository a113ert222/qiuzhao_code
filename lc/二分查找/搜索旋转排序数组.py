class Solution:
    def findMin(self, nums: List[int]) -> int:
        l, r = 0, len(nums)-2
        while l <= r:
            mid = (l + r) // 2
            if nums[mid] < nums[-1]:
                r = mid - 1
            else:
                l = mid + 1
        return l
        
    def lower_bound(self, nums: List[int], left: int, right: int, target: int) -> int:
        l = left
        r = right
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left if nums[left] == target else -1

    def search(self, nums: List[int], target: int) -> int:
        min_index = self.findMin(nums)
        n = len(nums)
        if target > nums[-1]:
            ans = self.lower_bound(nums, 0, min_index-1, target)
        else:
            ans = self.lower_bound(nums, min_index, n-1, target)
        return ans