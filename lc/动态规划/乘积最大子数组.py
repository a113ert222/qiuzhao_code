class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        f_max = [0] * n
        f_min = [0] * n
        f_max[0] = f_min[0] = nums[0]
        for i in range(1, n):
            f_max[i] = max(f_max[i-1] * nums[i], f_min[i-1] * nums[i], nums[i])
            f_min[i] = min(f_max[i-1] * nums[i], f_min[i-1] * nums[i], nums[i])
        return max(f_max)