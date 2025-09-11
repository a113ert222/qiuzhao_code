class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        slow, fast = 0, 1
        while fast < len(nums):
            if nums[slow] == 0 and nums[fast] != 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]
                slow += 1
            if nums[slow] != 0:
                slow += 1
            fast += 1