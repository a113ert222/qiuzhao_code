class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        ans = []
        n = len(nums)
        nums.sort()
        for i in range(n):
            if nums[i] == nums[i-1] and i > 0:
                continue
            if nums[i] > 0:
                break
            l, r = i + 1, n-1
            while l < r:
                s = nums[i] + nums[l] + nums[r]
                if s > 0:
                    r -= 1
                elif s < 0:
                    l += 1
                else:
                    ans.append([nums[i], nums[l], nums[r]])
                    l += 1
                    while nums[l] == nums[l-1] and l < r:
                        l += 1
                    r -= 1
                    while nums[r] == nums[r+1] and l < r:
                        r -= 1
        return ans