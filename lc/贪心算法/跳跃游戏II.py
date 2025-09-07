class Solution:
    def jump(self, nums: List[int]) -> int:
        cur_right = 0
        next_right = 0
        ans = 0
        for i in range(len(nums) - 1): # 遍历的是所有造桥可能需要的位置，所以只需要到n-2就可以了
            next_right = max(next_right, i + nums[i])
            if i == cur_right:
                cur_right = next_right
                ans += 1
        return ans