class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = prices[0]
        ans = 0
        for p in prices:
            ans = max(p - min_price, ans)
            min_price = min(min_price, p)
        return ans