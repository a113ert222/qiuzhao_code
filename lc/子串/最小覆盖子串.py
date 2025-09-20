from collections import Counter


class Solution:
    def minWindow(self, s: str, t: str) -> str:
        cnt_s = Counter()
        cnt_t = Counter(t)
        ans_l, ans_r = -1, len(s)-1
        l = 0
        for r in range(len(s)):
            cnt_s[s[r]] += 1
            while cnt_s >= cnt_t:
                if r - l < ans_r - ans_l:
                    ans_r = r
                    ans_l = l
                cnt_s[s[l]] -= 1
                l += 1
        return "" if ans_l < 0 else s[ans_l:ans_r+1]