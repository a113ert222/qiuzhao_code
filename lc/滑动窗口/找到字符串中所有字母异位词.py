from collections import Counter


class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        cnt_p = Counter(p)
        cnt_s = Counter()
        ans = []
        for r, c in enumerate(s):
            cnt_s[c] += 1
            l = r - len(p) + 1
            if l < 0:
                continue
            if cnt_s == cnt_p:
                ans.append(l)
            cnt_s[s[l]] -= 1
            if cnt_s[s[l]] == 0:
                del cnt_s[s[l]]
        return ans