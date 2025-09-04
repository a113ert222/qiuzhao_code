class Solution:
    def longestValidParentheses(self, s: str) -> int:
        ans = 0
        stack = []
        for i, c in enumerate(s):
            if stack and s[stack[-1]] == '(' and c == ')':
                stack.pop()
                if not stack:
                    ans = max(ans, i+1)
                else:
                    ans = max(ans, i-stack[-1])
            else:
                stack.append(i)
        return ans