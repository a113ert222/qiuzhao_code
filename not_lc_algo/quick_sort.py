"""
[ 已处理且≤pivot的元素 | 已处理且>pivot的元素 | 未处理的元素 | pivot ]
  l         j-1         j         i-1         i         r-1       r
"""

"""
•
​​索引 l到 j-1​​：所有​​小于等于​​基准值的元素

•
​​索引 j到 i-1​​：所有​​大于​​基准值的元素

•
​​索引 i到 r-1​​：尚未处理的元素

•
​​索引 r​​：基准值本身
"""

def quick_sort(lst, l, r):
    if l >= r:
        return
    idx = patition(lst, l, r)
    quick_sort(lst, l, idx-1)
    quick_sort(lst, idx+1, r)

def patition(lst, l, r):
    pivot = lst[r]
    j = l
    for i in range(l, r):
        if lst[i] > pivot:
            lst[i], lst[j] = lst[j], lst[i]
            j += 1
    lst[j], lst[r] = lst[r], lst[j]
    return j
