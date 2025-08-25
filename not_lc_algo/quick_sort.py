"""
[ 已处理且大于pivot的元素 | 已处理且小于等于pivot的元素 | 未处理的元素 | pivot ]
  l         j-1         j         i-1         i         r-1       r
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

if __name__ == '__main__':
    lst = [1,3,2,5,3,6,2,7]
    quick_sort(lst,0,len(lst)-1)
    print(lst)