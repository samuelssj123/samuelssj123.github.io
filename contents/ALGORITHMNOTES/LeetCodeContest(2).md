LeetCode Biweekly Contest 150

# [3467. 将数组按照奇数偶数转化](https://leetcode.cn/problems/transform-array-by-parity/description/)

破题：取余，记录几个奇数几个偶数

```python
class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        cnt = Counter(x % 2 for x in nums)
        return [0] * cnt[0] + [1] * cnt[1]
```

# [3468.可行数组的数目](https://leetcode.cn/problems/find-the-number-of-copy-arrays/description/)

```python
class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        mn, mx = -inf, inf
        for x, (u, v) in zip(original, bounds):
            d = x - original[0]
            mn = max(mn, u - d)
            mx = min(mx, v - d)
        return max(mx - mn + 1, 0)

"""
    1.copy[i]−copy[i−1]=original[i]−original[i−1]

    copy[1]−copy[0] = original[1]−original[0]
    copy[2]−copy[1] = original[2]−original[1]
    copy[3]−copy[2] = original[3]−original[2]
      ⋮
    copy[i]−copy[i−1] = original[i]−original[i−1] 

    累加等号左边的所有项，累加等号右边的所有项，得:copy[i]−copy[0]=original[i]−original[0]
    移项得:copy[i]=copy[0]+original[i]−original[0]
    确定了 copy[0]，那么整个数组也就确定了。
    所以 copy[0] 的取值范围（整数集合）的大小就是答案。

    2.题目要求

    ui ≤ copy[i] ≤vi
    ​设di = original[i] − original[0] 用 copy[i]=copy[0]+di替换上式中的 copy[i]，得
    ui ≤ copy[0] + di ≤vi
    ​移项得 ui - di ≤ copy[i] ≤vi - di

    可以得到 n 个关于 copy[0] 的不等式，或者说区间：

    [u0, v0]
    [u1 - d1, v1 - d1]
    [u2 - d2, v2 - d2]
      ⋮
    [u_{n - 1} - d1_{n - 1}, v1_{n - 1} - d1_{n - 1}]
    
    的交集，即为 copy[0] 能取到的值。

    区间交集的大小即为答案。如果交集为空，返回 0。

    交集的范围：所有区间左端点取最大值，右端点取最小值。
"""
```


# [3469.移除所有数组元素的最小代价](https://leetcode.cn/problems/find-minimum-cost-to-remove-array-elements/description/)


```python
class Solution:
    def minCost(self, nums: List[int]) -> int:
        n = len(nums)
        f = [[0] * i for i in range(n + 1)]
        f[n] = nums
        f[n - 1] = [max(x, nums[-1]) for x in nums]
        for i in range(n - 3 + n % 2, 0, -2):
            b, c = nums[i], nums[i + 1]
            for j in range(i):
                a = nums[j]
                f[i][j] = min(f[i + 2][j] + max(b, c),
                            f[i + 2][i] + max(a, c),
                            f[i + 2][i + 1] + max(a, b))
        return f[1][0]
```

# [3470.全排列Ⅳ](https://leetcode.cn/problems/permutations-iv/description/)
