LeetCode Biweekly Contest 150

# [3467. 将数组按照奇数偶数转化](https://leetcode.cn/problems/transform-array-by-parity/description/)

破题：取余，记录几个奇数几个偶数

```python
class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        cnt = Counter(x % 2 for x in nums)
        return [0] * cnt[0] + [1] * cnt[1]
```
