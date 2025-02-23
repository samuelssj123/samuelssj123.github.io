# 209.长度最小的子数组

[Leetcode](https://leetcode.cn/problems/minimum-size-subarray-sum/description/)

- 滑动窗口： 由于题目要求连续子数组，滑动窗口是一种天然的解决方式。

由于数组元素都是正整数，当子数组的右边界向右扩展时，子数组的和会增大；当左边界向右收缩时，子数组的和会减小。利用这个特性，我们可以使用滑动窗口的方法来动态调整子数组的范围，从而找到最短的满足条件的子数组。

使用滑动窗口的方法，通过两个指针 left 和 right 来表示窗口的左右边界。right 指针不断向右移动，扩大窗口，同时累加窗口内元素的和 s。当窗口内元素的和 s 大于等于 target 时，尝试收缩窗口，即移动 left 指针，看是否可以在满足条件的情况下使窗口长度更小。在这个过程中，记录满足条件的最小窗口长度。

- 复杂度分析：
  
时间复杂度：O(n)，每个元素最多被访问两次（一次被右指针访问，一次被左指针访问）。

空间复杂度：O(1)，只使用了常数空间。

- 写法一：
  
```Python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n = len(nums)
        ans = n + 1 #inf
        s = 0
        left = 0
        for right , x in enumerate(nums): #right 会依次取到列表元素的索引，x 会依次取到列表中的元素。 x = nums[right] 扩大滑动窗口的范围
            s += x
            while s - nums[left] >= target: #收缩窗口
#由于每次收缩窗口时 left 指针最多移动到和 right 指针相同的位置（即窗口收缩到只剩一个元素）
#当 left 等于 right 时，s - nums[left] 就相当于 0，此时不会进入 while 循环继续收缩窗口，因为 s - nums[left] >= target 不成立。所以，不会出现 left 指针超过 right 指针的情况，因此不需要额外判断 left < right。
                s -= nums[left]
                left += 1
            if s >= target:
                ans = min(ans, right - left + 1)
        return ans if ans <= n else 0
```

- 写法二：

```Python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n = len(nums)
        ans = n + 1 #inf
        s = 0
        left = 0
        for right , x in enumerate(nums): #right 会依次取到列表元素的索引，x 会依次取到列表中的元素。 x = nums[right] 扩大滑动窗口的范围
            s += x
            while s >= target:
                ans = min(ans, right - left + 1)
                s -= nums[left]
                left += 1
        return ans if ans <= n else 0
```

由于题目中明确数组 nums 是由正整数组成，当窗口内元素和大于等于 target 时，通过不断移除左边界元素（因为元素都是正数，移除后和会减小），可以逐步找到满足条件的最小窗口。

# 713.乘积小于k的子数组

[Leetcode](https://leetcode.cn/problems/subarray-product-less-than-k/description/)

本题的破题点在于如何高效地找出满足乘积小于 k 的所有连续子数组。由于数组中的元素都是正整数，当子数组的右边界向右扩展时，子数组的乘积会增大；当左边界向右收缩时，子数组的乘积会减小。利用这个特性，我们可以使用滑动窗口的方法来动态调整子数组的范围，从而统计满足条件的子数组的个数。

```Python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k < 1:
            return 0
        ans = 0
        prod = 1
        left = 0
        for right, x in enumerate(nums):
            prod *= x
            while left <= right and prod >= k:
#在某些情况下，比如 k 非常小，而数组元素较大，可能会导致 prod 迅速增大，使得 left 指针需要不断右移来缩小窗口。如果不判断 left <= right，left 指针可能会右移到超过 right 指针的位置，此时再访问 nums[left] 就会出现索引越界错误。
                prod /= nums[left]
                left += 1
            ans += right - left + 1
        return ans
```
复杂度分析

时间复杂度：O(n)，每个元素最多被访问两次（一次被右指针访问，一次被左指针访问）。

空间复杂度：O(1)，只使用了常数空间。

# 3.无重复字符的最长子串

[Leetcode](https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/)

可以利用滑动窗口的思想，通过一个哈希表（在 Python 中可以使用 Counter）来记录每个字符在当前窗口中出现的次数。当遇到重复字符时，通过移动窗口的左边界来消除重复，从而保证窗口内的子串始终不包含重复字符。

- 复杂度分析:

时间复杂度：O(n)，每个字符最多被访问两次（一次被右指针访问，一次被左指针访问）。

空间复杂度：O(min(m, n))，其中 m 是字符集的大小（如 ASCII 字符集为 128），n 是字符串的长度。哈希表最多存储 m 个字符。

```Python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        ans = 0
        cnt = Counter() # hashmap char int
        left = 0
        for right, c in enumerate(s):
            cnt[c] += 1
            while cnt[c] >1: #当 cnt[c] > 1 时，说明当前字符 c 在窗口中出现了重复，需要移动窗口的左边界 left。在移动过程中，将 s[left] 在 Counter 中的计数减 1，直到 cnt[c] <= 1，即消除了重复字符。
                cnt[s[left]] -= 1
                left += 1
            ans = max(ans, right - left + 1)
        return ans
```
