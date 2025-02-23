# 209.长度最小的子数组

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


