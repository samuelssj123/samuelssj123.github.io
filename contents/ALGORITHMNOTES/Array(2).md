# 209.长度最小的子数组  
[题目链接](https://leetcode.cn/problems/minimum-size-subarray-sum/) [文章讲解](https://programmercarl.com/0209.%E9%95%BF%E5%BA%A6%E6%9C%80%E5%B0%8F%E7%9A%84%E5%AD%90%E6%95%B0%E7%BB%84.html) [视频讲解](https://www.bilibili.com/video/BV1tZ4y1q7XE)

- 暴力解法：（超出时间限制）
```
from typing import List

class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        sublength = 0
        result = float('inf')
        for i in range(0, len(nums)):
            sum = 0
            for j in range(i, len(nums)):
                sum += nums[j]
                if sum >= target:
                    sublength = j - i + 1
                    result = min(result, sublength)  # Python 没有三元比较符，更新最小长度
                    break
        return result if result != float('inf') else 0  # 如果没有找到满足条件的子数组，返回 0
```

- 滑动窗口：
**滑动窗口**：不断调节子序列的起始位置和终止位置。终止位置需要遍历到最后，因此关键点是如何确定起始位置，否则仍然是暴力解法。  
需要明确的问题：1.窗口内是什么？是求和的元素集合。2.如何移动窗口的起始位置？如果当前窗口的值大于等于s，窗口就要向前移动了，也就是窗口该缩小了。3.如何移动窗口的结束位置？窗口的结束位置就是遍历数组的指针，也就是for循环里的索引。
为什么用while不用if？集合需要不断缩小。  
**时间复杂度**：每个元素在滑动窗后进来操作一次，出去操作一次，每个元素都是被操作两次，所以时间复杂度是 2 × n 也就是O(n)。  
```
from typing import List

class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        sublength = 0
        result = float('inf')
        i = 0 
        sum = 0
        for j in range(0, len(nums)):
            sum += nums [j]
            while sum >= target:
                sublength = j - i + 1
                result = min(result, sublength)
                sum -= nums[i]
                i += 1
        return result if result != float('inf') else 0  
```

# 59.螺旋矩阵II  
[题目链接](https://leetcode.cn/problems/spiral-matrix-ii/) [文章讲解](https://programmercarl.com/0059.%E8%9E%BA%E6%97%8B%E7%9F%A9%E9%98%B5II.html) [视频讲解](https://www.bilibili.com/video/BV1SL4y1N7mV/)

```
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        matrix = [[0] * n for _ in range(n)]
        startx = 0
        starty = 0
        offset = 1
        count = 1
        loop = n // 2
        while loop > 0 :
            for j in range (starty, n - offset) :
                matrix[startx][j] = count
                count += 1
            for i in range (startx, n - offset) :
                matrix[i][n - offset] = count
                count += 1
            for j in range(n - offset, starty, -1):
                matrix[n - offset][j] = count
                count += 1
            for i in range(n - offset, startx, -1):
                matrix[i][starty] = count
                count += 1
            startx += 1
            starty += 1
            offset += 1
            loop -= 1
        if n % 2 == 1:
            matrix[startx][starty] = count
        return matrix
```
