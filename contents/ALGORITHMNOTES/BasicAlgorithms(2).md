# 11.盛最多水的容器

[Leetcode](https://leetcode.cn/problems/container-with-most-water/)

- 思路：

本题的破题点在于如何高效地遍历所有可能的容器组合，以找到最大的容纳水量。由于容器的容量由两个因素决定：容器的宽度（两条垂线之间的距离）和容器的高度（两条垂线中较短的那条的高度）。如果使用暴力法，需要遍历所有可能的两条垂线组合，时间复杂度为 ，效率较低。

我们可以采用双指针法来优化这个过程。双指针法的核心思想是，从数组的两端开始向中间移动指针，每次移动较短的那条垂线对应的指针。这是因为移动较短的垂线有可能找到更高的垂线，从而增加容器的高度，进而有可能增加容器的容量；而移动较长的垂线，容器的高度不会增加，只会因为宽度的减小而使容量减小。

- 代码思路：
双指针法：用两个指针分别指向数组的起始位置（left）和结束位置（right）。

计算面积：容器的面积由两个因素决定：宽度：right - left。高度：min(height[left], height[right])。面积公式为：area = (right - left) * min(height[left], height[right])。

移动指针：为了找到更大的面积，移动高度较小的指针：如果 height[left] < height[right]，则 left 右移。否则，right 左移。

更新最大面积：在每次计算面积后，更新最大面积 ans。

- 关键点：
- 
双指针法：通过移动指针来高效地找到最大面积。

面积计算：面积由宽度和最小高度决定。

指针移动策略：移动高度较小的指针，因为移动高度较大的指针不会增加面积。

时间复杂度优化：双指针法将时间复杂度从暴力解法的 O(n²) 优化到 O(n)。

- 复杂度分析：
  
时间复杂度：O(n)，只需遍历一次数组。

空间复杂度：O(1)，仅使用常数空间。

```Python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        ans = 0
        left = 0
        right = len(height) - 1
        while left < right:
            area = (right - left) * min(height[left], height[right])
            ans = max(ans, area)
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return ans
```

# 42.接雨水

[Leetcode](https://leetcode.cn/problems/trapping-rain-water/description/)

## 前后缀分解法：

- 思路：

雨水积聚的高度不能超过左右两侧堤坝较低的那一侧，因为如果超过了，雨水就会从较低的那一侧流走。所以，这个位置能积聚雨水的最大高度就是左右两侧最高柱子高度中的较小值。而当前位置本身有柱子，柱子占据了一定的高度，所以要从这个最大可积聚高度中减去当前位置柱子的高度，剩下的就是该位置能接住的雨水量。

每个位置的雨水量其实是独立的，只与它周围的柱子高度有关。

- 代码思路：

前缀最大值：从左到右遍历数组，记录每个位置左侧的最大高度。

后缀最大值：从右到左遍历数组，记录每个位置右侧的最大高度。

计算雨水：对于每个位置，雨水的量等于其左侧最大值和右侧最大值中的较小值减去当前高度。

累加结果：将所有位置的雨水量累加，得到最终结果。

- 复杂度分析：

时间复杂度：O(n)，需要遍历数组三次。

空间复杂度：O(n)，需要两个辅助数组存储前缀最大值和后缀最大值。

```Python
class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        pre_max = [0] * n
        pre_max[0] = height[0]
        for i in range(1, n):
            pre_max[i] = max(pre_max[i-1], height[i])
        
        suf_max = [0] * n
        suf_max[-1] = height[-1]
        for i in range(n-2, -1, -1):
            suf_max[i] = max(suf_max[i+1], height[i])
        
        ans = 0
        for h, pre, suf in zip(height, pre_max, suf_max):
            ans += min(pre,suf) - h
        return ans
```
