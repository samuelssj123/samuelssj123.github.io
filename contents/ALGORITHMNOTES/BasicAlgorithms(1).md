# 167. 两数之和Ⅱ

[力扣](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/description/)

数组是排好序的，选最小和最大的数，挨着往中间尝试。

- 思路：

本题利用双指针技巧高效地找到两个数的和等于目标值。由于数组已按非递减顺序排列，可以通过调整左右指针来逼近目标值：

初始化指针：左指针指向数组起始位置，右指针指向数组末尾。

循环调整指针：计算两指针所指元素之和。若等于目标值，直接返回下标（注意题目要求下标从1开始）。若和小于目标值，左指针右移以增大和；若和大于目标值，右指针左移以减小和。

终止条件：当找到符合条件的两个数时跳出循环，返回结果。

- 时间复杂度：O(n)，每个元素最多被访问一次。空间复杂度：O(1)，仅使用常数空间。

- 关键点

双指针移动策略：通过比较当前和与目标值的大小，决定移动方向，确保不漏解。

有序数组特性：利用排序特性，避免暴力枚举，提升效率。

下标转换：返回结果时需将下标从0-based转换为1-based。

```Python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        #时间复杂度O（n）
        left = 0
        right = len(numbers) - 1
        while left < right:
            s = numbers[left] + numbers[right]
            if s == target:
                break
            if s < target:
                left += 1
            else:
                right -= 1
        return [left+1, right+1] #注意数组下标从0开始。
```

# 15.三数之和

[Leetcode](https://leetcode.cn/problems/3sum/)

三元组的顺序不重要。

不重复：当前枚举的数和上一个数一样，就跳过。

- 思路：

排序：首先将数组排序，方便后续操作。

遍历数组：固定一个数 nums[i]，然后在剩余部分使用双指针法寻找另外两个数 nums[j] 和 nums[k]，使得 nums[i] + nums[j] + nums[k] = 0。

去重：如果 nums[i] 与前一个数相同，跳过以避免重复。在双指针移动过程中，如果 nums[j] 或 nums[k] 与下一个数相同，跳过以避免重复。

剪枝优化：如果 nums[i] + nums[i+1] + nums[i+2] > 0，说明当前 nums[i] 已经过大，直接跳出循环。如果 nums[i] + nums[-2] + nums[-1] < 0，说明当前 nums[i] 过小，跳过当前循环。

- 关键点：
- 
排序：排序是双指针法的基础，确保数组有序后可以高效地调整指针。

双指针法：固定一个数后，用双指针在剩余部分寻找另外两个数，时间复杂度为 O(n)。

去重：通过跳过相同的数，避免重复的三元组。

剪枝优化：通过提前判断当前数是否过大或过小，减少不必要的计算。

- 复杂度分析：
  
时间复杂度：O(n²)，排序的时间复杂度为 O(n log n)，双指针部分为 O(n²)。

空间复杂度：O(1)，忽略存储结果的空间，仅使用常数空间。

```Python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans = []
        n = len(nums)
        for i in range(n - 2):
            x = nums[i]
            if i > 0 and x == nums[i - 1]:
                continue
            if x + nums[i + 1] + nums[i + 2] > 0: #从当前 i 开始往后的三个最小的数之和都大于 0，后续的组合肯定也都大于 0，所以可以提前结束外层循环
                break
            if x + nums[-2] + nums[-1] < 0: #当这个条件满足时，只能说明当前的 x 与数组中最大的两个数之和小于 0，但不能直接得出后续的所有组合都小于 0，因为后续可能会有更大的 x 使得组合的和为 0，所以这里不应该使用 break，而应该使用 continue 跳过当前的 x
                continue
            
            j = i + 1
            k = n - 1
            while j < k:
                s = x + nums[j] + nums[k]
                if s > 0:
                    k -= 1
                elif s < 0:
                    j += 1
                else:
                    ans.append([x, nums[j], nums[k]])
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:
                        j += 1
                    k -= 1
                    while j < k and nums[k] == nums[k + 1]:
                        k -= 1
        return ans
```
