# 34.在排序数组中查找元素的第一个和最后一个位置

[Leetcode](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/description/)

```Python
class Solution:
    # lower_bound 返回最小的满足 nums[i] >= target 的下标 i
    # 如果数组为空，或者所有数都 < target，则返回 len(nums)
    # 要求 nums 是非递减的，即 nums[i] <= nums[i + 1]
    def lower_bound1(self, nums: List[int], target:int) -> int: # 闭区间[left, right]
        left, right = 0, len(nums) - 1 
        while left <= right: # 区间不为空
         # 循环不变量：nums[left-1] < target , nums[right+1] >= target
            mid = (left + right) // 2
            if nums[mid] >= target:
                right = mid - 1 # 范围缩小到 [left, mid-1]
            else :
                left = mid + 1 # 范围缩小到 [mid + 1, right]
        return left 
# 循环结束后 left = right+1：随着循环的进行，left 指针会不断右移，right 指针会不断左移。最终，当 left 和 right 相遇（即 left == right）时，还会进行一次判断。如果 nums[mid] >= target，则 right = mid - 1，此时 left 不变，right 减 1，使得 left = right + 1；如果 nums[mid] < target，则 left = mid + 1，此时 right 不变，left 加 1，同样使得 left = right + 1。所以循环结束后，一定有 left = right + 1。

# 此时 nums[left-1] < target 而 nums[left] = nums[right+1] >= target, 当 nums[mid] < target 时，我们会将 left 更新为 mid + 1，所以 left 左侧的元素必然小于 target；当 nums[mid] >= target 时，我们会将 right 更新为 mid - 1，所以 right 右侧的元素必然大于等于 target

# 所以 left 就是第一个 >= target 的元素下标
    
    def lower_bound2(self, nums: List[int], target:int) -> int: # 左闭右开区间[left, right)
        left, right = 0, len(nums)  
        while left < right: # 区间不为空
         # 循环不变量：nums[left-1] < target , nums[right] >= target
            mid = (left + right) // 2
            if nums[mid] >= target:
                right = mid  # 范围缩小到 [left, mid)
            else :
                left = mid + 1 # 范围缩小到 [mid + 1, right)
        return left #循环结束后 left = right, 而nums[left-1] < target , nums[right] >= target，left是第一个下标

    def lower_bound3(self, nums: List[int], target:int) -> int: # 开区间(left, right)
        left, right = -1, len(nums)  
        while left + 1 < right: # 区间不为空
         # 循环不变量：nums[left] < target , nums[right] >= target
            mid = (left + right) // 2
            if nums[mid] >= target:
                right = mid  # 范围缩小到 (left, mid)
            else :
                left = mid  # 范围缩小到 (mid, right)
        return right #循环结束后 left + 1 = right, 而nums[left-1] < target , nums[right] >= target，right是第一个下标

    def lower_bound4(self, nums: List[int], target:int) -> int: # 左开右闭区间(left, right]
        left, right = -1, len(nums) - 1
        while left < right: # 区间不为空
         # 循环不变量：nums[left] < target , nums[right+1] >= target
            mid = left + (right - left) // 2 + 1 #left 开始为 -1，需要向右偏移，也就是向上取整
            if nums[mid] >= target:
                right = mid - 1  # 范围缩小到 (left, mid - 1]
            else :
                left = mid   # 范围缩小到 (mid , right]
        return right + 1 #循环结束后 left = right, 而nums[left] < target , nums[right+1] >= target，right是第一个下标


    def searchRange(self, nums: List[int], target: int) -> List[int]:
        start = self.lower_bound4(nums, target)
        if start == len(nums) or nums[start] != target:
            return [-1, -1]
        end = self.lower_bound4(nums, target + 1) - 1
        return [start, end]
```

- 左开右闭中，需要更新mid偏向右侧的原因：

假设我们使用 mid = (left + right) // 2。
当 left = -1，right = 0 时，如前面所说 mid = -1。
若 nums[mid] < target（假设 nums[0] >= target），则 left = mid = -1，此时 left 没有改变。
下一次循环时，left 还是 -1，right 还是 0，mid 还是 -1，如此循环下去，left 和 right 的值不会发生有效的变化，就会陷入死循环，无法找到正确的结果。
而使用 mid = left + (right - left) // 2 + 1，当 left = -1，right = 0 时，mid = -1+(0 - (-1))//2 + 1 = 0，可以正确地更新区间，避免死循环。

- 神来之笔： end = self.lower_bound4(nums, target + 1) - 1
  
要想找到≤target的最后一个数，无需单独再写一个二分。我们可以先找到这个数的右边相邻数字，也就是>target的第一个数。在所有数都是整数的前提下，>target等价于≥target+1，这样就可以复用我们已经写好的二分函数了，即lowerBound(nums, target + 1)，算出这个数的下标后，将其减一，就得到≤target的最后一个数的下标。

- 总结：

| 函数名 | 区间类型 | 初始化 `left`, `right` | `while` 条件 | `mid` 计算 | `nums[mid] >= target` 时更新 | `nums[mid] < target` 时更新 | 返回值 | 循环结束特征 | 循环不变量 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lower_bound1` | 闭区间 `[left, right]` | `0`, `len(nums) - 1` | `left <= right` | `(left + right) // 2` | `right = mid - 1` | `left = mid + 1` | `left` | `left = right + 1` | `nums[left - 1] < target`, `nums[right + 1] >= target` |
| `lower_bound2` | 左闭右开区间 `[left, right)` | `0`, `len(nums)` | `left < right` | `(left + right) // 2` | `right = mid` | `left = mid + 1` | `left` | `left = right` | `nums[left - 1] < target`, `nums[right] >= target` |
| `lower_bound3` | 开区间 `(left, right)` | `-1`, `len(nums)` | `left + 1 < right` | `(left + right) // 2` | `right = mid` | `left = mid` | `right` | `left + 1 = right` | `nums[left] < target`, `nums[right] >= target` |
| `lower_bound4` | 左开右闭区间 `(left, right]` | `-1`, `len(nums) - 1` | `left < right` | `left + (right - left) // 2 + 1` | `right = mid - 1` | `left = mid` | `right + 1` | `left = right` | `nums[left] < target`, `nums[right + 1] >= target` |

- 如何判断自己写的哪种区间？

看while循环的条件，如果是left <= right，就是闭区间；如果是left < right，就是半闭半开区间；如果是left + 1 < right，就是开区间。

