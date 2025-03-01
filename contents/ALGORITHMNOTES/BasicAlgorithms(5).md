# 二分查找

# 力扣162题题解：寻找峰值

[Leetcode](https://leetcode.cn/problems/find-peak-element/)

## 一、题目描述
峰值元素是指其值严格大于左右相邻值的元素。给你一个整数数组 `nums`，找到峰值元素并返回其索引。数组可能包含多个峰值，在这种情况下，返回 任何一个峰值 所在位置即可。

你可以假设 `nums[-1] = nums[n] = -∞` 。也就是说，第一个元素如果大于第二个元素，那么第一个元素就是峰值；最后一个元素如果大于倒数第二个元素，那么最后一个元素也是峰值。

## 二、解题思路
本题采用二分查找的方法来寻找峰值元素。

1. 定义两个指针 `left` 和 `right`，初始时 `left = -1`，`right = len(nums) - 1`。这里将 `left` 初始化为 `-1` 是为了方便后续与题目中假设的 `nums[-1] = -∞` 相对应，使得第一个元素也能参与到二分查找的逻辑中。
2. 进入循环 `while left + 1 < right`，只要 `left` 和 `right` 之间的距离大于 `1`，就继续循环。
3. 在循环内部，计算中间索引 `mid = (left + right) // 2`。
4. 比较 `nums[mid]` 和 `nums[mid + 1]` 的大小：
    - 如果 `nums[mid] > nums[mid + 1]`，说明在 `mid` 右侧的元素是递减的趋势，那么峰值可能在 `mid` 及其左侧，所以将 `right` 更新为 `mid`。
    - 如果 `nums[mid] <= nums[mid + 1]`，说明在 `mid` 右侧的元素是递增的趋势，那么峰值可能在 `mid` 右侧，所以将 `left` 更新为 `mid`。
5. 当循环结束时，`left` 和 `right` 之间的距离为 `1`，此时 `right` 所指向的位置就是峰值元素的索引（因为我们的逻辑保证了最终 `right` 指向的元素比其左侧元素大，再结合题目中边界的假设，`right` 指向的就是峰值），返回 `right` 即可。

## 三、代码实现

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        #[0, n-2] -->开区间(-1, n-1)
        left = -1
        right = len(nums) - 1
        while left + 1 < right:
            mid = (left + right) // 2
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid
        return right
```

## 四、复杂度分析
1. **时间复杂度**：每次循环都会将搜索区间缩小一半，因此时间复杂度为 $O(\log n)$，其中 $n$ 是数组 `nums` 的长度。
2. **空间复杂度**：代码中只使用了常数个额外变量，所以空间复杂度为 $O(1)$。 




### 力扣 153 题：寻找旋转排序数组中的最小值题解

[Leetcode](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)

#### 题目描述
已知一个长度为 `n` 的数组，预先按照升序排列，经由 `1` 到 `n` 次 旋转 后，得到输入数组。例如，原数组 `nums = [0,1,2,4,5,6,7]` 在变化后可能得到：
 - 若旋转 `4` 次，则可以得到 `[4,5,6,7,0,1,2]`
 - 若旋转 `7` 次，则可以得到 `[0,1,2,4,5,6,7]`

注意，数组 `[a[0], a[1], a[2], ..., a[n-1]]` 旋转一次 的结果为数组 `[a[n-1], a[0], a[1], a[2], ..., a[n-2]]` 。

给你一个元素值 互不相同 的数组 `nums` ，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 最小元素 。

你必须设计一个时间复杂度为 $O(log n)$ 的算法解决此问题。

#### 解题思路
本题可使用二分查找算法来解决。二分查找的核心思想是通过不断将搜索区间缩小一半，来找到目标元素。在旋转排序数组中，我们可以利用数组的特性，通过比较中间元素和数组最后一个元素的大小关系，来判断最小值所在的区间。

具体思路如下：
1. 初始化两个指针 `left` 和 `right`，分别指向搜索区间的左右边界。这里将 `left` 初始化为 `-1`，`right` 初始化为数组的最后一个元素的索引，形成一个开区间 `(-1, n - 1)`。
2. 进入循环，只要 `left + 1 < right`，就继续进行二分查找。
3. 计算中间索引 `mid`，并比较 `nums[mid]` 和 `nums[-1]` 的大小：
    - 如果 `nums[mid] < nums[-1]`，说明从 `mid` 到数组末尾是有序的，且最小值在 `mid` 或其左侧，因此将 `right` 更新为 `mid`。
    - 如果 `nums[mid] >= nums[-1]`，说明从 `mid` 到数组末尾不是有序的，最小值在 `mid` 的右侧，因此将 `left` 更新为 `mid`。
4. 当循环结束时，`left` 和 `right` 相邻，此时 `right` 指向的就是数组中的最小值。

#### 代码实现
```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        #[0, n - 2] --> (-1, n - 1)
        left = -1
        right = len(nums) - 1
        while left + 1 < right:
            mid = (left + right) // 2
            if nums[mid] < nums[-1]:
                right = mid
            else:
                left = mid
        return nums[right]
```

#### 复杂度分析
- **时间复杂度**：由于每次循环都将搜索区间缩小一半，因此时间复杂度为 $O(log n)$，其中 $n$ 是数组的长度。
- **空间复杂度**：只使用了常数级的额外空间，因此空间复杂度为 $O(1)$。

#### 代码解释
- **指针初始化**：`left = -1` 和 `right = len(nums) - 1` 形成了一个开区间 `(-1, n - 1)`，方便后续的二分查找操作。
- **二分查找循环**：`while left + 1 < right` 确保在 `left` 和 `right` 相邻时停止循环。
- **中间元素比较**：`if nums[mid] < nums[-1]` 判断中间元素和数组最后一个元素的大小关系，从而确定最小值所在的区间。
- **返回结果**：循环结束后，`right` 指向的就是数组中的最小值，因此返回 `nums[right]`。

通过上述方法，我们可以在 $O(log n)$ 的时间复杂度内找到旋转排序数组中的最小值。

# 力扣 33 题：搜索旋转排序数组题解

[Leetcode](https://leetcode.cn/problems/search-in-rotated-sorted-array/description/)

#### 题目描述
整数数组 `nums` 按升序排列，数组中的值 互不相同 。在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 旋转，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 从 0 开始 计数）。例如， `[0,1,2,4,5,6,7]` 在下标 `3` 处经旋转后可能变为 `[4,5,6,7,0,1,2]` 。

给你 旋转后 的数组 `nums` 和一个整数 `target` ，如果 `nums` 中存在这个目标值 `target` ，则返回它的下标，否则返回 `-1` 。你必须设计一个时间复杂度为 $O(\log n)$ 的算法解决此问题。

#### 解题思路
本题的核心是在一个旋转排序数组中查找目标值 `target` 的下标，并且要求时间复杂度为 $O(\log n)$，因此可以使用二分查找算法。由于数组经过了旋转，其不再是完全有序的，但可以将数组分为两部分，每部分都是有序的。我们通过定义一个辅助函数 `isblue` 来判断中间元素与目标值的位置关系，从而缩小搜索范围。

#### 代码实现
```python
from typing import List

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        def isblue(i):
            end = nums[-1]
            if nums[i] > end:
                return target > end and nums[i] >= target
            else:
                return target > end or nums[i] >= target

        #[0, n - 2] --> (-1, n)
        left = -1
        right = len(nums) 
        while left + 1 < right:
            mid = (left + right) // 2
            if isblue(mid):
                right = mid
            else:
                left = mid
        if right == len(nums) or nums[right] != target:
            return -1
        return right
```

#### 代码详细解释

##### 辅助函数 `isblue`
```python
def isblue(i):
    end = nums[-1]
    if nums[i] > end:
        return target > end and nums[i] >= target
    else:
        return target > end or nums[i] >= target
```
- `end = nums[-1]`：获取数组的最后一个元素，用于判断中间元素 `nums[i]` 位于旋转数组的前半部分还是后半部分。
- **`if nums[i] > end`**：说明 `nums[i]` 位于旋转数组的前半部分（即较大的部分）。此时，如果 `target` 也在这个较大部分，并且 `nums[i]` 大于等于 `target`，则返回 `True`，表示 `i` 是我们要找的蓝色区域（这里的蓝色区域是自定义的一个概念，用于划分搜索范围）。
- **`else`**：说明 `nums[i]` 位于旋转数组的后半部分（即较小的部分）。此时，只要 `target` 在较大部分或者 `nums[i]` 大于等于 `target`，就返回 `True`。

##### 二分查找部分
```python
left = -1
right = len(nums) 
while left + 1 < right:
    mid = (left + right) // 2
    if isblue(mid):
        right = mid
    else:
        left = mid
```
- **初始化指针**：`left = -1` 和 `right = len(nums)` 形成一个开区间 `(-1, n)`，用于二分查找。
- **二分查找循环**：`while left + 1 < right` 确保在 `left` 和 `right` 相邻时停止循环。
- **中间元素判断**：计算中间索引 `mid`，调用 `isblue(mid)` 判断 `mid` 是否在蓝色区域。如果是，则将 `right` 更新为 `mid`，缩小搜索范围到左半部分；否则，将 `left` 更新为 `mid`，缩小搜索范围到右半部分。

##### 结果判断
```python
if right == len(nums) or nums[right] != target:
    return -1
return right
```
- 如果 `right` 等于数组的长度，说明没有找到目标值；或者 `nums[right]` 不等于 `target`，也说明没有找到目标值，此时返回 `-1`。
- 否则，返回 `right`，即目标值的下标。

#### 复杂度分析
- **时间复杂度**：由于每次循环都将搜索范围缩小一半，因此时间复杂度为 $O(\log n)$，其中 $n$ 是数组的长度。
- **空间复杂度**：只使用了常数级的额外空间，因此空间复杂度为 $O(1)$。

综上所述，通过上述的二分查找方法，我们可以在旋转排序数组中高效地查找目标值的下标。 
