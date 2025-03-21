# 数组理论基础

文章链接：[数组理论基础](https://programmercarl.com/%E6%95%B0%E7%BB%84%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.html)  

数组：存放在**连续内存空间**上的**相同类型数据**的**集合**，可以方便的通过**下标索引**的方式获取到**下标对应的数据**。  

特点：数组下标都是**从0开始**的。数组内存空间的地址是**连续**的。数组的元素是不能删的，**只能覆盖**。  

# 704.二分查找

相关链接：[力扣原题](https://leetcode.cn/problems/binary-search/) [文章解读](https://programmercarl.com/0704.%E4%BA%8C%E5%88%86%E6%9F%A5%E6%89%BE.html) [视频解读](https://www.bilibili.com/video/BV1fA4y1o715)  

*易错点*：1. left<right? left<=right? 2. right=middle? right=middle-1?  

题目的前提是数组为有序数组，同时题目还强调数组中无重复元素，因为一旦有重复元素，使用二分查找法返回的元素下标可能不是唯一的，这些都是使用二分法的前提条件。  

*错误点*：<ins>在计算中间索引 middle 时，使用了普通除法 /，导致 middle 是一个浮点数，而 Python 的列表索引必须是整数或切片，不能是浮点数。将普通除法 / 改为整数除法 //，确保 middle 是整数。</ins>  

*循环不变量*：[left,right] or [left,right)  

- 左闭右闭：rihgt最初是数组末位下标，应为**numsize-1**。从两个易错点看起，left<right? or left<=right?，考虑【2，2】这个区间，是合法区间，只包含2这个数字，所以应该选择**小于等于**；  right=middle? or right=middle-1? 循环第一次，right是末位数，循环第二次，right应该是中间的数的前一个，因为中间的数已经被判断，而区间是左闭右闭，是middle-1。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left=0
        right=len(nums)-1
        while left<=right:
            middle=(left+right)//2
            if target<nums[middle]:
                right=middle-1
            elif target>nums[middle]:
                left=middle+1
            elif target==nums[middle]:
                return middle
        return -1
```        

- 左闭右开：rihgt最初是数组末位下标，应为**numsize**。从两个易错点看起，left<right? or left<=right?，考虑【2，2）这个区间，不是合法区间，所以应该选择**小于**；  right=middle? or right=middle-1? 循环第一次，right是末位数，循环第二次，right应该是中间的数的前一个，因为中间的数已经被判断，而区间是左闭右开，是middle。
  
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left=0
        right=len(nums)
        while left<right:
            middle=(left+right)//2
            if target<nums[middle]:
                right=middle
            elif target>nums[middle]:
                left=middle+1
            elif target==nums[middle]:
                return middle
        return -1
```

口诀：左闭右闭，右含右数；左闭右开，右不含右数。左不变，不重判。

# 27.移除元素

相关链接：[力扣原题](https://leetcode.cn/problems/remove-element/) [文章解读](https://programmercarl.com/0027.%E7%A7%BB%E9%99%A4%E5%85%83%E7%B4%A0.html) [视频解读](https://www.bilibili.com/video/BV12A4y1Z7LP)  

- 暴力解法：

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i = 0
        while i < len(nums):
            if nums[i] == val:
                for j in range(i, len(nums) - 1):
                    nums[j] = nums[j+1]
                nums.pop()
            else:
                i += 1
        return len(nums)
```

易错点：<ins>1.第一层循环中，由于列表长度随循环而更新，while 循环在每次迭代开始时都会重新计算循环条件表达式的值。而 for i in range(0, len(nums)) 这个循环的范围在开始时就已经确定好了，它不会随着列表长度的改变而动态调整。2.使用 nums.pop() 移除最后一个重复的元素。<ins>

- 双指针法：

```python
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        slow, fast = 0, 0
        for fast in range(0, len(nums)):
            if nums[fast] != val :
                nums[slow] = nums[fast]
                slow += 1
                fast += 1
            else :
                fast += 1
        return slow
```

核心思想：快指针-寻找新数组的元素 ，新数组就是不含有目标元素的数组；慢指针-指向更新 新数组下标的位置

# 977.有序数组的平方 

相关链接：[力扣原题](https://leetcode.cn/problems/squares-of-a-sorted-array/description/) [文章解读](https://programmercarl.com/0977.%E6%9C%89%E5%BA%8F%E6%95%B0%E7%BB%84%E7%9A%84%E5%B9%B3%E6%96%B9.html) [视频解读](https://www.bilibili.com/video/BV1QB4y1D7ep)

- 双指针法：

```python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        k = len(nums) - 1
        i = 0
        j = len(nums) - 1
        result = [0] * len(nums)
        while i <= j :
            if nums[i]*nums[i] <= nums[j]*nums[j] :
                result[k] = nums[j]*nums[j]
                k -= 1
                j -= 1
            else :
                result[k] = nums[i]*nums[i]
                k -= 1
                i += 1
        return result
```
思路：数组有序， 但负数平方之后可能成为最大数。那么数组平方的最大值就在数组的两端，不是最左就是最右，不可能是中间。考虑双指针法了，i指向起始位置，j指向终止位置。  
<ins>为什么while i <= j？ 如果i<j，那么当i与j相等时，就退出循环了，有一个数没有被判断。<ins>

总结：多变量区间开闭判合法，双指针快慢前后定去留。
