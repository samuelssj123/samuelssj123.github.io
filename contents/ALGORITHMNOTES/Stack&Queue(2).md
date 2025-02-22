List:150. 逆波兰表达式求值，239. 滑动窗口最大值，347.前 K 个高频元素，栈和队列总结

[150. 逆波兰表达式求值evaluate-reverse-polish-notation](#01)，[239. 滑动窗口最大值sliding-window-maximum](#02)，[347.前 K 个高频元素top-k-frequent-elements](#03)，[栈和队列总结](#04)
![image](../images/232-implement-queue-using-stacks.png)

# <span id="01">150. 逆波兰表达式求值evaluate-reverse-polish-notation</span>

[Leetcode](https://leetcode.cn/problems/evaluate-reverse-polish-notation/description/) [Learning Materials](https://programmercarl.com/0150.%E9%80%86%E6%B3%A2%E5%85%B0%E8%A1%A8%E8%BE%BE%E5%BC%8F%E6%B1%82%E5%80%BC.html)

```Python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for char in tokens:
            if char == '+' or char == '-' or char == '*' or char == '/':
                nums2 = stack.pop()
                nums1 = stack.pop()
                if char == '+':
                    stack.append(nums1 + nums2)
                if char == '-':
                    stack.append(nums1 - nums2)
                if char == '*':
                    stack.append(nums1 * nums2)
                if char == '/':
                    stack.append(int(nums1 / nums2))  #两个整数之间的除法总是 向零截断 
            else:
                stack.append(int(char))  #若不将 "3" 和 "2" 转换为整数，在执行加法运算时，Python 会将它们视为字符串进行拼接操作。
        return stack[0]
```

# <span id="02">239. 滑动窗口最大值sliding-window-maximum</span>

[Leetcode](https://leetcode.cn/problems/sliding-window-maximum/description/) [Learning Materials](https://programmercarl.com/0239.%E6%BB%91%E5%8A%A8%E7%AA%97%E5%8F%A3%E6%9C%80%E5%A4%A7%E5%80%BC.html)

```Python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        self.res = deque() #使用双端队列，允许在队列的两端（队首和队尾）进行元素的插入和移除操作
        result = []
        for i in range(k): #先将前k的元素放进队列
            self.push(nums[i])  #定义了自己的 push 和 pop 方法，应该使用 self.push 和 self.pop 来调用这些自定义方法
        result.append(self.getmax()) #result 记录前k的元素的最大值

        for i in range(k, len(nums)):
            self.pop(nums[i - k]) #滑动窗口移除最前面元素
            self.push(nums[i]) #滑动窗口前加入最后面的元素
            result.append(self.getmax()) #记录对应的最大值
        return result

    def pop(self, value):
        if self.res and self.res[0] == value:  #每次弹出的时候，比较当前要弹出的数值是否等于队列出口元素的数值，如果相等则弹出。
            self.res.popleft()

    def push(self, value):  #如果push的数值大于入口元素的数值，那么就将队列后端的数值弹出，直到push的数值小于等于队列入口元素的数值为止。
        while self.res and self.res[-1] < value:  #使用 while 循环，不断移除队尾小于当前值的元素，以保证队列中的元素始终是单调递减的。这样在每次获取最大值时，队首元素就是当前窗口的最大值。
            self.res.pop()
        self.res.append(value)

    def getmax(self) : #查询当前队列里的最大值 直接返回队列前端也就是front就可以了。
        return self.res[0] #用 self.res[0] 来访问队首元素
```

# <span id="03">347.前 K 个高频元素top-k-frequent-elements</span>

[Leetcode](https://leetcode.cn/problems/top-k-frequent-elements/description/) [Learning Materials](https://programmercarl.com/0347.%E5%89%8DK%E4%B8%AA%E9%AB%98%E9%A2%91%E5%85%83%E7%B4%A0.html)

# <span id="04">栈和队列总结</span>
 
[Learning Materials](https://programmercarl.com/%E6%A0%88%E4%B8%8E%E9%98%9F%E5%88%97%E6%80%BB%E7%BB%93.html)
