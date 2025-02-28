List: 654.最大二叉树，617.合并二叉树，700.二叉搜索树中的搜索，98.验证二叉搜索树


[654.最大二叉树maximum-binary-tree](#01)，[](#02)，[](#03)，[](#04),[](#05)

# <span id="01">654.最大二叉树maximum-binary-tree</span>

[Leetcode](https://leetcode.cn/problems/maximum-binary-tree/description/) 

[Learning Materials](https://programmercarl.com/0654.%E6%9C%80%E5%A4%A7%E4%BA%8C%E5%8F%89%E6%A0%91.html)

![image](../images/654-maximum-binary-tree.png)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        if len(nums) == 1:
            return TreeNode(nums[0])
        maxvalue, index = 0, 0
        for i in range(len(nums)):
            if maxvalue < nums[i]:
                maxvalue = nums[i]
                index = i 
        node = TreeNode(maxvalue)
        if index > 0:
            lnums = nums[:index]
            node.left = self.constructMaximumBinaryTree(lnums)
        if index < len(nums) - 1:
            rnums = nums[index + 1:]
            node.right = self.constructMaximumBinaryTree(rnums)
        return node
```

## 优化：不使用新数组，直接用下标：

允许空节点进入递归，所以不用在递归的时候加判断节点是否为空。终止条件也要有相应的改变。

类似用数组构造二叉树的题目，每次分隔尽量不要定义新的数组，而是通过下标索引直接在原数组上操作，这样可以节约时间和空间上的开销。

**要不要加if？如果让空节点（空指针）进入递归，就不加if，如果不让空节点进入递归，就加if限制一下， 终止条件也会相应的调整。**

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def constructMaximumBinaryTree(self, nums: List[int]) -> Optional[TreeNode]:
        return self.construct(nums, 0, len(nums))
    def construct(self, nums, left, right):
        #在左闭右开区间[left, right)，构造二叉树
        if left >= right: #左闭右开区间，相等时为空
            return
        # 分割点下标：
        index = left
        for i in range(left, right): #只需要在 [left, right) 这个区间内找到最大值
            if nums[index] < nums[i]:
                index = i 
        node = TreeNode(nums[index])
        # 左闭右开：[left, maxValueIndex)
        node.left = self.construct(nums, left, index)
        # 左闭右开：[maxValueIndex + 1, right)
        node.right = self.construct(nums, index + 1, right)
        return node
```

# <span id="02">理论基础</span>

[Leetcode]() 

[Learning Materials]()

![image](../images/.png)

# <span id="03">理论基础</span>

[Leetcode]() 

[Learning Materials]()

![image](../images/.png)

# <span id="04">理论基础</span>

[Leetcode]() 

[Learning Materials]()

![image](../images/.png)

# <span id="05">理论基础</span>

[Leetcode]() 

[Learning Materials]()

![image](../images/.png)
