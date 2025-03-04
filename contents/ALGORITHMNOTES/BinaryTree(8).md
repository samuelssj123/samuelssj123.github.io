ist:  669. 修剪二叉搜索树，108.将有序数组转换为二叉搜索树，538.把二叉搜索树转换为累加树，538.把二叉搜索树转换为累加树

[669. 修剪二叉搜索树trim-a-binary-search-tree](#01)，[](#02)，[](#03)，[](#04)

# <span id="01">669. 修剪二叉搜索树trim-a-binary-search-tree</span>

[Leetcode](https://leetcode.cn/problems/trim-a-binary-search-tree/description/) 

[Learning Materials](https://programmercarl.com/0669.%E4%BF%AE%E5%89%AA%E4%BA%8C%E5%8F%89%E6%90%9C%E7%B4%A2%E6%A0%91.html)

![image](../images/669-trim-a-binary-search-tree.png)

```python
class TreeNode:
    def __init__(self, val, left = None, right = None):
        self.val = val
        self.left = left
        self.right = right
```

## 递归法：

因为二叉搜索树的有序性，不需要使用栈模拟递归的过程。

在剪枝的时候，可以分为三步：

将root移动到[L, R] 范围内，注意是左闭右闭区间

剪枝左子树

剪枝右子树

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        if not root:
            return 
        if root.val < low:
            right = self.trimBST(root.right, low, high)
            return right
        if root.val > high:
            left = self.trimBST(root.left, low, high)
            return left
        root.left = self.trimBST(root.left, low, high)
        root.right = self.trimBST(root.right, low, high)
        return root
```

## 迭代法：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def trimBST(self, root: Optional[TreeNode], low: int, high: int) -> Optional[TreeNode]:
        if not root:
            return 
        
        # 处理头结点，让root移动到[L, R] 范围内，注意是左闭右闭
        while root and (root.val < low or root.val > high): #不在区间就继续找
            if root.val < low:
                root = root.right # 二叉搜索有序，小于L往右走
            else:
                root = root.left # 二叉搜索有序，大于R往左走
        
        cur = root
        #此时root已经在[L, R] 范围内，处理左孩子元素小于L的情况
        while cur:
            while cur.left and cur.left.val < low:
                cur.left = cur.left.right  #确保把当前节点左子树中所有值小于 low 的节点都修剪掉
            cur = cur.left   # 检查新的左子树

        cur = root # 回退
        #此时root已经在[L, R] 范围内，处理右孩子元素大于L的情况
        while cur:
            while cur.right and cur.right.val > high:
                cur.right = cur.right.left
            cur = cur.right

        return root

        
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

