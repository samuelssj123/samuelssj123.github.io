List: 226.翻转二叉树，101. 对称二叉树，104.二叉树的最大深度，111.二叉树的最小深度

[226.翻转二叉树invert-binary-tree](#01)，[](#02)，[](#03)，[](#04),[](#05)

# <span id="01">226.翻转二叉树invert-binary-tree</span>

[Leetcode](https://leetcode.cn/problems/invert-binary-tree/description/)

[Learning Materials](https://programmercarl.com/0226.%E7%BF%BB%E8%BD%AC%E4%BA%8C%E5%8F%89%E6%A0%91.html#%E7%AE%97%E6%B3%95%E5%85%AC%E5%BC%80%E8%AF%BE)

![image](../images/.png)

## 前序遍历：递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
```

## 后序遍历：递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        self.invertTree(root.left)
        self.invertTree(root.right)
        root.left, root.right = root.right, root.left
        return root
```

## 中序遍历：递归

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        self.invertTree(root.left)
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        return root
```

## 前序遍历：迭代

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return 
        st = [root]
        while st:
            node = st.pop()
            node.left, node.right = node.right, node.left
            if node.right:
                st.append(node.right)
            if node.left:
                st.append(node.left)
        return root
```

## 伪后序遍历：迭代（实际上它是前序遍历，只不过把中间节点处理逻辑放到了最后）

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return 
        st = [root]
        while st:
            node = st.pop()
            if node.right:
                st.append(node.right)
            if node.left:
                st.append(node.left)
            node.left, node.right = node.right, node.left
        return root
```

## 伪中序遍历：迭代（实际上它是前序遍历，只不过把中间节点处理逻辑放到了中间。）

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return 
        st = [root]
        while st:
            node = st.pop()
            if node.right:
                st.append(node.right)
            node.left, node.right = node.right, node.left
            if node.right:
                st.append(node.right)
        return root
```

## 层序遍历：迭代

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return 
        que = deque([root])
        while que:
            node = que.popleft()
            node.left, node.right = node.right, node.left
            if node.left:
                que.append(node.left)
            if node.right:
                que.append(node.right)
        return root
```

# <span id="02">理论基础</span>

[Leetcode]() [Learning Materials]()

![image](../images/.png)

# <span id="03">理论基础</span>

[Leetcode]() [Learning Materials]()

![image](../images/.png)

# <span id="04">理论基础</span>

[Leetcode]() [Learning Materials]()

![image](../images/.png)

# <span id="05">理论基础</span>

[Leetcode]() [Learning Materials]()

![image](../images/.png)
