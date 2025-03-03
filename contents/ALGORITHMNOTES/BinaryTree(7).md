List: 235. 二叉搜索树的最近公共祖先，701.二叉搜索树中的插入操作，450.删除二叉搜索树中的节点

[235. 二叉搜索树的最近公共祖先lowest-common-ancestor-of-a-binary-search-tree](#01)，[701.二叉搜索树中的插入操作insert-into-a-binary-search-tree](#02)，[450.删除二叉搜索树中的节点trim-a-binary-search-tree](#03)

# <span id="01">235. 二叉搜索树的最近公共祖先lowest-common-ancestor-of-a-binary-search-tree</span>

[Leetcode](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/description/) 

[Learning Materials](https://programmercarl.com/0235.%E4%BA%8C%E5%8F%89%E6%90%9C%E7%B4%A2%E6%A0%91%E7%9A%84%E6%9C%80%E8%BF%91%E5%85%AC%E5%85%B1%E7%A5%96%E5%85%88.html)

![image](../images/235-lowest-common-ancestor-of-a-binary-search-tree.png)

## 递归法：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def traversal(cur, p, q):
            if not cur:
                return
            if cur.val > p.val and cur.val > q.val:
                left = traversal(cur.left, p, q) 
                if left:
                    return left
            if cur.val < p.val and cur.val < q.val:
                right = traversal(cur.right, p, q)
                if right:
                    return right
            return cur
        return traversal(root, p, q)
```

## 迭代法：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        while root:
            if root.val > p.val and root.val > q.val:
                root = root.left
            elif root.val < p.val and root.val < q.val:
                root = root.right
            else:
                return root
        return None
```

# <span id="02">701.二叉搜索树中的插入操作insert-into-a-binary-search-tree</span>

[Leetcode](https://leetcode.cn/problems/insert-into-a-binary-search-tree/description/) 

[Learning Materials](https://programmercarl.com/0701.%E4%BA%8C%E5%8F%89%E6%90%9C%E7%B4%A2%E6%A0%91%E4%B8%AD%E7%9A%84%E6%8F%92%E5%85%A5%E6%93%8D%E4%BD%9C.html#%E7%AE%97%E6%B3%95%E5%85%AC%E5%BC%80%E8%AF%BE)

![image](../images/701-insert-into-a-binary-search-tree.png)

## 递归法：

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            node = TreeNode(val)
            return node
        if root.val > val:
            root.left = self.insertIntoBST(root.left, val)
        if root.val < val:
            root.right = self.insertIntoBST(root.right, val)
        return root
```

## 迭代法：

在迭代法遍历的过程中，需要记录一下当前遍历的节点的父节点，这样才能做插入节点的操作。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            node = TreeNode(val)
            return node
        cur = root
        parent = root
        while cur: # 需要记录上一个节点的值，否则无法赋值新节点
            parent = cur
            if cur.val > val:
                cur = cur.left
            else:
                cur = cur.right
        node = TreeNode(val)
        if val < parent.val:  # 赋值
            parent.left = node
        else:
            parent.right = node
        return root
```


# <span id="03">450.删除二叉搜索树中的节点trim-a-binary-search-tree</span>

[Leetcode](https://leetcode.cn/problems/trim-a-binary-search-tree/) 

[Learning Materials](https://programmercarl.com/0450.%E5%88%A0%E9%99%A4%E4%BA%8C%E5%8F%89%E6%90%9C%E7%B4%A2%E6%A0%91%E4%B8%AD%E7%9A%84%E8%8A%82%E7%82%B9.html#%E7%AE%97%E6%B3%95%E5%85%AC%E5%BC%80%E8%AF%BE)

![image](../images/450-trim-a-binary-search-tree.png)


