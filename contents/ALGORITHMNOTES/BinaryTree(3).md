List: 110.平衡二叉树，257. 二叉树的所有路径，404.左叶子之和，222.完全二叉树的节点个数

[110.平衡二叉树balanced-binary-tree](#01)，[](#02)，[](#03)，[](#04),[](#05)

# <span id="01">110.平衡二叉树balanced-binary-tree</span>

[Leetcode](https://leetcode.cn/problems/balanced-binary-tree/description/) 

[Learning Materials](https://programmercarl.com/0110.%E5%B9%B3%E8%A1%A1%E4%BA%8C%E5%8F%89%E6%A0%91.html)

![image](../images/110-balanced-binary-tree.png)

## 递归法：后序求高度

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        return self.getheight(root) != -1
    def getheight(self, root): 
        if not root:
            return 0
        leftheight = self.getheight(root.left)
        if leftheight == -1:
            return -1
        rightheight = self.getheight(root.right)
        if rightheight == -1:
            return -1
        if abs(rightheight - leftheight) > 1:
            return -1
        else:
            return 1 + max(leftheight, rightheight)
```

## 迭代法：栈模拟、专门求高度

在求二叉树最大深度时可以使用层序遍历求深度，但是不能用层次遍历求高度，深度、高度不是相反的关系，每个节点的深度、高度可能不对称。

通过栈模拟的后序遍历可以求每一个节点的高度，其实是通过求传入节点为根节点的最大深度来求的高度。


### 思路详细分析

大体思路：通过不断计算每个节点的左右子树高度，并比较它们的高度差，一旦发现高度差超过 1，就可以判定该二叉树不是平衡二叉树。

判断平衡：后序遍历入栈顺序应该是中、右、左。每次弹出一个节点，判断左右高度差是否符合条件。

计算每棵子树求高度：这里用了遍历的空指针法，每次从栈中弹出一个节点 node，如果 node 不为空，说明是真实的节点，不是标记。 深度 depth 加 1，表示进入了下一层。

如果弹出的是 None，说明已经处理完当前节点的左右子树，可以计算当前节点的高度了。 弹出栈顶的真实节点 node，深度 depth 减 1，表示回到上一层。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        st = []
        if not root:
            return True
        st.append(root)
        while st:
            node = st.pop()
            if (abs(self.getheight(node.right) - self.getheight(node.left)) > 1):
                return False
            if node.right:
                st.append(node.right)
            if node.left:
                st.append(node.left)
        return True
        
    def getheight(self, cur):
        st = []
        depth = 0
        result = 0
        if cur:
            st.append(cur)
        while st:
            node = st.pop()
            if node:
                st.append(node)
                st.append(None) # 中
                depth += 1
                if node.right:
                    st.append(node.right) #右
                if node.left:
                    st.append(node.left) #左
            else:
                node = st.pop()
                depth -= 1
            result = max(result, depth)
        return result
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
