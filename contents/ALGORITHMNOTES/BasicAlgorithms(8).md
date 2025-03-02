# 删除链表系列

# 力扣 237 题：删除链表中的节点

[leetcode](https://leetcode.cn/problems/delete-node-in-a-linked-list/description/)

#### 题目描述
请编写一个函数，用于 删除单链表中某个特定节点 。在设计函数时需要注意，你无法访问链表的头节点 `head` ，只能直接访问 要被删除的节点 。题目数据保证需要删除的节点 不是末尾节点 。

#### 解题思路
通常情况下，删除链表中的一个节点，我们需要找到该节点的前一个节点，然后将前一个节点的 `next` 指针指向要删除节点的下一个节点，从而跳过要删除的节点。但本题的特殊之处在于，我们无法访问链表的头节点，只能直接访问要被删除的节点。

由于要删除的节点不是末尾节点，所以它一定存在下一个节点。我们可以采用一种巧妙的方法：将下一个节点的值复制到当前要删除的节点，然后删除下一个节点。具体步骤如下：
1. 将当前节点 `node` 的值替换为它下一个节点的值。
2. 将当前节点 `node` 的 `next` 指针指向它下下一个节点，相当于跳过了原本下一个节点，从而达到删除的效果。

#### 代码实现
```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next
```

#### 代码详细解释
```python
node.val = node.next.val
```
这行代码将当前节点 `node` 的值替换为它下一个节点的值。因为我们不能直接删除当前节点，所以通过这种方式，让当前节点的值变成下一个节点的值，相当于把要删除的节点信息覆盖掉。

```python
node.next = node.next.next
```
这行代码将当前节点 `node` 的 `next` 指针指向它下下一个节点。这样一来，原本下一个节点就被跳过了，从链表中移除，实现了删除的效果。

#### 复杂度分析
- **时间复杂度**：$O(1)$。只进行了常数级的操作，即复制节点值和修改指针，不随链表长度的增加而增加操作次数。
- **空间复杂度**：$O(1)$。只使用了常数级的额外空间，没有使用与链表长度相关的额外数据结构。

综上所述，通过将下一个节点的值复制到当前节点，并跳过下一个节点，我们可以在不访问链表头节点的情况下，高效地删除指定的节点。 


# 力扣19.删除链表的倒数第N个节点

[leetcode](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

本题和[链表第二节](https://samuelssj123.github.io/contents/ALGORITHMNOTES/Linked_List(2).html)的题目内容是一样的。

# 力扣 83 题：删除排序链表中的重复元素

[Leetcode](https://leetcode.cn/problems/remove-duplicates-from-sorted-list/description/)

#### 题目描述
给定一个已排序的链表的头 `head` ， 删除所有重复的元素，使每个元素只出现一次 。返回 已排序的链表 。

#### 示例
- 输入：`head = [1,1,2]`
- 输出：`[1,2]`

#### 解题思路
本题的核心在于处理一个已排序的链表，移除其中所有重复的元素，使得每个元素仅出现一次。由于链表已经排序，重复的元素必然是相邻的，所以可以通过遍历链表，比较当前节点和下一个节点的值，若相同则跳过下一个节点，若不同则继续向后遍历。

#### 代码实现
```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 如果链表为空，直接返回该链表
        if not head:
            return head
        # 初始化当前节点为头节点
        cur = head
        # 当当前节点的下一个节点存在时，继续遍历
        while cur.next:
            # 如果当前节点的值和下一个节点的值相同
            if cur.next.val == cur.val:
                # 跳过下一个节点，即删除重复节点
                cur.next = cur.next.next
            else:
                # 若值不同，将当前节点移动到下一个节点
                cur = cur.next
        # 返回处理后的链表头节点
        return head
```

#### 代码详细解释
1. **边界条件处理**：
```python
if not head:
    return head
```
如果输入的链表为空，直接返回该链表，因为空链表不存在重复元素，无需处理。

2. **初始化当前节点**：
```python
cur = head
```
将当前节点 `cur` 初始化为链表的头节点，从链表头部开始遍历。

3. **遍历链表并删除重复元素**：
```python
while cur.next:
    if cur.next.val == cur.val:
        cur.next = cur.next.next
    else:
        cur = cur.next
```
- 使用 `while` 循环遍历链表，只要当前节点 `cur` 的下一个节点存在，就继续循环。
- 比较当前节点 `cur` 的值和下一个节点 `cur.next` 的值：
  - 如果它们的值相同，说明存在重复元素，将当前节点的 `next` 指针指向 `cur.next.next`，即跳过下一个节点，实现删除重复节点的目的。
  - 如果它们的值不同，将当前节点 `cur` 移动到下一个节点，继续向后遍历。

4. **返回处理后的链表**：
```python
return head
```
遍历结束后，链表中所有重复元素都已被删除，返回链表的头节点 `head`。

#### 复杂度分析
- **时间复杂度**：$O(n)$，其中 $n$ 是链表的长度。因为只需要遍历一次链表，每个节点最多被访问一次。
- **空间复杂度**：$O(1)$，只使用了常数级的额外空间，只需要几个指针变量来辅助遍历链表。

综上所述，通过上述的遍历和指针操作，我们可以高效地删除排序链表中的重复元素。 

# 力扣 82 题：删除排序链表中的重复元素 II 题解

[leetcode](https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/)

#### 题目描述
给定一个已排序的链表的头 `head` ， 删除原始链表中所有重复数字的节点，只留下不同的数字 。返回 已排序的链表 。

#### 示例
- 输入：`head = [1,2,3,3,4,4,5]`
- 输出：`[1,2,5]`

#### 解题思路
本题要求在已排序的链表中删除所有重复数字的节点，只保留出现次数为 1 的节点。与力扣 83 题不同，83 题只需要删除重复的元素，使每个元素仅出现一次，而本题需要将重复出现的元素全部删除。

为了方便处理头节点可能被删除的情况，我们创建一个虚拟头节点 `dummy_node`，让它的 `next` 指针指向原链表的头节点 `head`。然后使用一个指针 `cur` 从虚拟头节点开始遍历链表，通过两层循环来检查和删除重复节点。

#### 代码实现
```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 创建虚拟头节点，方便处理头节点可能被删除的情况
        dummy_node = ListNode(next = head)
        # 初始化当前节点为虚拟头节点
        cur = dummy_node
        # 当当前节点的下一个节点和下下一个节点都存在时，继续遍历
        while cur.next and cur.next.next:
            # 记录当前节点下一个节点的值
            val = cur.next.val
            # 如果下下一个节点的值和当前记录的值相同，说明存在重复元素
            if cur.next.next.val == val:
                # 内层循环，持续删除所有值为 val 的节点
                while cur.next and cur.next.val == val:
                    cur.next = cur.next.next
            else:
                # 如果不存在重复元素，将当前节点向后移动一位
                cur = cur.next
        # 返回处理后的链表头节点（虚拟头节点的下一个节点）
        return dummy_node.next
```

#### 代码详细解释
1. **创建虚拟头节点**：
```python
dummy_node = ListNode(next = head)
cur = dummy_node
```
创建一个虚拟头节点 `dummy_node`，并将其 `next` 指针指向原链表的头节点 `head`。然后将当前节点 `cur` 初始化为虚拟头节点，这样可以避免在处理头节点可能被删除的情况时出现复杂的边界条件判断。

2. **外层循环遍历链表**：
```python
while cur.next and cur.next.next:
    val = cur.next.val
```
使用 `while` 循环遍历链表，只要当前节点 `cur` 的下一个节点和下下一个节点都存在，就继续循环。记录当前节点下一个节点的值 `val`，用于后续比较。

3. **检查并删除重复节点**：
```python
if cur.next.next.val == val:
    while cur.next and cur.next.val == val:
        cur.next = cur.next.next
else:
    cur = cur.next
```
- **存在重复元素的情况**：如果下下一个节点的值和 `val` 相同，说明存在重复元素。使用内层 `while` 循环，持续删除所有值为 `val` 的节点，直到遇到值不同的节点为止。
- **不存在重复元素的情况**：如果下下一个节点的值和 `val` 不同，说明当前节点的下一个节点不存在重复，将当前节点 `cur` 向后移动一位。

4. **返回处理后的链表**：
```python
return dummy_node.next
```
遍历结束后，链表中所有重复元素的节点都已被删除，返回虚拟头节点的下一个节点，即处理后的链表头节点。

#### 复杂度分析
- **时间复杂度**：$O(n)$，其中 $n$ 是链表的长度。因为只需要遍历一次链表，每个节点最多被访问两次（一次在外层循环，一次在内层循环）。
- **空间复杂度**：$O(1)$，只使用了常数级的额外空间，主要是几个指针变量和虚拟头节点。

综上所述，通过使用虚拟头节点和两层循环，我们可以高效地删除排序链表中所有重复数字的节点。 

