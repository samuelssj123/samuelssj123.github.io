# 反转链表：

### 力扣 206 题：反转链表题解

[leetcode](https://leetcode.cn/problems/reverse-linked-list/description/)

#### 题目描述
给你单链表的头节点 `head` ，请你反转链表，并返回反转后的链表。

示例：
- 输入：`head = [1,2,3,4,5]`
- 输出：`[5,4,3,2,1]`

#### 解题思路
本题采用迭代的方法来反转链表。核心思路是遍历链表，在遍历过程中改变每个节点的指针方向，将原本指向下一个节点的指针改为指向前一个节点。为了实现这一操作，我们需要使用三个指针：`pre` 用于记录当前节点的前一个节点，`cur` 用于遍历链表的当前节点，`temp` 用于临时保存当前节点的下一个节点，防止在改变指针方向时丢失后续节点的信息。

#### 代码实现
```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre = None
        cur = head
        while cur:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        return pre
```

#### 代码详细解释

##### 初始化指针
```python
pre = None
cur = head
```
- `pre` 初始化为 `None`，表示当前反转后的链表初始为空。
- `cur` 初始化为链表的头节点 `head`，用于从链表的第一个节点开始遍历。

##### 遍历链表并反转指针
```python
while cur:
    temp = cur.next
    cur.next = pre
    pre = cur
    cur = temp
```
- **`temp = cur.next`**：使用 `temp` 临时保存当前节点 `cur` 的下一个节点，避免在后续改变 `cur.next` 时丢失后续节点的信息。
- **`cur.next = pre`**：将当前节点 `cur` 的指针方向反转，使其指向 `pre`，即前一个节点。
- **`pre = cur`**：将 `pre` 指针向后移动一位，指向当前节点 `cur`，为下一次反转做准备。
- **`cur = temp`**：将 `cur` 指针向后移动一位，指向之前保存的下一个节点 `temp`，继续遍历链表。

##### 返回结果
```python
return pre
```
当遍历完整个链表后，`cur` 会变为 `None`，此时 `pre` 指向反转后链表的头节点，因此返回 `pre`。

#### 复杂度分析
- **时间复杂度**：$O(n)$，其中 $n$ 是链表的长度。因为需要遍历链表中的每个节点一次。
- **空间复杂度**：$O(1)$，只使用了常数级的额外空间，只需要 `pre`、`cur` 和 `temp` 三个指针。

综上所述，通过迭代的方式，我们可以高效地反转单链表。 


# 力扣 92 题：反转链表 II 题解

[leetcode](https://leetcode.cn/problems/reverse-linked-list-ii/description/)

#### 题目描述
给你单链表的头指针 `head` 和两个整数 `left` 和 `right` ，其中 `left <= right` 。请你反转从位置 `left` 到位置 `right` 的链表节点，返回 反转后的链表 。

#### 解题思路
本题要求反转链表中从位置 `left` 到位置 `right` 的部分节点。为了方便处理，我们可以使用一个虚拟头节点 `dummy_node` ，它不存储实际数据，只是作为链表的起始点，这样可以避免处理头节点反转时的边界情况。

整体步骤如下：
1. 找到第 `left - 1` 个节点 `p0`，它是需要反转部分的前一个节点。（为了防止left是第一个，那么用哨兵）
2. 从第 `left` 个节点开始，使用迭代的方法反转 `right - left + 1` 个节点。
3. 将反转后的部分链表与原链表的剩余部分连接起来。（首尾都要拼起来）

#### 代码实现
```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        # 创建虚拟头节点
        dummy_node = ListNode(next = head)
        p0 = dummy_node
        # 找到第 left - 1 个节点
        for i in range(left - 1):
            p0 = p0.next

        pre = None
        cur = p0.next
        # 反转从 left 到 right 的节点
        for i in range(right - left + 1):
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        
        # 连接反转后的链表与原链表的剩余部分
        p0.next.next = cur
        p0.next = pre
        return dummy_node.next
```

#### 代码详细解释

##### 1. 创建虚拟头节点并找到 `p0`
```python
dummy_node = ListNode(next = head)
p0 = dummy_node
for i in range(left - 1):
    p0 = p0.next
```
- `dummy_node` 是虚拟头节点，它的 `next` 指针指向原链表的头节点 `head`。
- 通过 `for` 循环，将 `p0` 移动到第 `left - 1` 个节点的位置，`p0` 之后的节点就是需要反转的部分。

##### 2. 反转从 `left` 到 `right` 的节点
```python
pre = None
cur = p0.next
for i in range(right - left + 1):
    temp = cur.next
    cur.next = pre
    pre = cur
    cur = temp
```
这部分代码使用迭代的方法反转从 `left` 到 `right` 的节点，与反转整个链表的逻辑类似。反转完成后，`pre` 指向反转后的子链表的头节点，`cur` 指向第 `right + 1` 个节点，即反转部分之后的第一个节点。

##### 3. 连接反转后的链表与原链表的剩余部分
```python
p0.next.next = cur
p0.next = pre
```
这两行代码是本题的关键，下面详细解释：
- **`p0.next.next = cur`**：
  - `p0.next` 原本指向第 `left` 个节点，在反转过程中，第 `left` 个节点会成为反转后子链表的尾节点。
  - `cur` 指向第 `right + 1` 个节点，也就是反转部分之后的第一个节点。
  - 这行代码的作用是将反转后子链表的尾节点的 `next` 指针指向 `cur`，从而将反转后的子链表与原链表的剩余部分连接起来。

- **`p0.next = pre`**：
  - `pre` 指向反转后子链表的头节点。
  - 这行代码将 `p0` 的 `next` 指针指向 `pre`，使得原链表中第 `left - 1` 个节点（即 `p0`）与反转后的子链表连接起来。

##### 4. 返回结果
```python
return dummy_node.next
```
最后返回 `dummy_node.next`，即反转后的链表的头节点。

#### 复杂度分析
- **时间复杂度**：$O(n)$，其中 $n$ 是链表的长度。因为只需要遍历链表一次。
- **空间复杂度**：$O(1)$，只使用了常数级的额外空间。

综上所述，通过以上步骤，我们可以有效地反转链表中指定位置的部分节点。 

# 力扣 25 题：K 个一组翻转链表题解

[leetcode](https://leetcode.cn/problems/reverse-nodes-in-k-group/description/)

#### 题目描述
给你链表的头节点 `head` ，每 `k` 个节点一组进行翻转，请你返回修改后的链表。`k` 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 `k` 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

#### 解题思路
本题要求将链表按照每 `k` 个节点一组进行反转。整体的解题思路可以分为以下几个步骤：
1. **计算链表长度**：首先遍历链表，计算出链表的总节点数 `n`，这样可以知道有多少组 `k` 个节点需要反转。
2. **创建虚拟头节点**：使用一个虚拟头节点 `dummy_node` ，方便处理头节点反转的情况，避免边界条件的复杂判断。
3. **分组反转**：只要剩余节点数 `n` 大于等于 `k` ，就进行一组 `k` 个节点的反转操作。在每组反转完成后，将反转后的子链表正确连接回原链表。
4. **返回结果**：最后返回虚拟头节点的下一个节点，即反转后的链表头节点。

#### 代码实现
```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        # 计算链表长度
        n = 0
        cur = head
        while cur:
            n += 1
            cur = cur.next
        
        # 创建虚拟头节点
        dummy_node = ListNode(next = head)
        p0 = dummy_node
        pre = None
        cur = p0.next
        
        # 分组反转
        while n >= k:
            n -= k
            for i in range(k):
                temp = cur.next
                cur.next = pre
                pre = cur
                cur = temp
            temp = p0.next
            p0.next.next = cur
            p0.next = pre
            p0 = temp
        return dummy_node.next
```

#### 代码详细解释

##### 1. 计算链表长度
```python
n = 0
cur = head
while cur:
    n += 1
    cur = cur.next
```
- 初始化变量 `n` 为 0 ，用于记录链表的节点数。
- 使用指针 `cur` 从链表头开始遍历，每遍历一个节点，`n` 加 1 ，直到遍历完整个链表。

##### 2. 创建虚拟头节点并初始化指针
```python
dummy_node = ListNode(next = head)
p0 = dummy_node
pre = None
cur = p0.next
```
- 创建虚拟头节点 `dummy_node` ，其 `next` 指针指向原链表的头节点 `head`。
- `p0` 初始化为 `dummy_node` ，用于记录每组需要反转的子链表的前一个节点。
- `pre` 初始化为 `None` ，用于反转链表时保存前一个节点。
- `cur` 初始化为 `p0.next` ，即从链表的第一个节点开始处理。

##### 3. 分组反转
```python
while n >= k:
    n -= k
    for i in range(k):
        temp = cur.next
        cur.next = pre
        pre = cur
        cur = temp
    temp = p0.next
    p0.next.next = cur
    p0.next = pre
    p0 = temp
```
- **外层 `while` 循环**：只要剩余节点数 `n` 大于等于 `k` ，就进行一组 `k` 个节点的反转操作。每次反转一组后，`n` 减去 `k` 。
- **内层 `for` 循环**：对当前组的 `k` 个节点进行反转，使用 `temp` 临时保存当前节点的下一个节点，然后将当前节点的 `next` 指针指向前一个节点，接着更新 `pre` 和 `cur` 指针。
- **连接操作**：
  - `temp = p0.next` ：保存当前组反转前的第一个节点（反转后会成为该组的最后一个节点）。
  - `p0.next.next = cur` ：将当前组反转后的尾节点的 `next` 指针指向 `cur` ，即下一组的第一个节点。
  - `p0.next = pre` ：将 `p0` 的 `next` 指针指向当前组反转后的头节点 `pre` 。
  - `p0 = temp` ：将 `p0` 移动到当前组反转后的尾节点，为下一组反转做准备。

##### 4. 返回结果
```python
return dummy_node.next
```
最后返回虚拟头节点 `dummy_node` 的下一个节点，即反转后的链表头节点。

#### 复杂度分析
- **时间复杂度**：$O(n)$，其中 $n$ 是链表的长度。因为需要遍历链表一次，每个节点最多被访问两次（一次用于计算长度，一次用于反转）。
- **空间复杂度**：$O(1)$，只使用了常数级的额外空间，主要是几个指针变量。

综上所述，通过上述步骤可以实现将链表按每 `k` 个节点一组进行反转的功能。 
