# 快慢指针

# 876.链表的中间节点

[leetcode](https://leetcode.cn/problems/middle-of-the-linked-list/description/)

快指针走两步，慢指针走一步。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
```

# 141.环形链表

[leetcode](https://leetcode.cn/problems/linked-list-cycle/description/)

在上一题基础上，判断快慢指针能不能遇上，能遇上就存在环。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                return True
        return False
```


# 142.环形链表Ⅱ

[leetcode](https://leetcode.cn/problems/linked-list-cycle-ii/)

在上一题基础上，根据等式判断。见[链表第2节](https://samuelssj123.github.io/contents/ALGORITHMNOTES/Linked_List(2).html)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        fast = slow = head
        while fast and fast.next :
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                index1 = fast
                index2 = head
                while index1 != index2:
                    index1 = index1.next
                    index2 = index2.next
                return index1
        return None
```

# 143.重排链表

[leetcode](https://leetcode.cn/problems/reorder-list/description/)

在之前两个题目基础上，拼接起来就行。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    # 876. 链表的中间节点
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    # 206.反转链表
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre = None
        cur = head
        while cur:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        return pre
    
    
    def reorderList(self, head: Optional[ListNode]) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        mid = self.middleNode(head)
        head2 = self.reverseList(mid)
        while head2.next:
            temp1 = head.next
            temp2 = head2.next
            head.next = head2
            head2.next = temp1
            head = temp1
            head2 = temp2
```
