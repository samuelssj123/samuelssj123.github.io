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
