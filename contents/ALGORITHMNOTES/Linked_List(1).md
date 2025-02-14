[链表理论基础](#01)，[203.移除链表元素](#02)，707.设计链表，206.反转链表

# <span id="01">链表理论基础</span>

[Textual Interpretation](https://programmercarl.com/%E9%93%BE%E8%A1%A8%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.html)

类型：单链表中的指针域只能指向节点的下一个节点。双链表的每一个节点有两个指针域，一个指向下一个节点，一个指向上一个节点。循环链表，就是链表首尾相连，可以用来解决约瑟夫环问题。

存储方式：链表在内存中可不是连续分布的。链表是通过指针域的指针链接在内存中各个节点。所以链表中的节点在内存中不是连续分布的 ，而是散乱分布在内存中的某地址上，分配机制取决于操作系统的内存管理。

定义：

```C++
// 单链表
struct ListNode {
    int val;  // 节点上存储的元素
    ListNode *next;  // 指向下一个节点的指针
    ListNode(int x) : val(x), next(NULL) {}  // 节点的构造函数
};
```

操作：删除-换连接，释放内存（Python自动回收），O(n)需要找到删除节点的位置；添加-插入节点O(1)

```Python
class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
```

# <span id="02">203.移除链表元素</span>

[Related Explaination](https://programmercarl.com/0203.%E7%A7%BB%E9%99%A4%E9%93%BE%E8%A1%A8%E5%85%83%E7%B4%A0.html)

-直接删除：

```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        if not head:
            return None
        while head != None and head.val == val :
            head = head.next
        current = head
        while current != None and current.next != None:
            if current.next.val == val:
                current.next = current.next.next
            else:
                current = current.next
        return head
```
