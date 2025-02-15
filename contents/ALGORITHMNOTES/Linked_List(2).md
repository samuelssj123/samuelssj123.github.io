To do list: 24. 两两交换链表中的节点，19.删除链表的倒数第N个节点，面试题 02.07. 链表相交，142.环形链表II，总结
[24. 两两交换链表中的节点](#01)，[19.删除链表的倒数第N个节点](#02)，[面试题 02.07. 链表相交](#03)，[142.环形链表II](#04)，[总结](#05)

# <span id="01">24. 两两交换链表中的节点</span>

[题目链接/文章讲解/视频讲解](https://programmercarl.com/0024.%E4%B8%A4%E4%B8%A4%E4%BA%A4%E6%8D%A2%E9%93%BE%E8%A1%A8%E4%B8%AD%E7%9A%84%E8%8A%82%E7%82%B9.html)

**Hint**: temp can be used to save the temporary node.

```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy_head = ListNode (next = head)
        current = dummy_head
        while current.next != None and current.next.next != None:
            temp1 = current.next
            temp2 = current.next.next.next
            current.next = current.next.next
            current.next.next = temp1
            temp1.next = temp2
            current = current.next.next
        return dummy_head.next
```


# <span id="02">19.删除链表的倒数第N个节点</span>

[题目链接/文章讲解/视频讲解](https://programmercarl.com/0019.%E5%88%A0%E9%99%A4%E9%93%BE%E8%A1%A8%E7%9A%84%E5%80%92%E6%95%B0%E7%AC%ACN%E4%B8%AA%E8%8A%82%E7%82%B9.html)

**关键点**：要删第几个，就要知道它的前一个；快慢指针，快的先走n步，然后再一起走，slow就是倒数第n个

```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy_head = ListNode (next = head)
        slow = dummy_head
        fast = dummy_head
        n += 1
        while n :
            fast = fast.next
            n -= 1
        while fast:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy_head.next
```

# <span id="03">面试题 02.07. 链表相交</span>

[题目链接/文章讲解](https://programmercarl.com/%E9%9D%A2%E8%AF%95%E9%A2%9802.07.%E9%93%BE%E8%A1%A8%E7%9B%B8%E4%BA%A4.html)

**关键点**：求两个链表交点节点的指针。要注意，交点不是数值相等，而是指针相等。

步骤：1.求两个链长度；2.求差值；3.长的那条走到等长的点；4.找交点

```Python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:
        lengthA = lengthB = 0
        curA = headA
        curB = headB

        while curA :
            curA = curA.next
            lengthA += 1
        while curB :
            curB = curB.next
            lengthB += 1
        
        gap = abs(lengthA - lengthB)
        curA = headA
        curB = headB

        if lengthA > lengthB:
            while gap:
                curA = curA.next
                gap -= 1
        else :
            while gap:
                curB = curB.next
                gap -= 1
        
        while curA and curB:
            if curA == curB :
                return curA
            else :
                curA = curA.next
                curB = curB.next
```

# <span id="04">142.环形链表II </span>

[题目链接/文章讲解/视频讲解](https://programmercarl.com/0142.%E7%8E%AF%E5%BD%A2%E9%93%BE%E8%A1%A8II.html)



# <span id="05">总结 </span>

[题目链接/文章讲解/视频讲解](https://www.programmercarl.com/%E9%93%BE%E8%A1%A8%E6%80%BB%E7%BB%93%E7%AF%87.html)

- 链表的主要知识：链表的种类主要为：单链表，双链表，循环链表。链表的存储方式：链表的节点在内存中是分散存储的，通过指针连在一起。链表是如何进行增删改查的。数组和链表在不同场景下的性能分析。

- 链表的一大问题就是操作当前节点必须要找前一个节点才能操作。这就造成了，头结点的尴尬，因为头结点没有前一个节点了。每次对应头结点的情况都要单独处理，所以使用**虚拟头结点**的技巧，就可以解决这个问题。

- 反转链表：迭代法、递归法。

- 结合虚拟头结点和双指针法来移除链表倒数第N个节点。

- 使用双指针来找到两个链表的交点（引用完全相同，即：内存地址完全相同的交点）
