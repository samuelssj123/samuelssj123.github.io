To do list: 24. 两两交换链表中的节点，19.删除链表的倒数第N个节点，面试题 02.07. 链表相交，142.环形链表II，总结
[24. 两两交换链表中的节点](#01)，[19.删除链表的倒数第N个节点](#02)，[面试题 02.07. 链表相交](#03)，[142.环形链表II](#04)，[总结](#05)

# <span id="01">24. 两两交换链表中的节点</span>

[题目链接/文章讲解/视频讲解](https://programmercarl.com/0024.%E4%B8%A4%E4%B8%A4%E4%BA%A4%E6%8D%A2%E9%93%BE%E8%A1%A8%E4%B8%AD%E7%9A%84%E8%8A%82%E7%82%B9.html)

Hint: temp can be used to save the temporary node.

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



# <span id="03">面试题 02.07. 链表相交</span>

[题目链接/文章讲解](https://programmercarl.com/%E9%9D%A2%E8%AF%95%E9%A2%9802.07.%E9%93%BE%E8%A1%A8%E7%9B%B8%E4%BA%A4.html)



# <span id="04">142.环形链表II </span>

[题目链接/文章讲解/视频讲解](https://programmercarl.com/0142.%E7%8E%AF%E5%BD%A2%E9%93%BE%E8%A1%A8II.html)



# <span id="05">总结 </span>

[题目链接/文章讲解/视频讲解](https://www.programmercarl.com/%E9%93%BE%E8%A1%A8%E6%80%BB%E7%BB%93%E7%AF%87.html)


