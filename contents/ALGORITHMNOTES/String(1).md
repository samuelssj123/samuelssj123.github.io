
344.反转字符串 reverse string

[Leetcode](https://leetcode.cn/problems/reverse-string/description/) [Learning Materials](https://programmercarl.com/0344.%E5%8F%8D%E8%BD%AC%E5%AD%97%E7%AC%A6%E4%B8%B2.html)

如果题目关键的部分直接用库函数就可以解决，建议不要使用库函数。

```Python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        i = 0
        j = len(s) - 1
        for i in range(len(s) // 2):
            s[i] , s[j] = s[j], s[i]
            i += 1
            j -= 1
```
-边界条件：

```C++
for (int i = 0, j = s.size() - 1; i < s.size()/2; i++, j--)      #i < s.size()/2,不需要加等号，因为i、j相等时，无需交换。
            swap(s[i],s[j]);
```

[Leetcode]() [Learning Materials]()

[Leetcode]() [Learning Materials]()
