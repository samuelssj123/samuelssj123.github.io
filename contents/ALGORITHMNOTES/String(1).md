
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

541. 反转字符串II  reverse string ii
     
[Leetcode](https://leetcode.cn/problems/reverse-string-ii/description/) [Learning Materials](https://programmercarl.com/0541.%E5%8F%8D%E8%BD%AC%E5%AD%97%E7%AC%A6%E4%B8%B2II.html#%E7%AE%97%E6%B3%95%E5%85%AC%E5%BC%80%E8%AF%BE)

-思路：

1. 每隔 2k 个字符的前 k 个字符进行反转
   
2. 剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符
   
3. 剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符

```Python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        """
        1. 使用range(start, end, step)来确定需要调换的初始位置
        2. 对于字符串s = 'abc'，如果使用s[0:999] ===> 'abc'。字符串末尾如果超过最大长度，则会返回至字符串最后一个值，这个特性可以避免一些边界条件的处理。
        3. 用切片整体替换，而不是一个个替换.
        """ 
        def reverse(text):
            i = 0
            j = len(text) - 1
            while i < j:
                text[i] , text[j] = text[j], text[i]
                i += 1
                j -= 1
            return text
        
        res = list(s)

        for i in range(0, len(s), 2 * k):
            res[i: i+k] = reverse(res[i : i+k])

        return ''.join(res)
```

卡码网：54.替换数字 

[Leetcode](https://kamacoder.com/problempage.php?pid=1064) [Learning Materials](https://programmercarl.com/kamacoder/0054.%E6%9B%BF%E6%8D%A2%E6%95%B0%E5%AD%97.html#%E6%80%9D%E8%B7%AF)
