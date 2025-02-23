[344.反转字符串 reverse string](#01)，[541. 反转字符串II  reverse string ii](#02)，[卡码网：54.替换数字](#03)

# <span id="01">344.反转字符串 reverse string</span>


[Leetcode](https://leetcode.cn/problems/reverse-string/description/) [Learning Materials](https://programmercarl.com/0344.%E5%8F%8D%E8%BD%AC%E5%AD%97%E7%AC%A6%E4%B8%B2.html)

如果题目关键的部分直接用库函数就可以解决，建议不要使用库函数。

```python
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

```c++
for (int i = 0, j = s.size() - 1; i < s.size()/2; i++, j--)      #i < s.size()/2,不需要加等号，因为i、j相等时，无需交换。
            swap(s[i],s[j]);
```

# <span id="02">541. 反转字符串II  reverse string ii</span>

     
[Leetcode](https://leetcode.cn/problems/reverse-string-ii/description/) [Learning Materials](https://programmercarl.com/0541.%E5%8F%8D%E8%BD%AC%E5%AD%97%E7%AC%A6%E4%B8%B2II.html#%E7%AE%97%E6%B3%95%E5%85%AC%E5%BC%80%E8%AF%BE)

-思路：

1. 每隔 2k 个字符的前 k 个字符进行反转
   
2. 剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符
   
3. 剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符

```python
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

# <span id="03">卡码网：54.替换数字 </span>


[Leetcode](https://kamacoder.com/problempage.php?pid=1064) [Learning Materials](https://programmercarl.com/kamacoder/0054.%E6%9B%BF%E6%8D%A2%E6%95%B0%E5%AD%97.html#%E6%80%9D%E8%B7%AF)

- 思路：
  
首先扩充数组到每个数字字符替换成 "number" 之后的大小。

然后**从后向前**替换数字字符，也就是双指针法，过程如下：i指向新长度的末尾，j指向旧长度的末尾。（从前向后填充就是O(n^2)的算法了，因为每次添加元素都要将添加元素之后的所有元素整体向后移动。）

**数组填充类的问题，其做法都是先预先给数组扩容带填充后的大小，然后在从后向前进行操作。**

**好处**：不用申请新数组。从后向前填充元素，避免了从前向后填充元素时，每次添加元素都要将添加元素之后的所有元素向后移动的问题。

```c++
#include <iostream>
using namespace std;
int main() {
    string s;
    while (cin >> s) {
        int sOldIndex = s.size() - 1;
        int count = 0; // 统计数字的个数
        for (int i = 0; i < s.size(); i++) {
            if (s[i] >= '0' && s[i] <= '9') {
                count++;
            }
        }
        // 扩充字符串s的大小，也就是将每个数字替换成"number"之后的大小
        s.resize(s.size() + count * 5);
        int sNewIndex = s.size() - 1;
        // 从后往前将数字替换为"number"
        while (sOldIndex >= 0) {
            if (s[sOldIndex] >= '0' && s[sOldIndex] <= '9') {
                s[sNewIndex--] = 'r';
                s[sNewIndex--] = 'e';
                s[sNewIndex--] = 'b';
                s[sNewIndex--] = 'm';
                s[sNewIndex--] = 'u';
                s[sNewIndex--] = 'n';
            } else {
                s[sNewIndex--] = s[sOldIndex];
            }
            sOldIndex--;
        }
        cout << s << endl;       
    }
}
```

```python
def replace_digits_with_number(s: str) -> str:
    result = []
    for char in s:
        if char.isdigit():
            result.append("number")
        else:
            result.append(char)
    return ''.join(result)
```
s = input().strip() 

print(replace_digits_with_number(s))
```

```c++
#include <iostream>
using namespace std;
int main(){
    string s;
    while (cin>>s){
        int sOldIndex = s.size()-1;
        int count = 0;
        for (int i=0; i < s.size(); i++){
            if (s[i] >= '0' && s[i] <= '9'){
                count ++;
            }
        }
        s.resize(s.size()+count*5);
        int sNewIndex = s.size()-1;
        while (sOldIndex >= 0){
            if (s[sOldIndex] >= '0' && s[sOldIndex] <= '9'){
                s[sNewIndex--]='r';
                s[sNewIndex--]='e';
                s[sNewIndex--]='b';
                s[sNewIndex--]='m';
                s[sNewIndex--]='u';
                s[sNewIndex--]='n';
            } else{
                s[sNewIndex--]=s[sOldIndex];
            }
            sOldIndex--;
        }
        cout << s << endl;
    }
}
```
