List: 647. 回文子串，516.最长回文子序列

[647. 回文子串palindromic-substrings](#01)，[](#02)，[](#03)

# <span id="01">647. 回文子串palindromic-substrings</span>

[Leetcode](https://leetcode.cn/problems/palindromic-substrings/) 

[Learning Materials](https://programmercarl.com/0647.%E5%9B%9E%E6%96%87%E5%AD%90%E4%B8%B2.html)

![image](../images/647-palindromic-substrings.png)

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        dp = [[False] * len(s) for _ in range(len(s))]
        result = 0
        for i in range(len(s) - 1, -1, -1):
            for j in range(i, len(s)):
                if s[i] == s[j]:
                    if j - i <= 1:
                        dp[i][j] = True
                        result += 1
                    elif dp[i + 1][j - 1] == True:
                        dp[i][j] = True
                        result += 1
        return result
```

## 双指针法：

```python
class Solution:
    def countSubstrings(self, s: str) -> int:
        result = 0
        for i in range(len(s)):
            result += self.extend(s, i, i, len(s)) #以i为中心
            result += self.extend(s, i, i+1, len(s)) #以i和i+1为中心
        return result
    
    def extend(self, s, i, j, n):
        res = 0
        while i >= 0 and j < n and s[i] == s[j]:
            i -= 1
            j += 1
            res += 1
        return res
```

# <span id="02">516.最长回文子序列longest-palindromic-subsequence</span>

[Leetcode](https://leetcode.cn/problems/longest-palindromic-subsequence/description/) 

[Learning Materials](https://programmercarl.com/0516.%E6%9C%80%E9%95%BF%E5%9B%9E%E6%96%87%E5%AD%90%E5%BA%8F%E5%88%97.html#%E7%AE%97%E6%B3%95%E5%85%AC%E5%BC%80%E8%AF%BE)

![image](../images/516-longest-palindromic-subsequence.png)

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        dp = [[0] * len(s) for _ in range(len(s))]
        for i in range(len(s)):
            dp[i][i] = 1
        for i in range(len(s) - 1, -1, -1):
            for j in range(i + 1, len(s)):
                if s[i] == s[j]:
                    dp[i][j] = dp[i + 1][j - 1] + 2
                else:
                    dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
        return dp[0][len(s) - 1]
```

# <span id="03">动态规划总结</span>


[Learning Materials](https://programmercarl.com/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%E6%80%BB%E7%BB%93%E7%AF%87.html)

- **基础题目**：以斐波那契数、爬楼梯等为代表，是理解动态规划基本概念和方法的入门题目，解题关键在于清晰地定义状态和状态转移方程，例如斐波那契数中通过定义`dp[i]`表示第`i`个斐波那契数，利用`dp[i]=dp[i - 1]+dp[i - 2]`的递推公式求解。

|题目名称|题目特点|解题要点|dp数组定义|注意事项|
|---|---|---|---|---|
|动态规划：斐波那契数|动规入门题|用动规五部曲解题，题目已给出递推公式和dp数组初始化方式|通常`dp[i]`表示第`i`个斐波那契数|用于熟悉动规解题方法|
|动态规划：爬楼梯|类似斐波那契数列|正常应先推导递推公式，再发现其与斐波那契数列的关系；初始化`dp[1]=1`，`dp[2]=2`，`i`从3开始遍历|`dp[i]`表示爬到第`i`层楼梯的方法数|`dp[0]`无意义，可不初始化；可深化为完全背包问题，如一步可爬1到m个台阶的情况|
|动态规划：使用最小花费爬楼梯|在爬楼梯基础上增加花费|理解题意中体力花费的方式，有不同的dp数组定义方式|可定义为第一步花费体力，最后一步不花费；也可定义为第一步不花费，最后一步花费|两种定义方式都能解题，代码实现略有不同|


**不同路径（包含无障碍和有障碍两种情况）**
   
- **无障碍**：给定一个m x n的网格，机器人从左上角出发，每次只能向下或向右移动一步，求到达右下角的不同路径数。定义二维数组`dp[i][j]`表示从(0, 0)到(i, j)的路径数，初始化`dp[i][0]`和`dp[0][j]`为1 ，因为从起点到第一行和第一列的每个位置都只有一种走法。通过双重循环遍历，利用递推公式`dp[i][j] = dp[i - 1][j] + dp[i][j - 1]`计算每个位置的路径数，即当前位置的路径数是其上方位置和左方位置的路径数之和。

- **有障碍**：在网格中增加了障碍，有障碍的位置不能通过。`dp[i][j]`的定义不变，但初始化时需要跳过障碍位置，若`obstacleGrid[i][0] == 1`，则`dp[i][0]`之后的元素为0；若`obstacleGrid[0][j] == 1`，则`dp[0][j]`之后的元素为0。遍历和递推过程中，遇到障碍位置`obstacleGrid[i][j] == 1`时，直接跳过该位置不进行计算，其他位置的递推公式与无障碍情况相同。

**整数拆分**：对于给定的正整数n，将其拆分为至少两个正整数的和，求这些正整数的最大乘积。定义`dp[i]`为分拆数字i的最大乘积，只初始化`dp[2] = 1` ，因为2只能拆分为1+1，乘积为1。通过双重循环遍历，内层循环从1到`i - 1` ，计算`dp[i] = max(dp[i], max((i - j) * j, dp[i - j] * j))` ，其中`(i - j) * j`表示将i拆分为j和`i - j`的乘积，`dp[i - j] * j`表示将i拆分为`i - j`和j ，且`dp[i - j]`是已计算出的最大乘积，取这两者的最大值与当前`dp[i]`比较并更新。

**不同的二叉搜索树**：给定一个整数n，求由1到n为节点组成的不同二叉搜索树的个数。定义`dp[i]`为1到i为节点组成的二叉搜索树的个数 ，初始化只需要`dp[0] = 1` ，这是为了后续计算方便，0个节点时可认为有一种空树的情况。通过循环遍历，对于每个i ，内层循环遍历j从1到i ，利用递推公式`dp[i] += dp[j - 1] * dp[i - j]`计算，其中`dp[j - 1]`表示以j - 1个节点组成的左子树的数量，`dp[i - j]`表示以`i - j`个节点组成的右子树的数量，两者相乘再累加得到`dp[i]`。 

      
- **背包问题系列**：包括01背包、完全背包、多重背包等。01背包中物品不可重复选取，完全背包物品可无限重复选取，多重背包物品有数量限制。这类问题通常需要仔细分析物品的选取策略，确定合适的状态和转移方程，如在分割等和子集中，通过将问题转化为01背包问题，利用`dp[j]`表示背包容量为`j`时能否达到目标和，根据物品的重量和价值更新`dp`数组。
    
- **打家劫舍系列**：解题要点在于处理好相邻元素不能同时选取的限制，通过合理定义状态来避免这种冲突。例如在经典的打家劫舍问题中，定义`dp[i]`表示抢劫到第`i`个房屋时能获得的最大金额，通过比较抢劫和不抢劫第`i`个房屋的情况来更新`dp`数组。
    
- **股票系列**：涉及不同买卖次数和条件下的股票买卖最佳时机问题。解题时需要根据具体的买卖规则，考虑多种状态，如是否持有股票、买卖的次数限制等，定义相应的状态和转移方程。例如在只能买卖一次的情况下，通过记录最低买入价格，计算当前卖出价格与最低买入价格的差值来更新最大利润。
    
- **子序列系列**：分为不连续和连续子序列以及回文相关题目。对于不连续子序列问题，如最长上升子序列，通常需要遍历每个元素，比较其与之前元素的大小关系，确定状态转移方程；对于连续子序列问题，如最长连续递增序列，只需关注相邻元素的大小关系；回文相关题目则要根据回文的特性，通过动态规划的方法判断子串是否为回文，如在判断回文子串时，利用`dp[i][j]`表示从`i`到`j`的子串是否为回文，根据子串长度和两端字符是否相等来更新`dp`数组。 

