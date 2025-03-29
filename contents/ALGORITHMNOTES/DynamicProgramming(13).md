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

---
### 一段话总结
本文是动态规划专题的总结，作者强调**动规五部曲**（确定dp数组及下标的含义、确定递推公式、dp数组如何初始化、确定遍历顺序、举例推导dp数组）对解动规题目的重要性。文中回顾了已讲解的42道动态规划经典题目，涵盖动态规划基础、背包问题系列、打家劫舍系列、股票系列、子序列系列等，还指出树形DP、数位DP等在面试中出现概率低，掌握文中题目对面试应对动态规划问题已足够。
---
### 思维导图
```mindmap
## **动态规划解题方法**
- 确定dp数组及下标的含义
- 确定递推公式
- dp数组的初始化方式
- 确定遍历顺序
- 举例推导dp数组
## **动态规划题目分类**
- 基础题目：斐波那契数、爬楼梯等7道题
- 背包问题：01背包、完全背包、多重背包等13道题
- 打家劫舍系列：3道题
- 股票系列：6道题
- 子序列系列：12道题
## **动态规划题目难度及应用**
- 简单题用于巩固方法论
- 难题需全面考虑动规五部曲
- 部分类型面试中出现概率低
```
---
### 详细总结
1. **动态规划解题五部曲**：作者强调解动态规划题目的核心方法为动规五部曲，即确定dp数组（dp table）以及下标的含义、确定递推公式、dp数组如何初始化、确定遍历顺序、举例推导dp数组。这五步对解决动规题目至关重要，任何一步思考不充分都可能导致题目无法正确解答。
2. **动态规划题目分类及题目数量**
    - **动态规划基础**：共7道题，包括斐波那契数、爬楼梯、使用最小花费爬楼梯、不同路径、不同路径（含障碍）、整数拆分、不同的二叉搜索树。
    - **背包问题系列**：共13道题，涵盖01背包、完全背包、多重背包相关题目，如分割等和子集、最后一块石头的重量II、目标和、一和零、零钱兑换II等。
    - **打家劫舍系列**：有3道题，逐步深入该类型问题的解法。
    - **股票系列**：包含6道题，涉及不同买卖次数和条件下的股票买卖最佳时机问题，如只能买卖一次、可以买卖多次、最多买卖两次、最多买卖k次、含冷冻期、含手续费等情况。
    - **子序列系列**：有12道题，分为子序列（不连续）和子序列（连续）以及回文相关题目，如最长上升子序列、最长公共子序列、最长连续递增序列、最长重复子数组、判断子序列、编辑距离、回文子串、最长回文子序列等。
3. **动态规划题目难度及应用场景**：简单动规题目可用于巩固方法论，复杂题目则需要全面、深入地思考动规五部曲的各个环节。此外，树形DP、数位DP、区间DP、概率型DP、博弈型DP、状态压缩dp等在面试中出现的概率较低，掌握文中列举的题目，应对面试中的动规问题已足够。

|分类|题目数量|典型题目|
|---|---|---|
|动态规划基础|7道|斐波那契数、爬楼梯等|
|背包问题系列|13道|分割等和子集、零钱兑换II等|
|打家劫舍系列|3道|动态规划：开始打家劫舍等|
|股票系列|6道|买卖股票的最佳时机（不同买卖条件）|
|子序列系列|12道|最长上升子序列、编辑距离等|
---
### 关键问题
1. **动规五部曲中哪一步最重要？**
    - 动规五部曲中每一步都至关重要，缺少任何一步的深入思考都可能导致无法正确解答题目。比如没想清楚dp数组的含义，递推公式就难以确定，初始化也可能出错；遍历顺序错误可能导致结果不准确；不通过举例推导dp数组，很难发现程序中的逻辑错误。
2. **背包问题系列包含哪些具体类型？**
    - 背包问题系列包含01背包、完全背包、多重背包。01背包中物品不可重复选取；完全背包中物品可无限重复选取；多重背包则是在01背包基础上，物品有一定的数量限制 。
3. **面试中

