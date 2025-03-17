List: 62.不同路径，63. 不同路径 II，343. 整数拆分，96.不同的二叉搜索树


[62.不同路径unique-paths](#01)，[63. 不同路径 IIunique-paths-ii](#02)，[](#03)，[](#04),[](#05)

# <span id="01">62.不同路径unique-paths</span>

[Leetcode](https://leetcode.cn/problems/unique-paths/) 

[Learning Materials](https://programmercarl.com/0062.%E4%B8%8D%E5%90%8C%E8%B7%AF%E5%BE%84.html)

![image](../images/62-unique-paths.png)


## 方法一：动态规划

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        if m == 1 or n == 1:
            return 1
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            dp[i][0] = 1
        for j in range(n):
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]
```

## 方法二：滚动数组

由于 f(i,j) 仅与第 i 行和第 i−1 行的状态有关，因此我们可以使用滚动数组代替代码中的二维数组，使空间复杂度降低为 O(n)。

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        return dp[n - 1]
```

## 方法三：组合数

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # 初始化分子为 1，用于存储组合数公式中的分子部分
        numerator = 1
        # 初始化分母为 m - 1，用于存储组合数公式中的分母部分
        denominator = m - 1
        # 记录需要进行乘法运算的次数，也就是分子中需要乘的数的个数
        count = m - 1
        # 计算总共需要移动的步数，即 m + n - 2
        t = m + n - 2
        # 循环 m - 1 次，计算分子部分
        while count > 0:
            # 分子乘以当前的步数 t
            numerator *= t
            # 步数 t 减 1，准备乘下一个数
            t -= 1
            # 当分母不为 0 且分子能被分母整除时，进行约分操作
            while denominator != 0 and numerator % denominator == 0:
                # 分子除以分母
                numerator //= denominator
                # 分母减 1，准备约下一个数
                denominator -= 1
            # 乘法运算次数减 1
            count -= 1
        # 最终的分子即为不同路径的数量，返回结果
        return numerator
```

# <span id="02">63. 不同路径 IIunique-paths-ii</span>

[Leetcode](https://leetcode.cn/problems/unique-paths-ii/description/) 

[Learning Materials](https://programmercarl.com/0063.%E4%B8%8D%E5%90%8C%E8%B7%AF%E5%BE%84II.html#%E7%AE%97%E6%B3%95%E5%85%AC%E5%BC%80%E8%AF%BE)

![image](../images/63-unique-paths-ii.png)

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        if obstacleGrid[0][0] == 1:
            return 0
        dp = [[0] * n for _ in range(m)]
        for i in range(m):
            if obstacleGrid[i][0] != 0:
                break
            dp[i][0] = 1
        for j in range(n):
            if obstacleGrid[0][j] != 0:
                break
            dp[0][j] = 1
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[m - 1][n - 1]
```

# <span id="03">理论基础</span>

[Leetcode]() 

[Learning Materials]()

![image](../images/.png)

# <span id="04">理论基础</span>

[Leetcode]() 

[Learning Materials]()

![image](../images/.png)

# <span id="05">理论基础</span>

[Leetcode]() 

[Learning Materials]()

![image](../images/.png)
