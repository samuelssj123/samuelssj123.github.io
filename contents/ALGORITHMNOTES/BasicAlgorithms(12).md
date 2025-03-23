# 动态规划

- 状态定义？非状态转移方程？

- 启发思路：选或不选/选哪个

##  例题1 [198.打家劫舍](https://leetcode.cn/problems/house-robber/)

先考虑用回溯解决，把大问题变小问题。

当前操作？枚举第个房子选/不选

子问题？从前个房子中得到的最大金额和

下一个子问题？分类讨论：

  不选：从前一1个房子中得到的最大金额和
  
  选：从前一2个房子中得到的最大金额和

  dfs(i) =max (dfs(i - 1),dfs(i - 2) + nums[i])

  注意传入的是函数，不是一个数

  有树相同：把递归的计算结果保存下来，那么下次递归到同样的入参时，就直接返回先前保存的结果


  时间复杂度：O(n)。其中 n 为 nums 的长度。
  
  空间复杂度：O(n)。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        f = [0] * (len(nums) + 2)
        for i, x in enumerate(nums):
            f[i + 2] = max(f[i + 1], f[i] + x)
        return f[-1]
```

  再优化空间复杂度：递归变递推

  自顶向下算=记忆化搜索
  
自底向上算＝递推

dfs→f数组

1:1 翻译成递推

递归→循环

递归边界→数组初始值

dfs(i) =max(dfs(i- 1),dfs(i - 2) + nums[il)

f[i] =max(f[i-1],fli-2] + nums[l)

f[i + 2] =max(f[i + 1],fli] + nums[l)

当前=max（上一个，上上一个+nums[i]）

fo表示上上一个，f1表示上一个

newF =max(f1,fo + nums[il)

fo =fi

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        f0 = f1 = 0
        for x in nums:
            f0, f1 = f1, max(f1, f0 + x)
        return f1
```
fi=newF

## 背包类

例题2：01背包[494.目标和](https://leetcode.cn/problems/target-sum/)

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        s = sum(nums) - abs(target)
        if s < 0 or s % 2:
            return 0
        m = s // 2  # 背包容量

        @cache  # 缓存装饰器，避免重复计算 dfs 的结果（记忆化）
        def dfs(i: int, c: int) -> int:
            if i < 0:
                return 1 if c == 0 else 0
            if c < nums[i]:
                return dfs(i - 1, c)  # 只能不选
            return dfs(i - 1, c) + dfs(i - 1, c - nums[i])  # 不选 + 选
        return dfs(len(nums) - 1, m)
```

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        s = sum(nums) - abs(target)
        if s < 0 or s % 2:
            return 0
        m = s // 2  # 背包容量

        n = len(nums)
        f = [[0] * (m + 1) for _ in range(n + 1)]
        f[0][0] = 1
        for i, x in enumerate(nums):
            for c in range(m + 1):
                if c < x:
                    f[i + 1][c] = f[i][c]  # 只能不选
                else:
                    f[i + 1][c] = f[i][c] + f[i][c - x]  # 不选 + 选
        return f[n][m]
```

例题3：完全背包[322.零钱兑换](https://leetcode.cn/problems/coin-change/description/)


```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        @cache  # 缓存装饰器，避免重复计算 dfs 的结果（记忆化）
        def dfs(i: int, c: int) -> int:
            if i < 0:
                return 0 if c == 0 else inf
            if c < coins[i]:
                return dfs(i - 1, c)
            return min(dfs(i - 1, c), dfs(i, c - coins[i]) + 1)
        ans = dfs(len(coins) - 1, amount)
        return ans if ans < inf else -1
```

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        f = [[inf] * (amount + 1) for _ in range(n + 1)]
        f[0][0] = 0
        for i, x in enumerate(coins):
            for c in range(amount + 1):
                if c < x:
                    f[i + 1][c] = f[i][c]
                else:
                    f[i + 1][c] = min(f[i][c], f[i + 1][c - x] + 1)
        ans = f[n][amount]
        return ans if ans < inf else -1

```

# 线性DP

例题4：[1143.最长公共子序列](https://leetcode.cn/problems/longest-common-subsequence/description/)

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n, m = len(text1), len(text2)
        @cache
        def dfs(i, j):
            if i < 0 or j < 0:
                return 0
            if text1[i] == text2[j]:
                return dfs(i - 1, j - 1) + 1
            return max(dfs(i - 1, j), dfs(i, j - 1))
        return dfs(n - 1, m - 1)
```

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n, m = len(text1), len(text2)
        f = [[0] * (m + 1) for _ in range(n + 1)]
        for i, x in enumerate(text1):
            for j, y in enumerate(text2):
                f[i + 1][j + 1] = f[i][j] + 1 if x == y else max(f[i][j + 1], f[i + 1][j])
        return f[n][m]
```

例题5： [72.编辑距离](https://leetcode.cn/problems/edit-distance/description/)

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n, m = len(word1), len(word2)
        @cache
        def dfs(i, j):
            if i < 0 :
                return j + 1
            if j < 0:
                return i + 1
            if word1[i] == word2[j]:
                return dfs(i - 1, j - 1)
            return min(dfs(i - 1, j), dfs(i, j - 1), dfs(i - 1, j - 1)) + 1
        return dfs(n - 1, m - 1)
```

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n, m = len(word1), len(word2)
        f = [[0] * (m + 1) for _ in range(n + 1)]
        f[0] = list(range(m + 1))
        for i, x in enumerate(word1):
            f[i + 1][0] = i + 1
            for j, y in enumerate(word2):
                f[i + 1][j + 1] = f[i][j] if x == y else min(f[i][j + 1], f[i + 1][j], f[i][j]) + 1
        return f[n][m]
```

例题6：[300.最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/description/)

### 题目描述
这是 LeetCode 的第 300 题“最长递增子序列”。题目要求给定一个整数数组 `nums`，找到其中最长严格递增子序列的长度。子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。

### 代码思路分析
本题使用了记忆化搜索（带缓存的深度优先搜索）的方法来解决。下面详细解释代码的每一部分：

#### 代码实现
```python
from functools import cache
from typing import List

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # 使用 cache 装饰器来缓存 dfs 函数的结果，避免重复计算
        @cache
        def dfs(i: int) -> int:
            # 初始化以 nums[i] 结尾的最长递增子序列的长度为 0
            res = 0
            # 遍历 nums[i] 之前的所有元素
            for j in range(i):
                # 如果 nums[j] 小于 nums[i]，说明可以将 nums[i] 接在以 nums[j] 结尾的递增子序列后面
                if nums[j] < nums[i]:
                    # 更新 res 为当前 res 和以 nums[j] 结尾的最长递增子序列长度的最大值
                    res = max(res, dfs(j))
            # 因为当前元素 nums[i] 自身也算一个长度，所以在最长的子序列长度基础上加 1
            return res + 1
        # 遍历数组中的每个元素，计算以每个元素结尾的最长递增子序列长度，并取最大值
        return max(dfs(i) for i in range(len(nums)))

```

#### 复杂度分析
- **时间复杂度**：$O(n^2)$，其中 $n$ 是数组 `nums` 的长度。对于每个位置 `i`，都需要遍历其前面的所有位置 `j`，因此总的时间复杂度为 $O(n^2)$。
- **空间复杂度**：$O(n)$，主要是递归调用栈和缓存的空间开销，递归深度最大为 $n$，缓存也需要存储 $n$ 个结果。

### 代码解释
1. **`@cache` 装饰器**：`@cache` 是 Python 3.9 及以上版本中 `functools` 模块提供的装饰器，用于自动缓存函数的输入和输出。在递归过程中，如果多次调用 `dfs(i)` 且输入参数 `i` 相同，就可以直接从缓存中获取结果，避免重复计算，大大提高了效率。
2. **`dfs(i)` 函数**：该函数用于计算以 `nums[i]` 结尾的最长递增子序列的长度。具体步骤如下：
    - 初始化 `res` 为 0，表示当前找到的最长递增子序列的长度。
    - 遍历 `nums[i]` 之前的所有元素 `nums[j]`（`j` 从 0 到 `i-1`）。
    - 如果 `nums[j] < nums[i]`，说明可以将 `nums[i]` 接在以 `nums[j]` 结尾的递增子序列后面，更新 `res` 为当前 `res` 和 `dfs(j)` 的最大值。
    - 最后返回 `res + 1`，因为当前元素 `nums[i]` 自身也算一个长度。
3. **`max(dfs(i) for i in range(len(nums)))`**：遍历数组中的每个元素，计算以每个元素结尾的最长递增子序列长度，并取最大值，即为整个数组的最长递增子序列长度。

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        f = [0] * len(nums)
        for i, x in enumerate(nums):
            for j, y in enumerate(nums[:i]):
                if x > y:
                    f[i] = max(f[i], f[j])
            f[i] += 1
        return max(f)
```

- 拔高思路：

### 方法思路

这种方法通过维护一个数组 `g`，其中 `g[i]` 表示长度为 `i+1` 的所有递增子序列中末尾元素的最小值。遍历数组中的每个元素时，使用二分查找确定该元素在 `g` 中的位置，并更新 `g` 数组。最终，`g` 的长度即为最长递增子序列（LIS）的长度。

**关键思路**：
1. **贪心策略**：在构建递增子序列时，尽可能使用较小的元素作为末尾，以便后续元素更容易形成更长的子序列。
2. **二分优化**：通过二分查找快速定位当前元素在 `g` 数组中的位置，确保时间复杂度为 O(n log n)。

### 数学证明

**命题**：`g` 数组的长度等于最长递增子序列的长度。

**证明**：
1. **单调性**：`g` 数组始终保持严格递增。归纳法证明：
   - 初始时 `g` 为空。
   - 每次插入或替换元素后，`g` 仍保持递增。例如，插入位置 `j` 时，原 `g[j] >= x`，替换为 `x` 后，`g[j-1] < x <= g[j]`（若存在），保持递增。
2. **最优性**：`g[i]` 是长度为 `i+1` 的递增子序列的最小末尾。归纳法证明：
   - 初始成立。
   - 处理元素 `x` 时，找到 `j` 使得 `x` 可接在长度为 `j` 的子序列后，形成长度为 `j+1` 的子序列。此时若 `x < g[j]`，则替换 `g[j]` 为 `x`，维护了最小末尾。
3. **结论**：`g` 的长度即为 LIS 的长度，因为每次扩展 `g` 的长度时，必然找到了更长的递增子序列。

### 实现代码

```python
from bisect import bisect_left
from typing import List

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        g = []
        for x in nums:
            j = bisect_left(g, x)
            if j == len(g):
                g.append(x)
            else:
                g[j] = x
        return len(g)
```

### 代码解释

- **初始化**：空数组 `g` 用于维护递增子序列的最小末尾。
- **遍历数组**：对于每个元素 `x`，使用 `bisect_left` 找到其在 `g` 中的插入位置 `j`。
  - **扩展长度**：若 `j` 等于 `g` 的长度，说明 `x` 可扩展当前最长子序列，将其追加到 `g`。
  - **更新末尾**：否则，替换 `g[j]` 为 `x`，以维护该长度下的最小末尾。
- **返回结果**：最终 `g` 的长度即为 LIS 的长度。

**时间复杂度**：O(n log n)，其中 `n` 为数组长度。每次二分查找和插入操作的时间为 O(log k)，`k` 为当前 `g` 的长度，总共有 `n` 次操作。

**空间复杂度**：O(k)，其中 `k` 为 LIS 的长度，最坏情况下 `k = n`。

#本部分代码参考作者：灵茶山艾府
