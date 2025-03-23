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
#本部分代码参考作者：灵茶山艾府
