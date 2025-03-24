List: 121. 买卖股票的最佳时机，122.买卖股票的最佳时机II，123.买卖股票的最佳时机III

[121. 买卖股票的最佳时机](#01)，[122.买卖股票的最佳时机IIbest-time-to-buy-and-sell-stock-ii](#02)，[123.买卖股票的最佳时机IIIbest-time-to-buy-and-sell-stock-iii](#03)

# <span id="01">121. 买卖股票的最佳时机</span>

[Leetcode](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/) 

[Learning Materials](https://programmercarl.com/0121.%E4%B9%B0%E5%8D%96%E8%82%A1%E7%A5%A8%E7%9A%84%E6%9C%80%E4%BD%B3%E6%97%B6%E6%9C%BA.html)

![image](../images/121-best-time-to-buy-and-sell-stock.png)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[0] * 2 for _ in range(len(prices))]
        dp[0][0] = -prices[0]
        dp[0][1] = 0
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], - prices[i])
            dp[i][1] = max(dp[i - 1][0] + prices[i], dp[i - 1][1])
        return max(dp[len(prices) - 1][0], dp[len(prices) - 1][1])
```

- 优化：从递推公式可以看出，dp[i]只是依赖于dp[i - 1]的状态。那么我们只需要记录 当前天的dp状态和前一天的dp状态就可以了，可以使用滚动数组来节省空间，代码如下：

```c++
        int len = prices.size();
        vector<vector<int>> dp(2, vector<int>(2)); // 注意这里只开辟了一个2 * 2大小的二维数组
        dp[0][0] -= prices[0];
        dp[0][1] = 0;
        for (int i = 1; i < len; i++) {
            dp[i % 2][0] = max(dp[(i - 1) % 2][0], -prices[i]);
            dp[i % 2][1] = max(dp[(i - 1) % 2][1], prices[i] + dp[(i - 1) % 2][0]);
        }
        return dp[(len - 1) % 2][1];
```

# <span id="02">122.买卖股票的最佳时机IIbest-time-to-buy-and-sell-stock-ii</span>

[Leetcode](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/description/) 

[Learning Materials](https://programmercarl.com/0122.%E4%B9%B0%E5%8D%96%E8%82%A1%E7%A5%A8%E7%9A%84%E6%9C%80%E4%BD%B3%E6%97%B6%E6%9C%BAII%EF%BC%88%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92%EF%BC%89.html#%E7%AE%97%E6%B3%95%E5%85%AC%E5%BC%80%E8%AF%BE)

![image](../images/122-best-time-to-buy-and-sell-stock-ii.png)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[0] * 2 for _ in range(len(prices))]
        dp[0][0] = -prices[0]
        dp[0][1] = 0
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i])  #唯一区别
            dp[i][1] = max(dp[i - 1][0] + prices[i], dp[i - 1][1])
        return max(dp[len(prices) - 1][0], dp[len(prices) - 1][1])
```

# <span id="03">123.买卖股票的最佳时机IIIbest-time-to-buy-and-sell-stock-iii</span>

[Leetcode](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/description/) 

[Learning Materials](https://programmercarl.com/0123.%E4%B9%B0%E5%8D%96%E8%82%A1%E7%A5%A8%E7%9A%84%E6%9C%80%E4%BD%B3%E6%97%B6%E6%9C%BAIII.html)

![image](../images/123-best-time-to-buy-and-sell-stock-iii.png)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[0] * 5 for _ in range(len(prices))]
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        dp[0][2] = 0
        dp[0][3] = -prices[0]
        dp[0][4] = 0
        for i in range(1, len(prices)):
            dp[i][0] = dp[i - 1][0]
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])  
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + prices[i])
            dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i])  
            dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i])
        return dp[len(prices) - 1][4]
```


- 优化的c++版

```c++
// 版本二
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.size() == 0) return 0;
        vector<int> dp(5, 0);
        dp[1] = -prices[0];
        dp[3] = -prices[0];
        for (int i = 1; i < prices.size(); i++) {
            dp[1] = max(dp[1], dp[0] - prices[i]);
            dp[2] = max(dp[2], dp[1] + prices[i]);
            dp[3] = max(dp[3], dp[2] - prices[i]);
            dp[4] = max(dp[4], dp[3] + prices[i]);
        }
        return dp[4];
    }
};
```

- 解释：

dp[1] = max(dp[1], dp[0] - prices[i]); 如果dp[1]取dp[1]，即保持买入股票的状态，那么 dp[2] = max(dp[2], dp[1] + prices[i]);中dp[1] + prices[i] 就是今天卖出。

如果dp[1]取dp[0] - prices[i]，今天买入股票，那么dp[2] = max(dp[2], dp[1] + prices[i]);中的dp[1] + prices[i]相当于是今天再卖出股票，一买一卖收益为0，对所得现金没有影响。相当于今天买入股票又卖出股票，等于没有操作，保持昨天卖出股票的状态了。


在动态规划的状态设计中，`dp[i][4]` 表示的是**最多进行两次完整交易（两次买入+两次卖出）**后的最大利润，而非允许无限次交易。以下是具体原因分析：

---

### 1. **状态定义的严格限制**
代码中定义了 `dp[i][j]` 的 5 种状态，其中：
- `dp[i][0]`：无任何交易。
- `dp[i][1]`：第一次买入后未卖出。
- `dp[i][2]`：第一次卖出后结束第一次交易。
- `dp[i][3]`：第二次买入后未卖出（必须在第一次卖出后才能进行）。
- `dp[i][4]`：第二次卖出后结束第二次交易（必须在第二次买入后才能进行）。

**关键点**：每个状态的转移都严格遵循交易的顺序。例如：
- 要进入第二次买入状态 `dp[i][3]`，必须从第一次卖出状态 `dp[i-1][2]` 转移而来（即必须完成第一次交易后才能进行第二次买入）。
- 要进入第二次卖出状态 `dp[i][4]`，必须从第二次买入状态 `dp[i-1][3]` 转移而来（即必须完成第二次买入后才能卖出）。

这种状态设计从根本上限制了交易次数最多为 **两次完整的买入+卖出**，无法进行第三次交易。

---

### 2. **用户举例的分析**
用户提出的场景：
- 第 1 天买入 → 第 1 天卖出（第一次交易）。
- 第 2 天买入 → 第 2 天卖出（第二次交易）。
- 第 3 天再次买入（第三次交易）。

在代码的状态设计下，这种场景是否可能？

**分析过程**：
1. 第 1 天买入后卖出，状态变为 `dp[1][2]`（第一次卖出）。
2. 第 2 天买入，必须从 `dp[1][2]` 转移到 `dp[2][3]`（第二次买入）。
3. 第 2 天卖出，从 `dp[2][3]` 转移到 `dp[2][4]`（第二次卖出）。
4. 第 3 天若想再次买入，需要从 `dp[2][4]` 转移到新的买入状态。但代码中没有第三次买入的状态（只有 `dp[i][3]` 表示第二次买入），因此**无法进行第三次买入**。

因此，代码的状态设计天然不支持超过两次的交易，用户举例的第三次买入在状态转移中没有对应的状态，因此不可能发生。

---

### 3. **状态转移的顺序保证**
状态转移方程严格按照交易顺序进行：
- 第一次买入（`dp[i][1]`）只能由无交易状态 `dp[i-1][0]` 转移而来。
- 第一次卖出（`dp[i][2]`）只能由第一次买入状态 `dp[i-1][1]` 转移而来。
- 第二次买入（`dp[i][3]`）只能由第一次卖出状态 `dp[i-1][2]` 转移而来。
- 第二次卖出（`dp[i][4]`）只能由第二次买入状态 `dp[i-1][3]` 转移而来。

这种顺序保证了每一次买入必须在前一次卖出之后，且最多进行两次完整的交易。即使某天买入后当天卖出，也会被视为一次完整的交易，后续只能再进行一次交易。

---

### 结论
通过状态定义和转移方程的严格限制，`dp[i][4]` 始终表示**最多两次完整交易**后的最大利润。任何超过两次交易的操作（如用户举例的第三次买入）在状态设计中没有对应的状态，因此不可能出现。
