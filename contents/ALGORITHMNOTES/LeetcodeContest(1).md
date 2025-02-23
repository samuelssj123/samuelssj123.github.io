LeetCode Biweekly Contest 150

# 3452.好数字之和

[LeetCode](https://leetcode.cn/problems/sum-of-good-numbers/description/)

## 审题

- 当`i - k`和`i + k`这两个下标对应的元素都存在时，需要同时满足`nums[i] > nums[i - k]`和`nums[i] > nums[i + k]`，`nums[i]`才是好元素。
  
- 当`i - k`不存在但`i + k`存在时，只需要满足`nums[i] > nums[i + k]`，`nums[i]`就是好元素。
  
- 当`i + k`不存在但`i - k`存在时，只需要满足`nums[i] > nums[i - k]`，`nums[i]`就是好元素。
  
- 当`i - k`和`i + k`都不存在时，`nums[i]`直接就是好元素。
  
```python
class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        good_sum = 0
        for i in range(n):
            # i - k不存在且i + k不存在
            if i < k and i + k >= n:
                good_sum += nums[i]
            # i - k不存在但i + k存在
            elif i < k and nums[i] > nums[i + k]:
                good_sum += nums[i]
            # i + k不存在但i - k存在
            elif i + k >= n and nums[i] > nums[i - k]:
                good_sum += nums[i]
            # i - k和i + k都存在
            elif nums[i] > nums[i - k] and nums[i] > nums[i + k]:
                good_sum += nums[i]
        return good_sum
```

## 简化代码

对于数组 `nums` 中的元素 `nums[i]`，如果它严格大于下标 `i - k` 和 `i + k` 处的元素（前提是这两个下标对应的元素存在），那么 `nums[i]` 就是好元素；若 `i - k` 和 `i + k` 这两个下标都不存在，`nums[i]` 同样被视为好元素。

### 分别分析 `i - k` 和 `i + k` 的情况

#### 分析 `i - k` 的情况
- **`i - k` 不存在**：当 `i < k` 时，`i - k` 为负数，这意味着 `i - k` 位置在数组中没有对应的元素。根据题目规则，此时 `nums[i]` 天然满足关于 `i - k` 位置的要求。
- **`i - k` 存在**：当 `i >= k` 时，`i - k` 是一个有效的数组索引。此时要使 `nums[i]` 满足关于 `i - k` 位置的要求，就需要 `nums[i] > nums[i - k]`。

综合这两种情况，对于 `i - k` 位置的判断可以用逻辑或（`or`）连接起来，得到子条件 `i < k or nums[i] > nums[i - k]`。逻辑或的特点是只要其中一个条件为 `True`，整个子条件就为 `True`，正好符合上述两种情况中只要满足其一即可的逻辑。

#### 分析 `i + k` 的情况
- **`i + k` 不存在**：当 `i + k >= len(nums)` 时，说明 `i + k` 超出了数组的长度范围，即 `i + k` 位置在数组中没有对应的元素。按照题目规则，此时 `nums[i]` 满足关于 `i + k` 位置的要求。
- **`i + k` 存在**：当 `i + k < len(nums)` 时，`i + k` 是一个有效的数组索引。要使 `nums[i]` 满足关于 `i + k` 位置的要求，就需要 `nums[i] > nums[i + k]`。

### 合并两个子条件
由于要同时满足关于 `i - k` 位置和 `i + k` 位置的要求，所以需要用逻辑与（`and`）将上述两个子条件连接起来，最终就得到了判断条件 `(i < k or nums[i] > nums[i - k]) and (i + k >= len(nums) or nums[i] > nums[i + k])`。逻辑与的特点是只有当两个子条件都为 `True` 时，整个判断条件才为 `True`，这与题目要求的同时满足两个位置的条件相契合。

```python
class Solution:
    def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
        ans = 0
        for i in range(len(nums)):
            if (i < k or nums[i] > nums[i - k]) and (i + k >= len(nums) or nums[i] > nums[i + k]):
                ans += nums[i]
        return ans
```


# 3453. 分割正方形Ⅰ

[Leetcode](https://leetcode.cn/problems/separate-squares-i/description/)

## 方法一：浮点二分

![image](../images/3453-separate-squares-i-1.png)

```python
class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        M = 100000
        S = sum(l * l for xi, yi, l in squares)

        def check(y: float) -> bool:
            area = 0
            for xi, yi, l in squares:
                if yi < y:
                    area += l * min(y - yi, l)
            return area >= S / 2
        
        left = 0
        right = maxy = max(y + l for _, y, l in squares)
        for i in range((maxy * M).bit_length()):
            mid = (left + right) / 2
            if check(mid):
                right = mid
            else:
                left = mid
        return (left + right) / 2  # 区间中点误差小
```

## 方法二： 整数二分

![image](../images/3453-separate-squares-i-2.png)

### 写法一：

```python
class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        M = 100000
        S = sum(l * l for xi, yi, l in squares)

        def check(y: int) -> bool:
            area = 0
            for xi, yi, l in squares:
                if yi * M < y:
                    area += l * min(y - yi * M, l * M)
            return area >= (S * M) / 2
        
        maxy = max(y + l for _, y, l in squares)
        return bisect_left(range(maxy * M), True, key = check) / M
# return bisect_left(range(maxy * M), True, key = check) / M 这行代码的整体作用是利用二分查找算法，在所有可能的分割线位置中，找到满足分割线下方正方形面积之和至少为总面积一半的最小分割线位置，并将其转换为浮点数形式返回。
```

### 写法二：

```python
class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        def calcarea(y: int) -> int:
            area = 0
            for xi, yi, l in squares:
                if yi < y:
                    area += l * min(y - yi, l)
            return area
                    
        S = sum(l * l for xi, yi, l in squares)
        maxy = max(y + l for _, y, l in squares)
        y = bisect_left(range(maxy), S, key = lambda y : calcarea(y) * 2) 

        area_y = calcarea(y)
        sum_l = area_y - calcarea(y - 1)
        return y - (area_y * 2 - S) / (sum_l * 2)
```

### 写法三：

```python
class Solution:
    def separateSquares(self, squares: List[List[int]]) -> float:
        S = 0
        diff = defaultdict(int) # 使用 defaultdict(int) 创建一个差分数组，用于记录每个高度上正方形边长的变化。
        for xi, yi, l in squares: #遍历正方形并更新差分数组和总面积
            S += l * l 
            diff[yi] += l  #在差分数组中，将当前正方形底部的高度 yi 处的边长增加 l，表示从该高度开始有新的正方形出现。
            diff[yi + l] -= l  #在差分数组中，将当前正方形顶部的高度 yi + l 处的边长减少 l，表示该正方形在该高度结束。

        area = sum_l = 0
        for y, y2 in pairwise(sorted(diff)): #pairwise(sorted(diff))：对差分数组的键进行排序，然后使用 pairwise 函数生成相邻元素对。
            sum_l += diff[y] #更新当前高度上的总边长，加上差分数组中当前高度 y 处的边长变化。
            area += sum_l * (y2 - y) #计算当前高度区间 [y, y2) 内新增的面积（总边长乘以高度差），并累加到总面积 area 中。
            if area * 2 >= S:
                return y2 - (area * 2 - S) / (sum_l * 2) 
```

- 差分数组

差分数组是一种用于高效处理数组区间操作的数据结构。作用是：

**快速区间更新**：在一些需要对数组的某个区间进行频繁更新操作的场景中，差分数组能发挥很大的作用。比如要对原数组`arr`的`[l, r]`区间内的所有元素都加上一个值`x`，如果直接操作原数组，时间复杂度为$O(r - l + 1)$。但利用差分数组，只需要让`diff[l] += x`，`diff[r + 1] -= x`，时间复杂度为$O(1)$。之后通过对差分数组求前缀和就可以得到更新后的原数组。

本题中，差分数组能够记录每个高度上正方形边长的变化情况，并据此计算不同高度区间内的面积。

**底部增加边长**：当在正方形的底部高度 `yi` 处执行 `diff[yi] += l` 时，这意味着在这个高度上开始有一个边长为 `l` 的正方形出现，记录下这个增加量，就可以知道在该高度有新的正方形覆盖进来，边长总和增加了 `l`。

**顶部减少边长**：在正方形的顶部高度 `yi + l` 处执行 `diff[yi + l] -= l`，表示在这个高度上之前在 `yi` 开始的正方形结束了，所以要把对应的边长 `l` 减掉。这样，差分数组中每个键值对就清晰地反映了在对应高度上正方形边长的变化情况，即哪些高度有正方形开始，哪些高度有正方形结束，从而整体上能反映出所有正方形在不同高度的覆盖情况。


# 3454. 分割正方形Ⅱ

[Leetcode](https://leetcode.cn/problems/separate-squares-ii/solutions/3078402/lazy-xian-duan-shu-sao-miao-xian-pythonj-eeqk/)

和上个题目不同的是：正方形 可能会 重叠。重叠区域只 统计一次 。

首先用扫描线方法，求出所有正方形的面积并`totArea`。然后再次扫描，设扫描线下方的面积和为 `area`，那么扫描线上方的面积和为 `totArea - area`。题目要求`area = totArea - area`， 即`area * 2 = totArea`。

设当前扫描线的纵坐标为 `y`，下一个需要经过的正方形上/下边界的纵坐标为 `y'`，被至少一个正方形覆盖的底边长之和为 `sumLen`，那么新的面积和为`area + sumLen * (y' - y)`

如果发现`(area + sumLen * (y' - y)) * 2 >= totArea`，取等号，解得`y' = y + (totalArea / 2 - area) / sumL = y + (totalArea - area * 2) / (sumL * 2)`即为答案。

- 编程技巧：把第一次扫描过程中的关键数据 `area` 和 `sumLen` 记录到一个数组中，然后遍历数组（或者二分），这样可以避免跑两遍线段树（空间换时间）。 


# 3455.最短匹配子字符串

[Leetcode](https://leetcode.cn/problems/shortest-matching-substring/description/)

核心思路：KMP+三指针

KMP的改进：记录每个匹配子串在p中的所有位置。

三指针：遍历p2，寻找最邻近的p1、p3的位置。

## 写代码时的注意点

1. **为什么先找 `pos3` 而不是 `pos1`**：
   代码中先找 `pos3` 而不是 `pos1` 主要是一种实现顺序的选择，从逻辑上来说先找哪一个都是可以的。在这里先找 `pos3` 的原因可能是在当前的枚举逻辑下，先确定右边（`pos3` 代表的 `p3` 的匹配位置）的合适位置，再去确定左边（`pos1` 代表的 `p1` 的匹配位置）的合适位置，这样在后续的条件判断和计算中会更加清晰和方便。因为在枚举 `p2` 的匹配位置 `j` 时，先确定 `pos3` 中不与 `p2` 重叠的位置，再根据这个位置去确定 `pos1` 中合适的位置，能够更准确地计算出满足条件的子字符串长度。但这只是一种实现方式，先找 `pos1` 然后再找 `pos3` 理论上也能实现相同的功能，只是后续的条件判断和计算顺序可能会有所不同。

2. **为什么找不重叠的不是 `pos3[k] < j` 而是 `pos3[k] < j + len(p2)`**：
   因为 `j` 是 `p2` 在 `s` 中匹配的起始位置，`len(p2)` 是 `p2` 的长度。要判断 `pos3` 中 `p3` 的匹配位置与 `p2` 的匹配位置不重叠，需要判断 `p3` 的起始位置（`pos3[k]`）是否在 `p2` 的结束位置（`j + len(p2)`）之前。如果只判断 `pos3[k] < j`，那么就只考虑了 `p3` 的起始位置是否在 `p2` 的起始位置之前，而没有考虑到 `p2` 本身是有长度的，可能会导致 `p3` 的一部分与 `p2` 重叠却没有被检测到。所以使用 `pos3[k] < j + len(p2)` 才能准确判断 `p3` 的匹配位置与 `p2` 的匹配位置不重叠。

3. **为什么 `pos1` 不与 `p2` 重叠判断条件是 `pos1[i] <= j - len(p1)` 而不是 `pos1[i] <= j - len(p2)`，和上面哪个有什么区别**：
   `pos1` 存储的是 `p1` 在 `s` 中匹配的位置，要判断 `pos1` 中 `p1` 的匹配位置与 `p2` 的匹配位置不重叠，应该考虑 `p1` 的长度。`pos1[i]` 是 `p1` 匹配的起始位置，`j` 是 `p2` 匹配的起始位置，`len(p1)` 是 `p1` 的长度。`pos1[i] <= j - len(p1)` 表示 `p1` 的结束位置（`pos1[i] + len(p1)`）要小于等于 `p2` 的起始位置 `j`，这样才能保证 `p1` 和 `p2` 不重叠。
   而 `pos1[i] <= j - len(p2)` 是错误的，因为 `p2` 的长度与判断 `p1` 和 `p2` 是否重叠没有直接关系，应该使用 `p1` 的长度来进行判断。与判断 `pos3` 不与 `p2` 重叠的条件 `pos3[k] < j + len(p2)` 的区别在于，一个是判断左边（`p1`）与中间（`p2`）不重叠，一个是判断右边（`p3`）与中间（`p2`）不重叠，并且判断条件的符号和使用的长度变量不同是根据它们的位置关系和匹配逻辑决定的。

4. **为什么 `ans = min(ans, pos3[k] + len(p3) - pos1[i - 1])` 是 `i - 1`**：
   在 `while i < len(pos1) and pos1[i] <= j - len(p1):` 这个循环中，当循环结束时，`i` 的值是使得 `pos1[i] > j - len(p1)` 的第一个索引。也就是说，`pos1[i - 1]` 是满足 `pos1[i - 1] <= j - len(p1)` 的最后一个位置，即离 `j` 最近且不与 `p2` 重叠的 `p1` 的匹配位置。要计算满足条件的子字符串的长度，子字符串的起始位置是 `pos1[i - 1]`，结束位置是 `pos3[k] + len(p3) - 1`（因为 `pos3[k]` 是 `p3` 的起始位置，加上 `p3` 的长度得到结束位置），所以子字符串的长度为 `pos3[k] + len(p3) - pos1[i - 1]`，使用 `i - 1` 才能正确计算出基于找到的合适的 `p1` 和 `p3` 匹配位置的子字符串长度。

5. **为什么找到了的终止条件是 `i > 0`？ 为什么判断 `pos3` 没有找到和 `pos1` 找到，而不是判断 `pos3` 找到和 `pos1` 没有找到**：
   - `i > 0` 作为找到的终止条件是因为在 `while i < len(pos1) and pos1[i] <= j - len(p1):` 循环中，当 `i` 从 `0` 开始遍历 `pos1`，如果 `i` 最终大于 `0`，说明在 `pos1` 中找到了满足 `pos1[i] <= j - len(p1)` 的位置，即找到了合适的 `p1` 的匹配位置。如果 `i` 一直没有增加（`i = 0`），则说明没有找到合适的 `p1` 的匹配位置。
   - 判断 `pos3` 没有找到（`k == len(pos3)`）和 `pos1` 找到（`i > 0`）是因为在当前的枚举逻辑下，对于一个给定的 `p2` 的匹配位置 `j`，先找 `pos3` 中合适的位置，如果 `k` 遍历完 `pos3` 都没有找到合适的位置（`k == len(pos3)`），那么对于这个 `j` 就不可能找到满足条件的子字符串了，因为右边（`p3`）没有合适的匹配位置。而判断 `pos1` 找到（`i > 0`）是为了确保左边（`p1`）也有合适的匹配位置，这样才能组成一个满足模式字符串 `p` 的子字符串。如果判断 `pos3` 找到和 `pos1` 没有找到，那么就无法组成满足模式的子字符串，因为缺少了 `p1` 的合适匹配位置，所以这种判断方式不符合算法的逻辑。

6. **`kmp` 算法和传统的 `kmp` 算法有什么不同**：
   这段代码中的 `kmp` 算法本质上与传统的 KMP 算法是一致的，都是利用部分匹配值（`pi` 数组）来减少字符串匹配过程中的回溯次数，提高匹配效率。不同之处可能在于具体的实现细节和代码组织方式：
   - 在代码中，`kmp` 方法将计算 `pi` 数组的操作封装在 `calc` 方法中，并且在 `kmp` 方法中调用 `calc` 方法来获取 `pi` 数组，这是一种代码组织方式的不同。传统的 KMP 算法实现中，计算 `pi` 数组的代码可能会直接写在匹配过程的代码中，而不是单独封装成一个方法。
   - 对于空模式字符串的处理，代码中 `kmp` 方法在 `p` 为空时返回 `list(range(len(s) + 1))`，认为 `s` 的所有位置都能匹配空串，这是一种特定的处理方式，不同的实现可能会有不同的处理逻辑，比如有些实现可能会直接返回空列表或者抛出异常等。但这些差异都是在具体实现上的一些变化，核心的 KMP 算法思想和匹配逻辑是相同的。 




```python
class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        p1, p2, p3 = p.split('*') ## 将模式字符串 p 按照 * 分割成三段，分别为 * 之前的部分 p1，两个 * 之间的部分 p2，* 之后的部分 p3
        
        # 使用 KMP 算法在字符串 s 中查找 p1 出现的所有位置，结果存储在 pos1 中
        pos1 = self.kmp(s, p1) 
        pos2 = self.kmp(s, p2)
        pos3 = self.kmp(s, p3)

        ans = inf # 初始化最短子字符串长度
        i = k = 0 # 初始化最短子字符串长度

        # 枚举 p2 在 s 中匹配的位置 j
        for j in pos2:
            while k < len(pos3) and pos3[k] < j + len(p2): # 寻找 pos3 中离 j 最近且不与 p2 重叠（即 pos3[k] >= j + len(p2)）的匹配位置，k 向后移动
                k += 1
            if k == len(pos3): 
                break # 如果已经遍历完 pos3 都没有找到合适的位置，说明对于当前的 j 没有满足条件的解，跳出循环
            while i < len(pos1) and pos1[i] <= j - len(p1):
                i += 1
            if i > 0: # 如果找到了合适的 pos1 中的位置（i > 0 表示找到了）
                ans = min(ans, pos3[k] + len(p3) - pos1[i - 1])
        return -1 if ans == inf else ans
    
    def getNext(self, p:str) -> List[int]:
        next = [0] * len(p) 
        j = 0 
        for i in range(1, len(p)): 
            while j > 0 and p[j] != p[i]: 
                j = next[j - 1]
            if p[i] == p[j]: 
                j += 1
            next[i] = j
        return next

    def kmp(self, s:str, p:str) -> List[int]:
        # 如果模式字符串 p 为空，认为 s 的所有位置都能匹配空串，返回所有位置的索引列表
        if not p: 
            return list(range(len(s) + 1))
        
        next = self.getNext(p)
        pos = [] # 用于存储匹配位置的列表
        j = 0 

        for i in range(len(s)):
            while j > 0 and s[i] != p[j]: 
                j = next[j - 1]
            if s[i] == p[j]: 
                j += 1
            if j == len(p): 
                pos.append(i - j + 1)
                j = next[j - 1] 
        return pos
```







题解参考：灵茶山艾府
