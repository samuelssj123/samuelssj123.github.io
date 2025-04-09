List: dijkstra（朴素版）精讲，dijkstra（堆优化版）精讲，Bellman_ford 算法精讲

[dijkstra（朴素版）精讲](#01)，[dijkstra（堆优化版）精讲](#02)，[Bellman_ford 算法精讲](#03)

# <span id="01">dijkstra（朴素版）精讲</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1047) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0047.%E5%8F%82%E4%BC%9Adijkstra%E6%9C%B4%E7%B4%A0.html#%E6%80%9D%E8%B7%AF)

![image](../images/GraphTheory(8)-1.png)

```python
n, m = map(int, input().split())
grid = [[float('inf')] * (n + 1) for _ in range(n + 1)]
for i in range(m):
    p1, p2, val = map(int, input().split())
    grid[p1][p2] = val
minDist = [float('inf')] * (n + 1)
visited = [False] * (n + 1)

minDist[1] = 0 
for i in range(n + 1):
    minval = float('inf')
    cur = 1
    
    for v in range(n + 1):
        if not visited[v] and minDist[v] < minval:
            minval = minDist[v]
            cur = v 
    
    visited[cur] = True

    for v in range(n + 1):
        if not visited[v] and grid[cur][v] != float('inf') and minDist[cur] + grid[cur][v] < minDist[v]:
            minDist[v] = minDist[cur] + grid[cur][v]

if minDist[n] == float('inf'):
    print(-1)
else:
    print(minDist[n])
```

# <span id="02">dijkstra（堆优化版）精讲</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1047) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0047.%E5%8F%82%E4%BC%9Adijkstra%E5%A0%86.html)

![image](../images/GraphTheory(8)-2.png)

```python
import heapq
class Edge:
    def __init__(self, to, val):
        self.to = to
        self.val = val
n, m = map(int, input().split())
grid = [[] for _ in range(n + 1)]
for i in range(m):
    p1, p2, val = map(int, input().split())
    grid[p1].append(Edge(p2, val))
minDist = [float('inf')] * (n + 1)
visited = [False] * (n + 1)
pq = []

heapq.heappush(pq, (0, 1))
minDist[1] = 0 

while pq:
    curdist, curnode = heapq.heappop(pq)
    if visited[curnode]:
        continue
    visited[curnode] = True
    for edge in grid[curnode]:
        if not visited[edge.to] and curdist + edge.val < minDist[edge.to]:
            minDist[edge.to] = curdist + edge.val
            heapq.heappush(pq, (minDist[edge.to], edge.to))

if minDist[n] == float('inf'):
    print(-1)
else:
    print(minDist[n])
```

### 代码思路概述
你所写的代码采用了优先队列（堆）优化的 Dijkstra 算法来解决单源最短路径问题，目标是找出从起点到终点的最短路径。Dijkstra 算法的核心思想是基于贪心策略，逐步扩展距离起点最近的节点，直到找到到达所有节点的最短路径。使用优先队列（堆）可以优化每次选择距离起点最近节点的时间复杂度，从朴素 Dijkstra 算法的 $O(V^2)$ 降低到 $O((V + E)\log V)$，其中 $V$ 是节点数量，$E$ 是边的数量。

### 详细步骤解释
1. **输入处理与图的构建**：
    - 读取节点数量 `n` 和边的数量 `m`。
    - 创建一个邻接表 `grid` 来存储图的结构，每个节点对应一个列表，列表中存储从该节点出发的边。
    - 读取每条边的信息，包括起点 `p1`、终点 `p2` 和边的权重 `val`，并将边的信息存储在邻接表中。

2. **初始化**：
    - `minDist` 数组：用于记录从起点到每个节点的最短距离，初始值设为无穷大 `float('inf')`。
    - `visited` 数组：用于标记每个节点是否已经被访问过，初始值为 `False`。
    - 优先队列 `pq`：用于存储待处理的节点及其距离起点的最短距离，初始为空。

3. **起点入队**：
    - 将起点（编号为 1）的距离设为 0，并将 `(0, 1)` 元组加入优先队列 `pq` 中。这里的 `0` 表示起点到自身的距离为 0，`1` 表示起点的编号。

4. **优先队列处理**：
    - 当优先队列 `pq` 不为空时，执行以下操作：
        - 从优先队列中取出距离起点最近的节点 `curnode` 及其对应的最短距离 `curdist`。
        - 如果该节点已经被访问过（`visited[curnode]` 为 `True`），则跳过该节点，继续处理下一个节点。
        - 标记该节点为已访问（`visited[curnode] = True`）。
        - 遍历该节点的所有邻接边 `edge`：
            - 如果邻接节点 `edge.to` 未被访问过，并且通过当前节点到达邻接节点的距离 `curdist + edge.val` 比之前记录的最短距离 `minDist[edge.to]` 更小，则更新 `minDist[edge.to]` 的值。
            - 将更新后的最短距离和邻接节点的编号组成的元组 `(minDist[edge.to], edge.to)` 加入优先队列 `pq` 中。

5. **输出结果**：
    - 最后检查 `minDist[n]` 的值，如果仍然为无穷大 `float('inf')`，说明无法从起点到达终点，输出 `-1`；否则，输出 `minDist[n]`，即从起点到终点的最短距离。

### 堆里面放的内容
堆（优先队列 `pq`）里面存放的是元组 `(minDist[edge.to], edge.to)`，其中：
- `minDist[edge.to]`：表示从起点到节点 `edge.to` 当前所记录的最短距离。这个值用于确定节点在堆中的优先级，堆会根据这个值进行排序，使得距离起点最近的节点总是位于堆的顶部。
- `edge.to`：表示目标节点的编号。在取出堆中的元素时，除了要知道最短距离，还需要知道这个距离对应的是哪个节点，所以将节点编号也作为元组的一部分存储在堆中。

### 存放的顺序
堆是一种完全二叉树，并且满足堆的性质：对于最小堆来说，每个节点的值都小于或等于其子节点的值。在优先队列 `pq` 中，元素按照 `minDist[edge.to]` 的值进行排序，即距离起点最近的节点位于堆的顶部。每次从堆中取出元素时，都会取出距离起点最近的节点进行处理，然后将更新后的节点信息重新加入堆中，堆会自动调整元素的顺序，以保证堆的性质不变。这样可以确保每次处理的节点都是当前距离起点最近的节点，从而实现 Dijkstra 算法的贪心策略。


### 能不能把 (minDist[edge.to], edge.to) 的两个元素调换顺序呢？不能！

#### 1. 入堆元素顺序错误
在代码里，`heapq.heappush(pq, (1, 0))` 这一行把节点编号和距离的顺序弄反了。在优先队列中，我们需要依据距离来进行排序，所以距离应该放在元组的第一个位置，节点编号放在第二个位置。正确的写法应该是 `heapq.heappush(pq, (0, 1))`。

#### 2. 后续入堆元素顺序错误
在循环里，`heapq.heappush(pq, (edge.to, minDist[edge.to]))` 同样把节点编号和距离的顺序弄反了。正确的写法应该是 `heapq.heappush(pq, (minDist[edge.to], edge.to))`。

### 修正后的代码

```python
import heapq

class Edge:
    def __init__(self, to, val):
        self.to = to
        self.val = val

n, m = map(int, input().split())
# 构建邻接表来存储图
grid = [[] for _ in range(n + 1)]
# 读取边的信息
for i in range(m):
    p1, p2, val = map(int, input().split())
    grid[p1].append(Edge(p2, val))

# 初始化最短距离数组，初始值设为无穷大
minDist = [float('inf')] * (n + 1)
# 初始化访问标记数组，初始值为 False
visited = [False] * (n + 1)
# 初始化优先队列
pq = []

# 将起点加入优先队列，距离为 0
heapq.heappush(pq, (0, 1))
# 起点到自身的距离为 0
minDist[1] = 0

while pq:
    # 从优先队列中取出距离最小的节点及其距离
    curdist, curnode = heapq.heappop(pq)
    # 如果该节点已经被访问过，跳过
    if visited[curnode]:
        continue
    # 标记该节点为已访问
    visited[curnode] = True
    # 遍历该节点的所有邻接边
    for edge in grid[curnode]:
        # 如果邻接节点未被访问且通过当前节点到达邻接节点的距离更短
        if not visited[edge.to] and curdist + edge.val < minDist[edge.to]:
            # 更新最短距离
            minDist[edge.to] = curdist + edge.val
            # 将更新后的距离和邻接节点加入优先队列
            heapq.heappush(pq, (minDist[edge.to], edge.to))

# 输出结果
if minDist[n] == float('inf'):
    print(-1)
else:
    print(minDist[n])

```

### 代码解释

1. **入堆元素顺序**：优先队列 `pq` 中的元素是元组，元组的第一个元素是距离，第二个元素是节点编号。这样做是为了让堆依据距离进行排序，保证每次从堆中取出的节点都是距离起点最近的节点。

2. **Dijkstra 算法核心逻辑**：
    - 从优先队列中取出距离最小的节点。
    - 若该节点已被访问过，跳过。
    - 标记该节点为已访问。
    - 遍历该节点的所有邻接边，若通过当前节点到达邻接节点的距离更短，则更新最短距离并将更新后的信息加入优先队列。

通过上述修正，代码就能正确求解从起点到终点的最短路径。 

# <span id="03">Bellman_ford 算法精讲</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1152) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0094.%E5%9F%8E%E5%B8%82%E9%97%B4%E8%B4%A7%E7%89%A9%E8%BF%90%E8%BE%93I.html)

![image](../images/GraphTheory(8)-3.png)

```python
n, m = map(int, input().split())
grid = []
minDist = [float('inf')] * (n + 1)
for i in range(m):
    p1, p2, val = map(int, input().split())
    grid.append([p1, p2, val])

minDist[1] = 0
for i in range(1, n):
    updated = False #没更新则终止，防止超时
    for side in grid:
        if minDist[side[0]] != float('inf') and minDist[side[1]] > minDist[side[0]] + side[2]:
            minDist[side[1]] = minDist[side[0]] + side[2]
            updated = True
    if not updated:  # 若边不再更新，即停止回圈
            break

if minDist[n] == float('inf'):
    print('unconnected')
else:
    print(minDist[n])
```


### 超时原因

#### 1. 算法复杂度问题
Bellman - Ford 算法的时间复杂度为 $O(VE)$，其中 $V$ 是节点数（城市数量），$E$ 是边数（道路数量）。在最坏情况下，需要对每条边进行 $V - 1$ 次松弛操作。当 $V$ 和 $E$ 较大时，算法的执行时间会显著增加，导致超时。

#### 2. 代码结构问题
代码中使用了嵌套循环，外层循环执行 $n$ 次，内层循环遍历所有的边。这种结构使得代码的执行时间随着节点数和边数的增加而迅速增长。

### 优化建议

#### 提前终止条件
在 Bellman - Ford 算法中，如果某一轮松弛操作没有对任何距离进行更新，说明已经找到了最短路径，可以提前终止算法，减少不必要的计算。

在代码中添加 `updated = False` 这一行，是为了实现 **Bellman-Ford 算法的提前终止优化**，避免不必要的循环计算。以下是具体原因和作用的详细解释：


### 一、Bellman-Ford 算法的特性
Bellman-Ford 算法通过 **松弛操作**（Relaxation）逐步更新从起点到所有节点的最短距离，最多需要进行 `n-1` 次松弛（`n` 是节点数）。  
但如果在某一轮松弛操作中，**没有任何节点的距离被更新**，说明最短路径已经全部确定（后续轮次也不会再更新），可以提前终止算法，无需执行完所有 `n-1` 次循环。


### 二、`updated = False` 的作用
1. **标记本轮是否有更新**：  
   在每一轮松弛操作开始前，将 `updated` 设为 `False`。如果在遍历所有边的过程中，有任何节点的距离被更新（即松弛成功），就将 `updated` 设为 `True`。

2. **提前终止循环**：  
   每轮松弛结束后，检查 `updated`。如果为 `False`（表示本轮没有任何更新），说明最短路径已稳定，直接跳出循环，不再执行后续轮次。


### 三、为什么必须加这一行？
#### 1. 避免无效计算，优化时间复杂度
- 原始代码没有提前终止逻辑，会强制执行 `n-1` 次循环（即使最短路径早已确定）。  
- 当图的边数 `m` 很大时，这会导致大量无效计算，甚至超时（如你之前遇到的问题）。  
- 添加 `updated` 后，算法会在最短路径稳定后立即终止，平均时间复杂度大幅降低。

#### 2. 保证算法正确性
- Bellman-Ford 算法的性质决定：若某一轮没有松弛成功，后续轮次也不会松弛成功（因为最短路径最多经过 `n-1` 条边，且每次松弛至少缩短一条路径）。  
- `updated` 标记确保算法在正确的时机终止，不影响最终结果的正确性。


### 四、类比理解
假设你要从起点出发，向所有可达节点传播最短距离：  
- 每轮松弛相当于“告诉所有节点：如果通过我能到达你的距离更近，就更新你的距离”。  
- 当某一轮“喊话”后，没有任何节点更新距离，说明“所有人都已经知道最短距离了，不用再喊了”，可以提前结束。  
- `updated = False` 就像一个“是否有人听讲并更新”的记录，避免无效喊话。


### 五、代码中的具体逻辑
```python
for i in range(1, n):
    updated = False  # 初始化标记：本轮是否有更新
    for side in grid:
        # 松弛操作：如果通过 side[0] 到 side[1] 的距离更近，就更新
        if minDist[side[0]] != float('inf') and minDist[side[1]] > minDist[side[0]] + side[2]:
            minDist[side[1]] = minDist[side[0]] + side[2]
            updated = True  # 只要有一次更新，标记为 True
    if not updated:  # 本轮没有任何更新，提前终止
        break
```


### 总结
`updated = False` 是 Bellman-Ford 算法的关键优化手段，作用是：  
1. **减少无效循环**：避免执行不必要的松弛操作，提升效率。  
2. **保持算法正确性**：利用最短路径的性质，确保提前终止不影响结果。  
3. **解决超时问题**：在数据量大时，通过提前终止避免时间超限。

这一行代码是 Bellman-Ford 算法的标准优化步骤，也是解决你之前代码超时的核心改动。
