List: Floyd 算法精讲，A * 算法精讲 （A star算法），最短路算法总结篇，图论总结篇

[Floyd 算法精讲](#01)，[A * 算法精讲 （A star算法）](#02)，[最短路算法总结篇](#03)，[图论总结篇](#04)

# <span id="01">Floyd 算法精讲</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1155) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0097.%E5%B0%8F%E6%98%8E%E9%80%9B%E5%85%AC%E5%9B%AD.html#%E6%80%9D%E8%B7%AF)

![image](../images/GraphTheory(10)-1.png)

```python
n, m = map(int, input().split())
grid = [[[float('inf')] * (n + 1) for _ in range(n+1)] for _ in range(n+1)]

for i in range(m):
    p1, p2, val = map(int, input().split())
    grid[p1][p2][0] = val
    grid[p2][p1][0] = val

for k in range(1, n + 1):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            grid[i][j][k] = min(grid[i][j][k - 1], grid[i][k][k - 1] + grid[k][j][k - 1])

z = int(input())
while z:
    start, end = map(int, input().split())
    if grid[start][end][n] == float('inf'):
        print(-1)
    else:
        print(grid[start][end][n])
    z -= 1
```

## 空间优化：

```python
n, m = map(int, input().split())
grid = [[float('inf')] * (n + 1) for _ in range(n+1)]

for i in range(m):
    p1, p2, val = map(int, input().split())
    grid[p1][p2] = val
    grid[p2][p1] = val

for k in range(1, n + 1):
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            grid[i][j] = min(grid[i][j], grid[i][k] + grid[k][j])

z = int(input())
while z:
    start, end = map(int, input().split())
    if grid[start][end] == float('inf'):
        print(-1)
    else:
        print(grid[start][end])
    z -= 1
```

# <span id="02">A * 算法精讲 （A star算法）</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1203) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0126.%E9%AA%91%E5%A3%AB%E7%9A%84%E6%94%BB%E5%87%BBastar.html)

![image](../images/GraphTheory(10)-2.png)

```python
# 初始化移动步数矩阵
moves = [[0] * 1001 for _ in range(1001)]
# 骑士的八个移动方向
direction = [(1, 2), (2, 1), (-1, 2), (2, -1), (1, -2), (-2, 1), (-1, -2), (-2, -1)]
class Knight:
    def __init__(self, x, y, g, h, f):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = f
    # 重载小于运算符，用于优先队列排序
    def __lt__(self, other):
        return self.f < other.f


# 使用 Python 的 heapq 模块实现优先队列，而不是 deque

# 启发式函数，计算当前位置到目标位置的启发式距离
def Heuristic(k, b1, b2):
    heuristic = (k.x - b1) * (k.x - b1) + (k.y - b2) * (k.y - b2) # 不开根号，提高运算效率
    return heuristic

# A*算法实现
import heapq
def astar(k, b1, b2):
    que = []
    heapq.heappush(que, k)
    while que:
        cur = heapq.heappop(que)
        if cur.x == b1 and cur.y == b2:
            break
        for i in range(8):
            nxx = cur.x + direction[i][0]
            nxy = cur.y + direction[i][1]
            nx = Knight(nxx, nxy, 0, 0, 0)
            if nx.x < 1 or nx.x > 1000 or nx.y < 1 or nx.y > 1000:
                continue
            if not moves[nx.x][nx.y]:
                moves[nx.x][nx.y] = moves[cur.x][cur.y] + 1
                nx.g = cur.g + 5 #不开根号，1*1+2*2 = 5
                nx.h = Heuristic(nx, b1, b2)
                nx.f = nx.g + nx.h 
                heapq.heappush(que, nx)

n = int(input())
while n:
    n -= 1
    a1, a2, b1, b2 = map(int, input().split())
    # 重置移动步数矩阵
    moves = [[0] * 1001 for _ in range(1001)]
    start = Knight(a1, a2, 0, 0, 0)
    start.h = Heuristic(start, b1, b2)
    start.f = start.g + start.h 
    astar(start, b1, b2)
    print(moves[b1][b2])
```

# <span id="03">最短路算法总结篇</span>


[Learning Materials](https://www.programmercarl.com/kamacoder/%E6%9C%80%E7%9F%AD%E8%B7%AF%E9%97%AE%E9%A2%98%E6%80%BB%E7%BB%93%E7%AF%87.html)

![image](../images/GraphTheory(10)-3.png)

如果遇到单源且边为正数，直接Dijkstra。

至于 使用朴素版还是 堆优化版 还是取决于图的稠密度， 多少节点多少边算是稠密图，多少算是稀疏图，这个没有量化，如果想量化只能写出两个版本然后做实验去测试，不同的判题机得出的结果还不太一样。

一般情况下，可以直接用堆优化版本。

如果遇到单源边可为负数，直接 Bellman-Ford，同样 SPFA 还是 Bellman-Ford 取决于图的稠密度。

一般情况下，直接用 SPFA。

如果有负权回路，优先 Bellman-Ford， 如果是有限节点最短路 也优先 Bellman-Ford，理由是写代码比较方便。

如果是遇到多源点求最短路，直接 Floyd。

除非 源点特别少，且边都是正数，那可以 多次 Dijkstra 求出最短路径，但这种情况很少，一般出现多个源点了，就是想让你用 Floyd 了。

对于A * ，由于其高效性，所以在实际工程应用中使用最为广泛 ，由于其 结果的不唯一性，也就是可能是次短路的特性，一般不适合作为算法题。

游戏开发、地图导航、数据包路由等都广泛使用 A * 算法。

# <span id="04">图论总结篇</span>


[Learning Materials](https://www.programmercarl.com/kamacoder/%E5%9B%BE%E8%AE%BA%E6%80%BB%E7%BB%93%E7%AF%87.html)



### 一段话总结

图论核心知识，涵盖深搜广搜（需掌握搜索方式、代码模板、应用场景，注意DFS两种写法及标记细节）、并查集（解决集合合并与查询问题，需理解原理、路径压缩及应用场景）、最小生成树（Prim适用于稠密图，Kruskal适用于稀疏图，后者结合并查集）、拓扑排序（两步法找入度为0节点并移除，可检测有向图环）及最短路算法（复杂且需结合场景选择），强调图的存储方式（邻接表、邻接矩阵）及各算法核心逻辑与适用场景。


### 详细总结
#### 一、深搜与广搜
1. **核心要点**  
   - **搜索方式**：深搜（DFS）是“一条路走到底”的深度优先搜索，广搜（BFS）是“逐层扩展”的广度优先搜索。  
   - **代码模板**：DFS有两种写法（处理当前节点或下一个节点），取决于是否需要回溯；BFS需在节点入队时标记已访问，避免重复入队。  
   - **应用场景**：适用于路径计算（如0098.所有可达路径）、染色问题（如0099.岛屿的数量）等，需根据问题选择效率更高的方式。  
   - **注意事项**：DFS可能需要回溯（如路径记录），而染色问题无需显式回溯；BFS超时常因标记时机错误（应在入队时标记而非出队时）。

#### 二、并查集
1. **理论基础**  
   - **核心功能**：高效处理集合的合并与查询，解决“是否属于同一集合”问题，优于二维数组或Map。  
   - **关键操作**：路径压缩（优化查找效率）和按秩合并（避免树退化），时间复杂度接近O(1)。  
2. **应用场景**  
   - 判断图是否为树（如0108.冗余连接）：树需满足“n-1条边且无环”，并查集可检测环。  
   - 处理有向树的复杂情况（如0109.冗余连接II）：需考虑多父节点等特殊情况。  
3. **常见误区**：路径压缩是递归或迭代更新父节点，确保树的高度最小化。

#### 三、最小生成树
| 算法       | 核心思想                | 适用场景   | 时间复杂度 | 数据结构       |
|------------|-------------------------|------------|------------|----------------|
| **Prim**   | 维护节点集合，选距离最小节点 | 稠密图     | O(n²)      | 邻接矩阵       |
| **Kruskal** | 边排序+并查集，选最小无环边 | 稀疏图     | O(m log m) | 邻接表+并查集  |
- **Prim三部曲**：  
  1. 选距离生成树最近的节点；  
  2. 将节点加入生成树；  
  3. 更新非生成树节点到生成树的距离（`minDist`数组是灵魂）。  
- **Kruskal核心**：利用并查集判断边是否成环，按权值从小到大添加边。

#### 四、拓扑排序
1. **定义与作用**  
   - 将有向图转换为线性排序，若存在环则无法排序，可用于检测循环依赖（如大学排课、文件下载依赖）。  
2. **步骤**  
   1. 统计所有节点的入度，找到入度为0的节点加入结果集；  
   2. 移除该节点及其所有出边，重复直至无入度为0节点（若节点未全处理则存在环）。  

#### 五、最短路算法
- **复杂性**：需根据图的权值（是否含负权）、稀疏程度选择算法（如Dijkstra适用于非负权图，Bellman-Ford适用于含负权图）。  
- **核心逻辑**：不同算法各有优化，如SPFA（队列优化的Bellman-Ford）可处理稀疏图的负权问题。

#### 六、图的存储方式
- **邻接表**：数组存储节点，每个节点对应链表或列表存储邻接边，适合稀疏图（空间效率高）。  
- **邻接矩阵**：二维数组存储节点间边权，适合稠密图（访问效率高，O(1)查询）。

### 关键问题
1. **深搜与广搜的核心区别是什么？**  
   答：深搜是深度优先，沿一条路径搜索到底，适合处理需要回溯或路径记录的问题；广搜是广度优先，逐层扩展，适合处理最短路径（无权图）或分层遍历问题。二者的代码模板和标记时机不同，广搜需在入队时标记已访问以避免重复入队。

2. **Prim算法和Kruskal算法分别适用于什么场景？二者的核心区别是什么？**  
   答：Prim算法适用于稠密图（边数接近n²），时间复杂度O(n²)，通过维护节点集合和距离数组选择最近节点；Kruskal算法适用于稀疏图（边数远小于n²），时间复杂度O(m log m)，通过边排序和并查集选择最小无环边。核心区别在于Prim维护节点集合，Kruskal维护边集合，且Kruskal依赖并查集检测环。

3. **拓扑排序的主要步骤是什么？如何通过拓扑排序检测有向图是否存在环？**  
   答：步骤为①统计节点入度，找到入度为0的节点加入结果集；②移除该节点及其出边，更新剩余节点入度，重复直至无入度为0节点。若最终结果集节点数小于图中节点总数，则存在环，否则无环。拓扑排序可应用于处理依赖关系，如课程排课顺序。
