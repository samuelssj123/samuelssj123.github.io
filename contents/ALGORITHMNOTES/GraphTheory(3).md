List: 101. 孤岛的总面积，102. 沉没孤岛，103. 水流问题，104.建造最大岛屿

[101. 孤岛的总面积](#01)，[102. 沉没孤岛](#02)，[103. 水流问题](#03)，[104.建造最大岛屿](#04)

# <span id="01">101. 孤岛的总面积</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1173) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0101.%E5%AD%A4%E5%B2%9B%E7%9A%84%E6%80%BB%E9%9D%A2%E7%A7%AF.html)

![image](../images/GraphTheory(3)-1.png)


## 深度优先搜索版

```python
direction = [[1, 0], [0, 1], [-1, 0], [0, -1]]

def dfs(grid, x, y):
    grid[x][y] = 0
    area = 1
    for i in range(4):
        nx = x + direction[i][0]
        ny = y + direction[i][1]
        if nx < 0 or ny < 0 or nx >= len(grid) or ny >= len(grid[0]):
            continue
        if grid[nx][ny] == 1:
            area += dfs(grid, nx, ny)
    return area


n, m = map(int, input().split())

# 邻接矩阵
grid = []
for i in range(n):
    grid.append(list(map(int, input().split())))

# 清除边界上的连通分量
for i in range(n):
    if grid[i][0] == 1: 
        dfs(grid, i, 0)
    if grid[i][m - 1] == 1: 
        dfs(grid, i, m - 1)

for j in range(m):
    if grid[0][j] == 1: 
        dfs(grid, 0, j)
    if grid[n - 1][j] == 1: 
        dfs(grid, n - 1, j)
    
total_area = 0
# 统计内部所有剩余的连通分量
for i in range(n):
    for j in range(m):
        if grid[i][j] == 1:
            total_area += dfs(grid, i, j)
            
print(total_area)

```


## 广度优先搜索版

```python
from collections import deque

direction = [[1, 0], [0, 1], [-1, 0], [0, -1]]
count = 0

n, m = map(int, input().split())

# 邻接矩阵
g = []
for i in range(n):
    g.append(list(map(int, input().split())))

def bfs(r, c):
    global count
    q = deque()
    q.append((r, c))
    g[r][c] = 0
    count += 1

    while q:
        r, c = q.popleft()
        for di in direction:
            next_r = r + di[0]
            next_c = c + di[1]
            if next_c < 0 or next_c >= m or next_r < 0 or next_r >= n:
                continue
            if g[next_r][next_c] == 1:
                q.append((next_r, next_c))
                g[next_r][next_c] = 0
                count += 1


# 清除边界上的连通分量
for i in range(n):
    if g[i][0] == 1: 
        bfs(i, 0)
    if g[i][m - 1] == 1: 
        bfs(i, m - 1)

for j in range(m):
    if g[0][j] == 1: 
        bfs(0, j)
    if g[n - 1][j] == 1: 
        bfs(n - 1, j)
    
count = 0
# 统计内部所有剩余的连通分量
for i in range(n):
    for j in range(m):
        if g[i][j] == 1:
            bfs(i, j)
            
print(count)
```

# <span id="02">102. 沉没孤岛</span>

[卡码网KamaCoder]() 

[Learning Materials](https://www.programmercarl.com/kamacoder/0102.%E6%B2%89%E6%B2%A1%E5%AD%A4%E5%B2%9B.html)

![image](../images/GraphTheory(3)-2.png)

思路依然是从地图周边出发，将周边空格相邻的陆地都做上标记，然后在遍历一遍地图，遇到 陆地 且没做过标记的，那么都是地图中间的 陆地 ，全部改成水域就行。

步骤一：深搜或者广搜将地图周边的 1 （陆地）全部改成 2 （特殊标记）

步骤二：将水域中间 1 （陆地）全部改成 水域（0）

步骤三：将之前标记的 2 改为 1 （陆地）

```python
direction = [[1, 0], [0, 1], [-1, 0], [0, -1]]

def dfs(grid, x, y):
    grid[x][y] = 2
    for i in range(4):
        nx = x + direction[i][0]
        ny = y + direction[i][1]
        if nx < 0 or ny < 0 or nx >= len(grid) or ny >= len(grid[0]):
            continue
        if grid[nx][ny] == 0 or grid[nx][ny] == 2:
            continue
        dfs(grid, nx, ny)


n, m = map(int, input().split())

# 邻接矩阵
grid = []
for i in range(n):
    grid.append(list(map(int, input().split())))

# 步骤一：
# 从左侧边，和右侧边 向中间遍历
for i in range(n):
    if grid[i][0] == 1: 
        dfs(grid, i, 0)
    if grid[i][m - 1] == 1: 
        dfs(grid, i, m - 1)
# 从上边和下边 向中间遍历
for j in range(m):
    if grid[0][j] == 1: 
        dfs(grid, 0, j)
    if grid[n - 1][j] == 1: 
        dfs(grid, n - 1, j)
    
# 步骤二、步骤三
for i in range(n):
    for j in range(m):
        if grid[i][j] == 1:
            grid[i][j] = 0
        if grid[i][j] == 2:
            grid[i][j] = 1

# 打印结果
for row in grid:
    print(' '.join(map(str, row)))
```

# <span id="03">103. 水流问题</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1175) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0103.%E6%B0%B4%E6%B5%81%E9%97%AE%E9%A2%98.html#%E6%80%9D%E8%B7%AF)

![image](../images/GraphTheory(3)-3.png)

一个比较直白的想法，其实就是 遍历每个点，然后看这个点 能不能同时到达第一组边界和第二组边界。

遍历每一个节点，是 m * n，遍历每一个节点的时候，都要做深搜，深搜的时间复杂度是： m * n

那么整体时间复杂度 就是 O(m^2 * n^2) ，这是一个四次方的时间复杂度。

反过来想，从第一组边界上的节点 逆流而上，将遍历过的节点都标记上。

同样从第二组边界的边上节点 逆流而上，将遍历过的节点也标记上。

然后两方都标记过的节点就是既可以流向第一组边界也可以流向第二组边界的节点。

最后，我们得到两个方向交界的这些节点，就是我们最后要求的节点。


```python
first = set()
second = set()
directions = [[-1, 0], [0, 1], [1, 0], [0, -1]]

def dfs(i, j, graph, visited, side):
    if visited[i][j]:
        return
    visited[i][j] = True
    side.add((i, j))
    for x, y in directions:
        nx = i + x 
        ny = j + y 
        if nx < 0 or ny < 0 or nx >= len(graph) or ny >= len(graph[0]):
            continue
        if int(graph[nx][ny]) >= int(graph[i][j]):
            dfs(nx, ny, graph, visited, side)

def main():
    global first
    global second
    n, m = map(int, input().strip().split())
    graph = []
    for _ in range(n):
        row = input().strip().split()
        graph.append(row)
    # 是否可到达第一边界
    visited = [[False] * m for _ in range(n)]
    for i in range(m):
        dfs(0, i, graph, visited, first)
    for i in range(n):
        dfs(i, 0, graph, visited, first)
    # 是否可到达第二边界
    visited = [[False] * m for _ in range(n)]
    for i in range(m):
        dfs(n - 1, i, graph, visited, second)
    for i in range(n):
        dfs(i, m - 1, graph, visited, second)
    # 可到达第一边界和第二边界
    res = first & second
    for x, y in res:
        print(f"{x} {y}")
    
if __name__ == "__main__":
    main()
```

# <span id="04">104.建造最大岛屿</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1176) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0104.%E5%BB%BA%E9%80%A0%E6%9C%80%E5%A4%A7%E5%B2%9B%E5%B1%BF.html#%E6%80%9D%E8%B7%AF)

![image](../images/GraphTheory(3)-4.png)

本题的一个暴力想法，应该是遍历地图尝试 将每一个 0 改成1，然后去搜索地图中的最大的岛屿面积。

计算地图的最大面积：遍历地图 + 深搜岛屿，时间复杂度为 n * n。

（其实使用深搜还是广搜都是可以的，其目的就是遍历岛屿做一个标记，相当于染色，那么使用哪个遍历方式都行，以下我用深搜来讲解）

每改变一个0的方格，都需要重新计算一个地图的最大面积，所以 整体时间复杂度为：n^4。

- 优化：

其实每次深搜遍历计算最大岛屿面积，我们都做了很多重复的工作。

只要用一次深搜把每个岛屿的面积记录下来就好。

第一步：一次遍历地图，得出各个岛屿的面积，并做编号记录。可以使用map记录，key为岛屿编号，value为岛屿面积

第二步：再遍历地图，遍历0的方格（因为要将0变成1），并统计该1（由0变成的1）周边岛屿面积，将其相邻面积相加在一起，遍历所有 0 之后，就可以得出 选一个0变成1 之后的最大面积。

当然这里还有一个优化的点，就是 可以不用 visited数组，因为有mark来标记，所以遍历过的grid[i][j]是不等于1的。


```python
from typing import List
from collections import defaultdict

direction = [(1,0),(-1,0),(0,1),(0,-1)]
res = 0
idx = 1
count_area = defaultdict(int)

def max_area_island(grid):
    global res, idx, count_area
    res = 0
    idx = 1
    count_area.clear()
    if not grid or len(grid) == 0 or len(grid[0]) == 0:
        return 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                idx += 1
                count_area[idx] = dfs(grid, i, j)
    # 计算改变一个0后的最大可能面积
    check_largest_connect_island(grid)
    # 返回最大值
    max_original = max(count_area.values(), default=0)
    return max(res, max_original)
    
def dfs(grid, row, col):
    if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] != 1:
        return 0
    grid[row][col] = idx
    area = 1
    for i, j in direction:
        nx = row + i
        ny = col + j
        area += dfs(grid, nx, ny)
    return area

def check_largest_connect_island(grid):
    global res
    m, n = len(grid), len(grid[0])
    for row in range(m):
        for col in range(n):
            if grid[row][col] == 0:
                area = 1
                visited = set()
                for i, j in direction:
                    nx = row + i 
                    ny = col + j
                    if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] != 0 and grid[nx][ny] not in visited:
                        visited.add(grid[nx][ny])
                        area += count_area[grid[nx][ny]]
                res = max(res, area)

m, n = map(int, input().split())
grid = []

for i in range(m):
    grid.append(list(map(int,input().split())))

print(max_area_island(grid))
```


### 题目理解
题目要求给定一个由 1（陆地）和 0（水）组成的矩阵，你最多可以将矩阵中的一格水变为一块陆地，然后计算在执行此操作之后，矩阵中最大的岛屿面积是多少。岛屿面积是组成岛屿的陆地的总数，岛屿是被水包围，并且通过水平方向或垂直方向上相邻的陆地连接而成的，同时假设矩阵外均被水包围。

### 整体思路
整体思路分为以下几个步骤：
1. **标记不同的岛屿**：遍历矩阵，使用深度优先搜索（DFS）标记每个岛屿，并记录每个岛屿的面积。
2. **尝试将一个水单元格变为陆地**：遍历矩阵中的每个水单元格，计算将其变为陆地后能连接的岛屿总面积。
3. **比较并返回最大值**：比较原始岛屿的最大面积和将一个水单元格变为陆地后的最大面积，返回其中的最大值。

### 代码各部分详细解释

#### 全局变量
```python
direction = [(1,0),(-1,0),(0,1),(0,-1)]
res = 0
idx = 1
count_area = defaultdict(int)
```
- `direction`：这是一个列表，存储了四个方向（下、上、右、左）的偏移量，用于在深度优先搜索中遍历相邻的单元格。
- `res`：用于记录将一个水单元格变为陆地后能得到的最大岛屿面积。
- `idx`：用于给不同的岛屿分配唯一的编号，初始值为 1。
- `count_area`：这是一个 `defaultdict`，用于记录每个岛屿的面积，键为岛屿的编号，值为岛屿的面积。

#### `max_area_island` 函数
```python
def max_area_island(grid):
    global res, idx, count_area
    res = 0
    idx = 1
    count_area.clear()
    if not grid or len(grid) == 0 or len(grid[0]) == 0:
        return 0
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 1:
                idx += 1
                count_area[idx] = dfs(grid, i, j)
    # 计算改变一个0后的最大可能面积
    check_largest_connect_island(grid)
    # 返回最大值
    max_original = max(count_area.values(), default=0)
    return max(res, max_original)
```
- **功能**：计算矩阵中最大的岛屿面积，考虑将一个水单元格变为陆地的情况。
- **步骤**：
  1. 重置全局变量 `res`、`idx` 和清空 `count_area`，确保每次调用函数时数据是干净的。
  2. 检查矩阵是否为空，如果为空则直接返回 0。
  3. 遍历矩阵中的每个单元格，如果是陆地（值为 1），则调用 `dfs` 函数标记该岛屿，并记录其面积到 `count_area` 中。
  4. 调用 `check_largest_connect_island` 函数，计算将一个水单元格变为陆地后能得到的最大岛屿面积。
  5. 计算原始岛屿的最大面积 `max_original`。
  6. 返回 `res` 和 `max_original` 中的最大值。

#### `dfs` 函数
```python
def dfs(grid, row, col):
    if row < 0 or row >= len(grid) or col < 0 or col >= len(grid[0]) or grid[row][col] != 1:
        return 0
    grid[row][col] = idx
    area = 1
    for i, j in direction:
        nx = row + i
        ny = col + j
        area += dfs(grid, nx, ny)
    return area
```
- **功能**：使用深度优先搜索标记一个岛屿，并计算该岛屿的面积。
- **步骤**：
  1. 检查当前单元格是否越界或者不是陆地（值不为 1），如果是则返回 0。
  2. 将当前单元格标记为当前岛屿的编号 `idx`。
  3. 初始化岛屿面积为 1。
  4. 遍历四个方向的相邻单元格，递归调用 `dfs` 函数，并将返回的面积累加到当前岛屿面积中。
  5. 返回当前岛屿的面积。

#### `check_largest_connect_island` 函数
```python
def check_largest_connect_island(grid):
    global res
    m, n = len(grid), len(grid[0])
    for row in range(m):
        for col in range(n):
            if grid[row][col] == 0:
                area = 1
                visited = set()
                for i, j in direction:
                    nx = row + i 
                    ny = col + j
                    if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] != 0 and grid[nx][ny] not in visited:
                        visited.add(grid[nx][ny])
                        area += count_area[grid[nx][ny]]
                res = max(res, area)
```
- **功能**：遍历矩阵中的每个水单元格，计算将其变为陆地后能连接的岛屿总面积，并更新 `res` 为最大值。
- **步骤**：
  1. 遍历矩阵中的每个单元格，如果是水（值为 0），则进行以下操作。
  2. 初始化面积为 1，表示将当前水单元格变为陆地。
  3. 使用 `visited` 集合记录已经访问过的岛屿编号，避免重复计算。
  4. 遍历四个方向的相邻单元格，如果相邻单元格是陆地且对应的岛屿编号未被访问过，则将该岛屿的面积累加到当前面积中，并将该岛屿编号加入 `visited` 集合。
  5. 更新 `res` 为当前面积和 `res` 中的最大值。

#### 主程序部分
```python
m, n = map(int, input().split())
grid = []

for i in range(m):
    grid.append(list(map(int,input().split())))

print(max_area_island(grid))
```
- **功能**：读取输入的矩阵行数和列数，以及矩阵的具体元素，然后调用 `max_area_island` 函数计算并输出最大岛屿面积。

### 总结
通过上述步骤，我们先标记出矩阵中的所有岛屿并记录其面积，然后尝试将每个水单元格变为陆地，计算连接后的岛屿总面积，最后返回最大的岛屿面积。
