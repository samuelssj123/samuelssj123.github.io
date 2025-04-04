List: 101. 孤岛的总面积，102. 沉没孤岛

[101. 孤岛的总面积](#01)，[102. 沉没孤岛](#02)，[](#03)，[](#04),[](#05)

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

# <span id="03">理论基础</span>

[卡码网KamaCoder]() 

[Learning Materials]()

![image](../images/.png)

# <span id="04">理论基础</span>

[卡码网KamaCoder]() 

[Learning Materials]()

![image](../images/.png)

# <span id="05">理论基础</span>

[Leetcode]() 

[Learning Materials]()

![image](../images/.png)
