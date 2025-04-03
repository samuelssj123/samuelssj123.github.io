List: 99. 岛屿数量

[99. 岛屿数量](#01)，[](#02)，[](#03)，[](#04),[](#05)

# <span id="01">99. 岛屿数量</span>

[卡码网题目链接（ACM模式）](https://kamacoder.com/problempage.php?pid=1171) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0099.%E5%B2%9B%E5%B1%BF%E7%9A%84%E6%95%B0%E9%87%8F%E6%B7%B1%E6%90%9C.html)

![image](../images/GraphTheory(2)-1.png)

## 深度优先搜索 版本一

```python
direction = [[0, 1], [1, 0], [0, -1], [-1, 0]]

def dfs(grid, visited, x, y):
    for i, j in direction:
        nx = x + i
        ny = y + j
        if nx < 0 or ny < 0 or nx >= len(grid) or ny >= len(grid[0]):
            continue
        if not visited[nx][ny] and grid[nx][ny] == 1:
            visited[nx][ny] = 1
            dfs(grid, visited, nx, ny)

if __name__ == '__main__':
    n, m = map(int, input().split())
    grid = []
    for i in range(n):
        grid.append(list(map(int, input().split())))
    visited = [[False] * m for _ in range(n)]
    result = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 1 and not visited[i][j]:
                result += 1
                visited[i][j] = 1
                dfs(grid, visited, i, j)
    print(result)
```

## 深度优先搜索 版本二

```python
direction = [[0, 1], [1, 0], [0, -1], [-1, 0]]

def dfs(grid, visited, x, y):
    if visited[x][y] == True or grid[x][y] == 0:
        return
    visited[x][y] = True

    for i, j in direction:
        nx = x + i
        ny = y + j
        if nx < 0 or ny < 0 or nx >= len(grid) or ny >= len(grid[0]):
            continue
        dfs(grid, visited, nx, ny)

if __name__ == '__main__':
    n, m = map(int, input().split())
    grid = []
    for i in range(n):
        grid.append(list(map(int, input().split())))
    visited = [[False] * m for _ in range(n)]
    result = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 1 and not visited[i][j]:
                result += 1
                dfs(grid, visited, i, j)
    print(result)
```

# <span id="02">理论基础</span>

[卡码网题目链接（ACM模式）](https://kamacoder.com/problempage.php?pid=1171) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0099.%E5%B2%9B%E5%B1%BF%E7%9A%84%E6%95%B0%E9%87%8F%E5%B9%BF%E6%90%9C.html#%E6%80%9D%E8%B7%AF)

![image](../images/GraphTheory(2)-2.png)

```python
from collections import deque

direction = [[0, 1], [1, 0], [0, -1], [-1, 0]]

def bfs(grid, visited, x, y):
    que = deque([])
    que.append([x, y])
    visited[x][y] = True
    while que:
        curx, cury = que.popleft()
        for i, j in direction:
            nx = curx + i
            ny = cury + j
            # 检查新坐标是否越界
            if nx < 0 or ny < 0 or nx >= len(grid) or ny >= len(grid[0]):
                continue
            # 检查新坐标是否为未访问的陆地
            if not visited[nx][ny] and grid[nx][ny] == 1:
                que.append([nx, ny])
                visited[nx][ny] = True


if __name__ == '__main__':
    n, m = map(int, input().split())
    grid = []
    for i in range(n):
        grid.append(list(map(int, input().split())))
    visited = [[False] * m for _ in range(n)]
    result = 0
    for i in range(n):
        for j in range(m):
            if grid[i][j] == 1 and not visited[i][j]:
                result += 1
                bfs(grid, visited, i, j)
    print(result)
```

# <span id="03">理论基础</span>

[卡码网题目链接（ACM模式）]() 

[Learning Materials]()

![image](../images/.png)
