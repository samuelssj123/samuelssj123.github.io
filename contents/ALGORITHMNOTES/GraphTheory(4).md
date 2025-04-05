List: 110. 字符串接龙，105.有向图的完全可达性，106. 岛屿的周长

[110. 字符串接龙](#01)，[105.有向图的完全可达性](#02)，[106. 岛屿的周长](#03)

# <span id="01">110. 字符串接龙</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1183) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0110.%E5%AD%97%E7%AC%A6%E4%B8%B2%E6%8E%A5%E9%BE%99.html#%E6%80%9D%E8%B7%AF)


![image](../images/GraphTheory(4)-1.png)

---
1. **一段话总结**：题目要求在给定的字典`strList`中，从`beginStr`转换到`endStr`，每次只能改变一个字符且中间字符串必须在字典中，求最短转换序列的字符串数目，不存在则返回0。解题**思路**是通过枚举替换字符判断字符串间的连接关系，构建图结构，然后利用**广度优先搜索（BFS）**在无权图中求最短路，同时使用`set`检查字符串是否在集合中，用`map`记录访问情况和路径长度，防止死循环。还可尝试**双向BFS**优化。

2. **详细总结**
    - **题目要求**：给定字典`strList`、起始字符串`beginStr`和结束字符串`endStr`，要求找出从`beginStr`到`endStr`的最短转换序列中的字符串数目。转换需满足：序列首为`beginStr`，尾为`endStr`；每次仅能改变一个位置的字符；中间字符串必须在`strList`中；`beginStr`和`endStr`不在`strList`中，且字符串仅由小写字母组成。若不存在转换序列，返回0。
    - **输入输出**：
|输入|描述|
|--|--|
|第一行|整数`N`，表示`strList`中字符串数量|
|第二行|两个字符串，用空格隔开，分别为`beginStr`和`endStr`|
|后续`N`行|每行一个字符串，代表`strList`中的字符串|
|输出|从`beginStr`转换到`endStr`最短转换序列的字符串数量，不存在则输出0|
    - **解题思路**：
        - **构建图结构**：在搜索过程中，枚举用26个字母替换当前字符串的每一个字符，若替换后的字符串在`strList`中出现，则判断这两个字符串有连接。
        - **求最短路径**：在无权图中求起点`beginStr`和终点`endStr`的最短路径，使用广度优先搜索（BFS）最合适。因为BFS以起点为中心向四周扩散搜索，搜到终点时的路径一定是最短的。而深度优先搜索（DFS）需在到达终点的不同路径中选择最短路，相对麻烦。
        - **防止死循环**：由于本题是无向图，需要用标记位记录节点是否走过，使用`unordered_map`来记录`strList`里的字符串是否被访问过，同时记录路径长度，用`unordered_set`检查字符串是否出现在字符串集合里，这样效率更高。
    - **代码实现**：提供了C++代码示例，通过`unordered_set`存储`strList`中的字符串，`unordered_map`记录字符串的访问情况和路径长度，`queue`实现BFS。在循环中，取出队列头部字符串，替换每个字符后检查是否为终点或在`strList`中且未被访问过，若满足条件则更新路径长度并加入队列。若未找到路径则输出0。同时提到可以用双向BFS优化。
4. **关键问题**
    - **问题1：为什么在本题中使用BFS比DFS更合适？**
        - **答案**：在无权图中求最短路，BFS以起点为中心向四周扩散搜索，一旦搜到终点，此时的路径一定是最短的。而DFS需要在到达终点的不同路径中选择一条最短路，实现起来更麻烦。
    - **问题2：代码中`unordered_set`和`unordered_map`分别起到什么作用？**
        - **答案**：`unordered_set`用于存储字典`strList`中的字符串，在检查某个替换后的字符串是否在字典中时，使用`unordered_set`的`find`方法效率更高。`unordered_map`用于记录`strList`里的字符串是否被访问过，同时记录从起点到该字符串的路径长度，以此来避免重复访问和计算路径长度。
    - **问题3：双向BFS与普通BFS相比有什么优势？**
        - **答案**：双向BFS从起始点和终点两端同时进行搜索，能更快地找到相遇点，从而减少搜索的范围和时间复杂度。普通BFS是从起点单向搜索，搜索范围相对较大，在一些复杂情况下双向BFS的效率更高，但实现相对复杂一些。 


```python
from collections import deque

def judge(s1, s2):
    count = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            count += 1
    return count == 1

n = int(input())
beginstr, endstr = map(str, input().split())

if beginstr == endstr:
    print(0)
    exit()

strlist = []
for _ in range(n):
    strlist.append(input())
visited = [False for _ in range(n)]
que = deque()
que.append([beginstr, 1])
while que:
    judgestr, step = que.popleft()
    if judge(judgestr, endstr):
        print(step + 1)
        exit()
    for i in range(n):
        if visited[i] == False and judge(strlist[i], judgestr):
            visited[i] = True
            que.append([strlist[i], step + 1])
print(0)
```

# <span id="02">105.有向图的完全可达性</span>


[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1177) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0105.%E6%9C%89%E5%90%91%E5%9B%BE%E7%9A%84%E5%AE%8C%E5%85%A8%E5%8F%AF%E8%BE%BE%E6%80%A7.html)

![image](../images/GraphTheory(4)-2.png)

---
1. **一段话总结**：题目要求判断有向图中从1号节点出发是否能到达所有节点，**给定节点数量N和边的数量K**，以及各条边的连接情况。解题**思路**是通过**深度优先搜索（DFS）或广度优先搜索（BFS）**来遍历图。DFS有两种写法，分别是处理当前访问节点和处理下一个要访问的节点，且本题不需要回溯；BFS则利用队列按层遍历。最后根据遍历结果，若所有节点都被访问到则输出1，否则输出-1。
2. **详细总结**
    - **题目要求**：给定一个有向图，包含`N`个节点（节点编号为1到`N`）和`K`条边，判断从1号节点出发能否通过边到达任何节点。如果可以，输出1；否则，输出 -1。
    - **输入输出**：
|输入|描述|
|--|--|
|第一行|两个正整数`N`和`K`，分别表示节点数量和边的数量|
|后续`K`行|每行两个正整数`s`和`t`，表示从`s`节点到`t`节点有一条单向边|
|输出|1（1号节点可到达所有节点）或 -1（1号节点不能到达所有节点）|
    - **解题思路**：
        - **深度优先搜索（DFS）**：
            - **处理当前访问的节点**：递归函数传入有向图、当前节点编号和访问记录数组。若当前节点已访问过，终止递归；否则标记当前节点为已访问，然后对当前节点的所有邻接节点递归调用DFS。
            - **处理下一个要访问的节点**：递归函数同样传入相关参数，遍历当前节点的邻接节点，若邻接节点未被访问，则标记为已访问并递归调用DFS。这种写法无需终止条件。
            - **回溯问题**：本题只需判断1号节点是否能到达所有节点，所以不需要回溯；而在搜索可行路径时需要回溯。
        - **广度优先搜索（BFS）**：使用队列存储待访问的节点。从1号节点开始，将其标记为已访问并加入队列。每次从队列取出一个节点，遍历其所有邻接节点，若邻接节点未被访问，则标记为已访问并加入队列，直到队列为空。
    - **代码实现**：提供了C++语言实现的DFS两种写法和BFS代码。在代码中，用邻接表存储有向图，通过遍历结果判断1号节点是否能到达所有节点。
4. **关键问题**
    - **问题1：为什么本题DFS不需要回溯操作？**
        - **答案**：本题目的是判断1号节点是否能到达所有节点，只要标记所有遍历过的节点即可，不需要回溯去撤销操作。而在搜索一条可行路径时，若不回溯就无法“调头”探索其他路径，所以需要回溯。
    - **问题2：DFS处理当前访问节点和处理下一个要访问节点的写法区别是什么？**
        - **答案**：处理当前访问节点时，有明确的终止条件，即当前节点已访问则终止递归，在进入递归前标记当前节点；处理下一个要访问节点时，没有单独的终止条件，在遍历邻接节点时判断下一个节点是否未访问，若未访问则标记并递归，且在代码实现中，处理下一个要访问节点的写法需要预先处理起始节点（如将1号节点预先标记为已访问）。
    - **问题3：BFS和DFS在解决本题时各自的优势是什么？**
        - **答案**：BFS利用队列按层遍历，逻辑相对清晰，代码实现较为直观，能确保按照距离起始节点由近及远的顺序访问节点；DFS的优势在于可以深入探索一条路径，对于一些需要深度探索的情况可能更高效，并且DFS的两种写法有助于理解递归的不同处理方式。 


## 深搜写法一：

```python
def dfs(graph, key, visited):
    if visited[key]:
        return
    visited[key] = True  
    keys = graph[key]
    for key in keys:
        dfs(graph, key, visited)

n, m = map(int, input().split())
graph = {i: [] for i in range(1, n + 1)}
for _ in range(m):
    s, t = map(int, input().split())
    graph[s].append(t)
visited = [False for _ in range(n + 1)]
dfs(graph, 1, visited)

def check(visited):
    for i in range(1, n + 1):
        if visited[i] == False:
            return -1
    return 1

print(check(visited))
```

## 深搜写法二：

```python
def dfs(graph, key, visited):
    keys = graph[key]
    for neighbor in keys:
        if visited[neighbor] == False:
            visited[neighbor] = True
            dfs(graph, neighbor, visited)

n, m = map(int, input().split())
graph = {i: [] for i in range(1, n + 1)}
for _ in range(m):
    s, t = map(int, input().split())
    graph[s].append(t)

visited = [False for _ in range(n + 1)]
visited[1] = True
dfs(graph, 1, visited)

def check(visited):
    for i in range(1, n + 1):
        if visited[i] == False:
            return -1
    return 1

print(check(visited))
```

在深度优先搜索（DFS）中，所谓“处理当前层”和“处理下一层”的不同写法，主要体现在代码逻辑结构和对节点处理的时机上。下面以你之前关于判断有向图可达性的问题为例，详细解释这两种写法的差异及其原因：

### 处理当前层的写法
```python
def dfs(graph, current_key, visited):
    if visited[current_key]:
        return
    visited[current_key] = True  
    neighbors = graph[current_key]
    for neighbor in neighbors:
        dfs(graph, neighbor, visited)
```
在这种写法中：
- **处理当前节点**：首先检查当前节点 `current_key` 是否已经被访问过（`if visited[current_key]: return`），如果已访问则直接返回，不再继续处理。如果未访问，则将其标记为已访问（`visited[current_key] = True`），这是对当前节点的处理。
- **递归进入下一层**：在处理完当前节点后，获取当前节点的所有邻接节点（`neighbors = graph[current_key]`），然后通过循环对每个邻接节点递归调用 `dfs` 函数（`dfs(graph, neighbor, visited)`），从而进入下一层的搜索。这种写法是先处理完当前节点的标记等操作，再递归进入下一层。

### 处理下一层的写法
```python
def dfs(graph, current_key, visited):
    neighbors = graph[current_key]
    for neighbor in neighbors:
        if not visited[neighbor]:
            visited[neighbor] = True
            dfs(graph, neighbor, visited)
```
在这种写法中：
- **不单独处理当前节点**：没有像上一种写法那样，在进入递归之前专门对当前节点 `current_key` 进行已访问的判断和标记操作。而是直接获取当前节点的邻接节点（`neighbors = graph[current_key]`）。
- **在处理下一层时处理节点**：在遍历邻接节点的循环中（`for neighbor in neighbors:`），对每个邻接节点进行判断，如果未被访问（`if not visited[neighbor]:`），则将其标记为已访问（`visited[neighbor] = True`），然后递归调用 `dfs` 函数进入下一层（`dfs(graph, neighbor, visited)`）。这种写法是在处理下一层节点时，顺便处理节点的访问标记等操作，重点更偏向于快速进入下一层的搜索。

### 总结
- **处理当前层**：先对当前节点进行操作（如标记为已访问），然后再递归进入下一层，这种写法逻辑相对清晰，明确地将当前节点的处理和下一层的搜索分开。
- **处理下一层**：更注重快速进入下一层的搜索，在处理下一层节点的过程中同时处理节点的相关操作，代码结构上相对简洁一些。

两种写法在本质上都是深度优先搜索，只是对节点处理的时机和代码结构有所不同，在实际应用中可以根据具体需求和个人习惯选择合适的写法。 

## 广搜版：

```python
from collections import deque
n, m = map(int, input().split())
graph = {i: [] for i in range(1, n + 1)}
for _ in range(m):
    s, t = map(int, input().split())
    graph[s].append(t)

visited = [False for _ in range(n + 1)]
visited[1] = True

que = deque()
que.append(1)
while que:
    key = que.popleft()
    keys = graph[key]
    for neighbor in keys:
        if visited[neighbor] == False:
            que.append(neighbor)
            visited[neighbor] = True

def check(visited):
    for i in range(1, n + 1):
        if visited[i] == False:
            return -1
    return 1

print(check(visited))
```

# <span id="03">106. 岛屿的周长</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1178) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0106.%E5%B2%9B%E5%B1%BF%E7%9A%84%E5%91%A8%E9%95%BF.html#%E6%80%9D%E8%B7%AF)

---
岛屿问题最容易让人想到BFS或者DFS，但本题确实还用不上。
1. **一段话总结**：题目要求计算由`1`（陆地）和`0`（水）组成的矩阵中岛屿的周长，**矩阵外均为水，岛屿由水平或垂直相邻的陆地构成，陆地边长为1**。提供了两种解法，解法一是遍历每个节点，遇到陆地时检查其上下左右，若周边是水域或出界则周长加1；解法二是先算出陆地总数乘以4，再减去相邻陆地对数乘以2，**避免重复计算**。同时给出了C++、Java、Python等语言的代码示例。
2. **详细总结**
    - **题目要求**：给定由`1`（陆地）和`0`（水）组成的矩阵，岛屿被水包围且通过水平或垂直相邻的陆地连接，假设矩阵外均是水，岛屿内部无水域，陆地边长为`1`，计算岛屿周长。
    - **输入输出**：
|输入|描述|
|--|--|
|第一行|两个整数`N`、`M`，表示矩阵的行数和列数|
|后续`N`行|每行`M`个数字（`1`或`0`），代表岛屿单元格|
|输出|一个整数，表示岛屿的周长|
    - **解法一**：遍历矩阵的每一个节点，当遇到陆地（值为`1`）时，检查其上下左右四个方向的节点情况。若周边节点是水域（值为`0`）或出界，则该陆地对应的这条边属于岛屿周长，周长加`1`。
    - **解法二**：先计算总的岛屿数量（陆地数量），假设每个岛屿（陆地）有`4`条边，即陆地数量乘以`4`。由于相邻的两个陆地会共用一条边，需要统计相邻陆地的数量（相邻对数），每对相邻陆地会使总边数减少`2`，最终结果为陆地数量乘以`4`减去相邻对数乘以`2`。在统计相邻陆地时，为避免重复计算，只统计陆地的上边和左边相邻陆地。
    - **代码实现**：分别给出了C++、Java、Python三种语言的代码实现。C++实现了两种解法；Java通过遍历陆地，探索其周围四个方向并记录周长；Python通过统计陆地数量和相邻数量来计算周长。
4. **关键问题**
    - **问题1：解法二中为什么只统计上边和左边的相邻陆地，而不统计下边和右边？**
        - **答案**：因为统计相邻陆地时，如果同时统计上边、左边、下边和右边，会导致每对相邻陆地被重复计算。例如，对于两个相邻陆地，从其中一个陆地统计时，会在其右边和下边的检查中再次统计到相邻关系，所以只统计上边和左边可避免重复。
    - **问题2：如果岛屿内部存在水域，这两种解法还适用吗？**
        - **答案**：两种解法均不适用。解法一基于陆地周边是水域或出界就计算周长，若岛屿内部有水域，会错误计算内部水域与陆地边界为周长；解法二计算陆地数量乘以`4`再减去相邻陆地对数乘以`2`，岛屿内部水域会干扰陆地数量和相邻对数的正确统计，无法得出正确周长。
    - **问题3：在解法一的代码中，`direction`数组的作用是什么？**
        - **答案**：`direction`数组用于表示陆地节点上下左右四个方向的偏移量。在检查陆地节点周边情况时，通过数组中的偏移量，计算出周边节点的坐标，进而判断周边节点是否是水域或出界，以此确定是否要将当前陆地对应的边计入周长。例如，`direction[0]`表示向右的偏移量，可通过它计算出当前陆地右边节点的坐标。 


## 解法一：

```python
direction = [(0, 1), (1, 0), (0, -1), (-1, 0)]
n, m = map(int, input().split())
grid = []
for _ in range(n):
    row = list(map(int, input().split()))
    grid.append(row)
result = 0
for i in range(n):
    for j in range(m):
        if grid[i][j] == 1:
            for k in range(4):
                nx = i + direction[k][0]
                ny = j + direction[k][1]
                if nx < 0 or nx >= len(grid) or ny < 0 or ny >= len(grid[0]) or grid[nx][ny] == 0:
                    result += 1
print(result)
```


```python
n, m = map(int, input().split())
grid = []
for _ in range(n):
    row = list(map(int, input().split()))
    grid.append(row)
island = 0 #陆地数量
cover = 0 #相邻数量
for i in range(n):
    for j in range(m):
        if grid[i][j] == 1:
            island += 1 #统计总的陆地数量
            if i - 1 >= 0 and grid[i - 1][j] == 1: #统计上边相邻陆地
                cover += 1
            if j - 1 >= 0 and grid[i][j - 1] == 1: #统计左边相邻陆地
                cover += 1
            #为什么没统计下边和右边？ 因为避免重复计算
print(island * 4 - cover * 2)
```
