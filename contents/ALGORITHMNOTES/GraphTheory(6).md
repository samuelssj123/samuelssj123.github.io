List: 108. 冗余连接，109. 冗余连接II

[108. 冗余连接](#01)，[109. 冗余连接II](#02)

# <span id="01">108. 冗余连接</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1181) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0108.%E5%86%97%E4%BD%99%E8%BF%9E%E6%8E%A5.html#%E6%80%9D%E8%B7%AF)

![image](../images/GraphTheory(6)-1.png)

```python
def init(size):
    for i in range(1, size + 1):
        father[i] = i

def find(u):
    if u == father[u]:
        return u
    else:
        father[u] = find(father[u])
        return father[u] 

def isSame(u, v):
    u = find(u)
    v = find(v)
    return u == v

def join(u, v):
    u = find(u)
    v = find(v)
    if u == v:
        return
    father[u] = v

n = int(input())
father = list(range(n + 1))
init(n)
for i in range(n):
    s, t = map(int, input().split())
    if isSame(s, t):
        result = str(s) + ' ' + str(t)
    else:
        join(s, t)
print(result)
```

# <span id="02">109. 冗余连接II</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1182) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0109.%E5%86%97%E4%BD%99%E8%BF%9E%E6%8E%A5II.html)

![image](../images/GraphTheory(6)-2.png)

```python
from collections import defaultdict

def init(size):
    global father
    father = list(range(size + 1))

def find(u):
    if u == father[u]:
        return u
    else:
        father[u] = find(father[u])
        return father[u] 

def isSame(u, v):
    u = find(u)
    v = find(v)
    return u == v

def join(u, v):
    u = find(u)
    v = find(v)
    if u == v:
        return
    father[u] = v

def getRemoveEdge(size, edges):
    init(size)
    for i in range(n):
        if isSame(edges[i][0],edges[i][1]):
            result = str(edges[i][0]) + ' ' + str(edges[i][1])
            print(result)
            return
        else:
            join(edges[i][0],edges[i][1])

def isTreeAfterRemoveEdge(size, edges, deleteedge):
    init(size)
    for i in range(n):
        if i == deleteedge:
            continue
        if isSame(edges[i][0],edges[i][1]):
            return False
        else:
            join(edges[i][0],edges[i][1])
    return True

n = int(input())
edges = []
indegree = defaultdict(int)

for i in range(n):
    s, t = map(int, input().split())
    indegree[t] += 1
    edges.append([s, t])

vec = list()
for i in range(n - 1, -1, -1):
    if indegree[edges[i][1]] == 2:
        vec.append(i)

if len(vec) > 0:
    if isTreeAfterRemoveEdge(n, edges, vec[0]):
        print(edges[vec[0]][0], edges[vec[0]][1])
    else:
        print(edges[vec[1]][0], edges[vec[1]][1])
else:
    getRemoveEdge(n, edges)
```

### `vec`、`edges[0][0]` 和 `edges[0][1]` 的含义

- **`vec`**：它是一个列表，用于存储入度为 2 的节点的入边在 `edges` 列表中的索引。代码通过倒序遍历 `edges` 列表，将入度为 2 的节点的入边索引添加到 `vec` 中。

- **`edges[0][0]`**：`edges` 是一个二维列表，`edges[0]` 表示 `edges` 列表中的第一条边，`edges[0][0]` 则表示第一条边的起始节点。

- **`edges[0][1]`**：同理，`edges[0][1]` 表示第一条边的终止节点。

### 代码解释

1. **并查集初始化**：`init` 函数把 `father` 列表初始化为每个节点的父节点是其自身。

2. **查找和合并操作**：`find` 函数用于查找节点的根节点，`join` 函数用于合并两个节点所在的集合。

3. **入度统计**：借助 `indegree` 字典统计每个节点的入度。

4. **冗余边判断**：
   - 若存在入度为 2 的节点，倒序遍历 `edges` 列表，将入度为 2 的节点的入边索引添加到 `vec` 中。
   - 若 `vec` 不为空，依次尝试删除 `vec` 中的边，判断删除后图是否为有向树。
   - 若 `vec` 为空，使用 `getRemoveEdge` 函数找出会形成环的边。
