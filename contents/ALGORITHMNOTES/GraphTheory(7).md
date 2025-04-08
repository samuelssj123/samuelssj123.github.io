List: prim算法精讲，kruskal算法精讲，拓扑排序精讲

[prim算法精讲](#01)，[kruskal算法精讲](#02)，[拓扑排序精讲](#03)
# <span id="01">prim算法精讲</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1053) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0053.%E5%AF%BB%E5%AE%9D-prim.html)

![image](../images/GraphTheory(7)-1.png)

```python
v, e = map(int, input().split())
grid = [[10001 for _ in range(v + 1)] for _ in range(v + 1)]
while e:
    x, y, k = map(int, input().split())
    grid[x][y] = k
    grid[y][x] = k
    e -= 1

minDist = [10001 for _ in range(v + 1)] 
minDist[1] = 0
isInTree = [False for _ in range(v + 1)]
for i in range(v):
    cur = -1
    minVal = 10001
    for j in range(1, v + 1):
        if not isInTree[j] and minDist[j] < minVal:
            minVal = minDist[j]
            cur = j
    isInTree[cur] = True
    for j in range(1, v + 1):
        if not isInTree[j] and minDist[j] > grid[cur][j]:
            minDist[j] = grid[cur][j]

result = 0
for i in range(2, v + 1):
    result += minDist[i]
print(result)
```

# <span id="02">kruskal算法精讲</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1053) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0053.%E5%AF%BB%E5%AE%9D-Kruskal.html)

![image](../images/GraphTheory(7)-2.png)

```python
class Edge:
    def __init__(self, l, r, val):
        self.l = l 
        self.r = r 
        self.val = val

n = 10001
father = list(range(n))

def init(size):
    global father
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

def kruskal(n, v, edges):
    edges.sort(key = lambda edge:edge.val)
    init(v)
    result = 0
    for edge in edges:
        x = find(edge.l)
        y = find(edge.r)
        if x != y:
            result += edge.val 
            join(x, y)
    return result

v, e = map(int, input().split())
edges = []
while e:
    x, y, k = map(int, input().split())
    edges.append(Edge(x, y, k))
    e -= 1

result = kruskal(n, v, edges)
print(result)
```

# <span id="03">拓扑排序精讲</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1191) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0117.%E8%BD%AF%E4%BB%B6%E6%9E%84%E5%BB%BA.html#%E6%8B%93%E6%89%91%E6%8E%92%E5%BA%8F%E7%9A%84%E8%83%8C%E6%99%AF)

![image](../images/GraphTheory(7)-3.png)

```python
from collections import deque, defaultdict

n, m = map(int, input().split())
indegree = [0] * n
umap = defaultdict(list)
que = deque()
result = []

while m:
    s, t = map(int, input().split())
    indegree[t] += 1
    umap[s].append(t)
    m -= 1

for i in range(n):
    if indegree[i] == 0:
        que.append(i)

while que:
    cur = que.popleft()
    result.append(cur)
    files = umap[cur]
    if files:
        for i in range(len(files)):
            indegree[files[i]] -= 1
            if indegree[files[i]] == 0:
                que.append(files[i])

if len(result) == n:
    print(" ".join(map(str, result)))
else:
    print(-1)
```


