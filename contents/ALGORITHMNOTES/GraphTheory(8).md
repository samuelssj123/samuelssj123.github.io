List: 拓扑排序精讲，dijkstra（朴素版）精讲

[拓扑排序精讲](#01)，[dijkstra（朴素版）精讲](#02)


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

