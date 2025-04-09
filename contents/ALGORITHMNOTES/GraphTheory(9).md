List: Bellman_ford 队列优化算法（又名SPFA），bellman_ford之判断负权回路，bellman_ford之单源有限最短路

[Bellman_ford 队列优化算法（又名SPFA）](#01)，[bellman_ford之判断负权回路](#02)，[bellman_ford之单源有限最短路](#03)

# <span id="01">Bellman_ford 队列优化算法（又名SPFA）</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1152) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0094.%E5%9F%8E%E5%B8%82%E9%97%B4%E8%B4%A7%E7%89%A9%E8%BF%90%E8%BE%93I-SPFA.html#%E8%83%8C%E6%99%AF)

![image](../images/GraphTheory(9)-1.png)

```python
from collections import deque
class Edge:
    def __init__(self, to, val):
        self.to = to
        self.val = val
n, m = map(int, input().split())
grid = [[] for _ in range(n + 1)]
minDist = [float('inf')] * (n + 1)
for i in range(m):
    p1, p2, val = map(int, input().split())
    grid[p1].append(Edge(p2, val))
inqueue = deque([1])
isinqueue = [False] * (n + 1)

minDist[1] = 0
isinqueue[1] = True
while inqueue:
    node = inqueue.popleft()
    isinqueue[node] = False
    for edge in grid[node]:
        if minDist[edge.to] > minDist[node] + edge.val:
            minDist[edge.to] = minDist[node] + edge.val
            if isinqueue[edge.to] == False:
                inqueue.append(edge.to)
                isinqueue[edge.to] = True

if minDist[n] == float('inf'):
    print('unconnected')
else:
    print(minDist[n])
```

# <span id="02">bellman_ford之判断负权回路</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1153) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0095.%E5%9F%8E%E5%B8%82%E9%97%B4%E8%B4%A7%E7%89%A9%E8%BF%90%E8%BE%93II.html)

![image](../images/GraphTheory(9)-2.png)

```python

```

# <span id="03">bellman_ford之单源有限最短路</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1154) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0096.%E5%9F%8E%E5%B8%82%E9%97%B4%E8%B4%A7%E7%89%A9%E8%BF%90%E8%BE%93III.html)

![image](../images/GraphTheory(9)-3.png)

```python

```

