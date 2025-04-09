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

## 超时代码

下面这个代码超时，各种大模型都调不好，除非改逻辑，但是答案也是几乎类似的做法不知道为什么又能通过！

```python
n, m = map(int, input().split())
grid = []
minDist = [float('inf')] * (n + 1)
for i in range(m):
    p1, p2, val = map(int, input().split())
    grid.append([p1, p2, val])

minDist[1] = 0
flag = False

for i in range(1, n + 1):
    for side in grid:
        sfrom = side[0]
        sto = side[1]
        sval = side[2]
        if i < n :
            if minDist[sfrom] != float('inf') and minDist[sto] > minDist[sfrom] + sval:
                minDist[sto] = minDist[sfrom] + sval
        else:
            if minDist[sfrom] != float('inf') and minDist[sto] > minDist[sfrom] + sval:
                flag = True

if flag:
    print('circle')
elif minDist[n] == float('inf'):
    print('unconnected')
else:
    print(minDist[n])
```

## 参考答案代码

```python
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    index = 0
    
    n = int(data[index])
    index += 1
    m = int(data[index])
    index += 1
    
    grid = []
    for i in range(m):
        p1 = int(data[index])
        index += 1
        p2 = int(data[index])
        index += 1
        val = int(data[index])
        index += 1
        # p1 指向 p2，权值为 val
        grid.append([p1, p2, val])

    start = 1  # 起点
    end = n    # 终点

    minDist = [float('inf')] * (n + 1)
    minDist[start] = 0
    flag = False

    for i in range(1, n + 1):  # 这里我们松弛n次，最后一次判断负权回路
        for side in grid:
            from_node = side[0]
            to = side[1]
            price = side[2]
            if i < n:
                if minDist[from_node] != float('inf') and minDist[to] > minDist[from_node] + price:
                    minDist[to] = minDist[from_node] + price
            else:  # 多加一次松弛判断负权回路
                if minDist[from_node] != float('inf') and minDist[to] > minDist[from_node] + price:
                    flag = True

    if flag:
        print("circle")
    elif minDist[end] == float('inf'):
        print("unconnected")
    else:
        print(minDist[end])

if __name__ == "__main__":
    main()
```

## 大模型改的代码也能通过，但是加了层提前终止的逻辑，而且还是用的sys输入

import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    m = int(input[idx])
    idx += 1
    
    edges = []
    for _ in range(m):
        s = int(input[idx])
        t = int(input[idx+1])
        v = int(input[idx+2])
        edges.append((s, t, v))
        idx += 3

    min_dist = [float('inf')] * (n + 1)
    min_dist[1] = 0

    # Bellman-Ford算法（加入提前终止优化）
    updated = False
    for _ in range(n - 1):
        updated = False
        for s, t, v in edges:
            if min_dist[s] != float('inf') and min_dist[t] > min_dist[s] + v:
                min_dist[t] = min_dist[s] + v
                updated = True
        if not updated:
            break  # 提前终止

    # 检测负权回路
    has_negative_cycle = False
    for s, t, v in edges:
        if min_dist[s] != float('inf') and min_dist[t] > min_dist[s] + v:
            has_negative_cycle = True
            break

    if has_negative_cycle:
        print("circle")
    elif min_dist[n] == float('inf'):
        print("unconnected")
    else:
        print(min_dist[n])

if __name__ == "__main__":
    main()

# <span id="03">bellman_ford之单源有限最短路</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1154) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0096.%E5%9F%8E%E5%B8%82%E9%97%B4%E8%B4%A7%E7%89%A9%E8%BF%90%E8%BE%93III.html)

![image](../images/GraphTheory(9)-3.png)

```python
n, m = map(int, input().split())
grid = []
minDist = [float('inf')] * (n + 1)
for i in range(m):
    p1, p2, val = map(int, input().split())
    grid.append([p1, p2, val])

src, dst, k = map(int, input().split())
minDist[src] = 0
for i in range(k + 1):
    updated = False #没更新则终止，防止超时
    minDistcopy = minDist.copy()
    for side in grid:
        sfrom = side[0]
        sto = side[1]
        sval = side[2]
        if minDistcopy[sfrom] != float('inf') and minDist[sto] > minDistcopy[sfrom] + sval:
            minDist[sto] = minDistcopy[sfrom] + sval
            updated = True
    if not updated:  # 若边不再更新，即停止回圈
        break

if minDist[dst] == float('inf'):
    print('unreachable')
else:
    print(minDist[dst])
```

