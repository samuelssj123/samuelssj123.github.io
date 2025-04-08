List: prim算法精讲，kruskal算法精讲

[prim算法精讲](#01)，[kruskal算法精讲](#02)，[](#03)
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

```

# <span id="03"></span>

[卡码网KamaCoder]() 

[Learning Materials]()

![image](../images/.png)

```python

```


