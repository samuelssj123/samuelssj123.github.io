List: 并查集理论基础，107. 寻找存在的路径

[并查集理论基础](#01)，[107. 寻找存在的路径](#02)

并查集主要有三个功能：

寻找根节点，函数：find(int u)，也就是判断这个节点的祖先节点是哪个

将两个节点接入到同一个集合，函数：join(int u, int v)，将两个节点连在同一个根节点上

判断两个节点是否在同一个集合，函数：isSame(int u, int v)，就是判断两个节点是不是同一个根节点

# <span id="01">并查集理论基础</span>

[Learning Materials](https://www.programmercarl.com/kamacoder/%E5%9B%BE%E8%AE%BA%E5%B9%B6%E6%9F%A5%E9%9B%86%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.html)

![image](../images/GraphTheory(5)-1.png)



# <span id="02">107. 寻找存在的路径</span>

[卡码网KamaCoder](https://kamacoder.com/problempage.php?pid=1179) 

[Learning Materials](https://www.programmercarl.com/kamacoder/0107.%E5%AF%BB%E6%89%BE%E5%AD%98%E5%9C%A8%E7%9A%84%E8%B7%AF%E5%BE%84.html)

![image](../images/GraphTheory(5)-2.png)

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

n, m = map(int, input().split())
father = list(range(n + 1))
init(n)
while m:
    s, t = map(int, input().split())
    join(s, t)
    m -= 1
source, destination = map(int, input().split())
if isSame(source, destination):
    print(1)
    exit()
print(0)
```
