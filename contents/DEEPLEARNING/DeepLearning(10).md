# 多输入输出通道


```python
import torch
from d2l import torch as d2l
```

## 多输入通道


```python
def corr2d_multi_in(X, K):
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K)) #先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
```


```python
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```




    tensor([[ 56.,  72.],
            [104., 120.]])



## 多输出通道


```python
def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)  #迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。最后将所有结果都叠加在一起
```


```python
K = torch.stack((K, K + 1, K + 2), 0)
K.shape
```




    torch.Size([3, 2, 2, 2])




```python
corr2d_multi_in_out(X, K)
```




    tensor([[[ 56.,  72.],
             [104., 120.]],
    
            [[ 76., 100.],
             [148., 172.]],
    
            [[ 96., 128.],
             [192., 224.]]])



## $1\times 1$ 卷积层

类似于一个全连接层


```python
def corr2d_multi_in_out_1x1(X, K):
    # 获取输入特征图 X 的形状，c_i 表示输入通道数，h 表示特征图的高度，w 表示特征图的宽度
    c_i, h, w = X.shape
    # 获取卷积核 K 的形状，c_o 表示输出通道数
    c_o = K.shape[0]
    # 将输入特征图 X 进行重塑，将每个通道的二维特征图展平为一维向量
    # 最终形状为 (c_i, h * w)，即每个通道对应一个长度为 h * w 的向量
    X = X.reshape((c_i, h * w))
    # 将卷积核 K 进行重塑，将其转换为形状为 (c_o, c_i) 的矩阵
    # 其中 c_o 是输出通道数，c_i 是输入通道数
    K = K.reshape((c_o, c_i))
    # 进行矩阵乘法，将卷积核矩阵 K 与展平后的输入特征图矩阵 X 相乘
    # 得到的结果 Y 的形状为 (c_o, h * w)
    Y = torch.matmul(K, X) # 全连接层中的矩阵乘法
    # 将结果 Y 重塑为 (c_o, h, w) 的形状，即恢复为多通道的二维特征图
    return Y.reshape((c_o, h, w))
```


```python
X = torch.normal(0, 1, (3, 3, 3))
K = torch.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
```


```python

```
