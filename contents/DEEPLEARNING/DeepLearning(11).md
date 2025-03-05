# 池化层


```python
import torch
from torch import nn
from d2l import torch as d2l
```


```python
def pool2d(X, pool_size, mode = 'max'):
    # 从 pool_size 元组中获取池化窗口的高度 p_h 和宽度 p_w
    p_h, p_w = pool_size
    # 初始化输出张量 Y，其形状根据输入张量 X 的形状和池化窗口大小计算得到
    # 输出张量的高度为 X.shape[0] - p_h + 1，宽度为 X.shape[1] - p_w + 1
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    # 遍历输出张量 Y 的每一行
    for i in range(Y.shape[0]):
        # 遍历输出张量 Y 的每一列
        for j in range(Y.shape[1]):
            # 如果模式为 'max'，表示进行最大池化操作
            if mode == 'max':
                # 从输入张量 X 中提取当前池化窗口内的元素，并取最大值赋给 Y[i, j]
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            # 如果模式为 'avg'，表示进行平均池化操作
            elif mode == 'avg':
                # 从输入张量 X 中提取当前池化窗口内的元素，并计算平均值赋给 Y[i, j]
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    # 返回池化后的输出张量 Y
    return Y
```


```python
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```




    tensor([[4., 5.],
            [7., 8.]])




```python
pool2d(X, (2, 2), 'avg')
```




    tensor([[2., 3.],
            [5., 6.]])




```python
# 填充和步幅
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
X
```




    tensor([[[[ 0.,  1.,  2.,  3.],
              [ 4.,  5.,  6.,  7.],
              [ 8.,  9., 10., 11.],
              [12., 13., 14., 15.]]]])




```python
pool2d = nn.MaxPool2d(3)
pool2d(X)
```




    tensor([[[[10.]]]])




```python
pool2d = nn.MaxPool2d(3, padding=1, stride=2)  # 手动设定步幅
pool2d(X)
```




    tensor([[[[ 5.,  7.],
              [13., 15.]]]])




```python
# 多个通道
X = torch.cat((X, X + 1), 1)
X
```




    tensor([[[[ 0.,  1.,  2.,  3.],
              [ 4.,  5.,  6.,  7.],
              [ 8.,  9., 10., 11.],
              [12., 13., 14., 15.]],
    
             [[ 1.,  2.,  3.,  4.],
              [ 5.,  6.,  7.,  8.],
              [ 9., 10., 11., 12.],
              [13., 14., 15., 16.]],
    
             [[ 1.,  2.,  3.,  4.],
              [ 5.,  6.,  7.,  8.],
              [ 9., 10., 11., 12.],
              [13., 14., 15., 16.]],
    
             [[ 2.,  3.,  4.,  5.],
              [ 6.,  7.,  8.,  9.],
              [10., 11., 12., 13.],
              [14., 15., 16., 17.]]]])




```python
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
```




    tensor([[[[ 5.,  7.],
              [13., 15.]],
    
             [[ 6.,  8.],
              [14., 16.]],
    
             [[ 6.,  8.],
              [14., 16.]],
    
             [[ 7.,  9.],
              [15., 17.]]]])




```python

```
