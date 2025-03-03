# 图像卷积



```python
import torch
from torch import nn
from d2l import torch as d2l
```

## 互相关运算


```python
def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]): #Y.shape[0] 表示 Y 张量的行数
        for j in range(Y.shape[1]):  #Y.shape[1] 表示 Y 张量的列数
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)
```




    tensor([[19., 25.],
            [37., 43.]])



## 卷积层


```python
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

X = torch.ones((6, 8))
X[:,2:6] = 0
X
```




    tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.]])



## 图像的边缘检测


```python
X = torch.ones((6, 8))
X[:, 2:6] = 0
X
```




    tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.],
            [1., 1., 0., 0., 0., 0., 1., 1.]])




```python
K = torch.tensor([[1.0, -1.0]])
```


```python
Y = corr2d(X, K)
Y
```




    tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
            [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])




```python
corr2d(X.t(), K)  # 只能检测垂直边缘而不能够水平边缘，t是转置
```




    tensor([[0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0.]])



## 学习卷积核


```python
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = nn.Conv2d(1, 1, kernel_size = (1, 2), bias = False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、通道、高度、宽度）， 批量大小指的是在一次训练迭代中同时处理的样本数量。
#通道数表示每个样本中特征图的数量。在图像处理中，通道可以理解为不同的颜色通道（如 RGB 图像有 3 个通道），或者是经过卷积操作后得到的不同特征图。
# 其中批量大小和通道数都为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2 

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad() #将模型中所有可学习参数的梯度清零
    l.sum().backward()
    conv2d.weight.data[:] -= lr * conv2d.weight.grad #将当前权重值减去学习率乘以梯度的值，实现参数的更新
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')
        
```

    epoch 2, loss 14.049
    epoch 4, loss 3.640
    epoch 6, loss 1.136
    epoch 8, loss 0.406
    epoch 10, loss 0.156
    


```python
conv2d.weight.data.reshape((1, 2))
```




    tensor([[ 1.0245, -0.9447]])




```python

```
