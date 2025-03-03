# 填充和步幅


```python
import torch
from torch import nn


# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    X = X.reshape((1,1) + X.shape)  # 这里的（1，1）表示批量大小和通道数都是1
    Y = conv2d(X)# 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])  #Y.shape[2:] 表示获取 Y 张量形状的后两个维度，即 (高度, 宽度)

conv2d = nn.Conv2d(1, 1, kernel_size = 3, padding = 1)
X = torch.rand(size = (8, 8))
comp_conv2d(conv2d, X).shape
```




    torch.Size([8, 8])




```python
conv2d = nn.Conv2d(1, 1, kernel_size = (5,3), padding = (2, 1)) #我们使用高度为5，宽度为3的卷积核，高度和宽度两边的填充分别为2和1。
comp_conv2d(conv2d, X).shape
```




    torch.Size([8, 8])




```python
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```




    torch.Size([4, 4])




```python
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```




    torch.Size([2, 2])




```python

```
