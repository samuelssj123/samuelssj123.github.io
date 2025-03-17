# 转置卷积


```python
import torch
from torch import nn
from d2l import torch as d2l
```


```python
def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y
```


```python
X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
trans_conv(X, K)
```




    tensor([[ 0.,  0.,  1.],
            [ 0.,  4.,  6.],
            [ 4., 12.,  9.]])




```python
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
tconv(X)
```




    tensor([[[[ 0.,  0.,  1.],
              [ 0.,  4.,  6.],
              [ 4., 12.,  9.]]]], grad_fn=<ConvolutionBackward0>)




```python
# 填充、步幅和多通道
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
tconv(X)
```




    tensor([[[[4.]]]], grad_fn=<ConvolutionBackward0>)




```python
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=2, bias=False)
tconv.weight.data = K
tconv(X)
```




    tensor([[[[0., 0., 0., 1.],
              [0., 0., 2., 3.],
              [0., 2., 0., 3.],
              [4., 6., 6., 9.]]]], grad_fn=<ConvolutionBackward0>)




```python
X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
tconv(conv(X)).shape == X.shape
```




    True




```python

```
