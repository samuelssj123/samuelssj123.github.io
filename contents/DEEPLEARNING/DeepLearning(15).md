# NiN模型


```python
import torch
from torch import nn
from d2l import torch as d2l
```


```python
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size = 1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size = 1), nn.ReLU())
```


```python
net = nn.Sequential(
    nin_block(1, 96, kernel_size = 11, strides = 4, padding = 0),
    nn.MaxPool2d(3, stride = 2),
    nin_block(96, 256, kernel_size = 5, strides = 1, padding = 2),
    nn.MaxPool2d(3, stride = 2),
    nin_block(256, 384, kernel_size = 3, strides = 1, padding = 1),
    nn.MaxPool2d(3, stride = 2),
    nn.Dropout(0.5),
    nin_block(384, 10, kernel_size = 3, strides = 1, padding = 1), # 标签类别数是10
    nn.AdaptiveAvgPool2d((1, 1)), # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten())
```


```python
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

    Sequential output shape:	 torch.Size([1, 96, 54, 54])
    MaxPool2d output shape:	 torch.Size([1, 96, 26, 26])
    Sequential output shape:	 torch.Size([1, 256, 26, 26])
    MaxPool2d output shape:	 torch.Size([1, 256, 12, 12])
    Sequential output shape:	 torch.Size([1, 384, 12, 12])
    MaxPool2d output shape:	 torch.Size([1, 384, 5, 5])
    Dropout output shape:	 torch.Size([1, 384, 5, 5])
    Sequential output shape:	 torch.Size([1, 10, 5, 5])
    AdaptiveAvgPool2d output shape:	 torch.Size([1, 10, 1, 1])
    Flatten output shape:	 torch.Size([1, 10])
    


```python
# 训练模型
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```


    
由于我的电脑性能，我并没有跑出完整的结果，但是代码是可以运行成功的。
    



```python

```
