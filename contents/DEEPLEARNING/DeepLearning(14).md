# 使用块的网络（VGG）
该函数有三个参数，分别对应于卷积层的数量num_convs、输入通道的数量in_channels 和输出通道的数量out_channels.


```python
import torch
from torch import nn
from d2l import torch as d2l
```


```python
def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for i in range(num_convs):
        # 创建一个卷积层，输入通道数为 in_channels，输出通道数为 out_channels
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1))
        # 添加 ReLU 激活函数
        layers.append(nn.ReLU())
        # 更新 in_channels 为当前卷积层的输出通道数，以便下一次卷积操作使用
        in_channels = out_channels
    # 添加最大池化层
    layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
    # 使用 nn.Sequential 将所有层组合成一个序列模块
    return nn.Sequential(*layers)
```


```python
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
#原始VGG网络有5个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层。
#第一个模块有64个输出通道，每个后续模块将输出通道数量翻倍，直到该数字达到512。由于该网络使用8个卷积层和3个全连接层，因此它通常被称为VGG-11。
```


```python
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential( *conv_blks, nn.Flatten(), 
                    # 全连接层部分
                    nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(4096, 10))

net = vgg(conv_arch)
```


```python
X = torch.randn(size=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)
```

    Sequential output shape:	 torch.Size([1, 64, 112, 112])
    Sequential output shape:	 torch.Size([1, 128, 56, 56])
    Sequential output shape:	 torch.Size([1, 256, 28, 28])
    Sequential output shape:	 torch.Size([1, 512, 14, 14])
    Sequential output shape:	 torch.Size([1, 512, 7, 7])
    Flatten output shape:	 torch.Size([1, 25088])
    Linear output shape:	 torch.Size([1, 4096])
    ReLU output shape:	 torch.Size([1, 4096])
    Dropout output shape:	 torch.Size([1, 4096])
    Linear output shape:	 torch.Size([1, 4096])
    ReLU output shape:	 torch.Size([1, 4096])
    Dropout output shape:	 torch.Size([1, 4096])
    Linear output shape:	 torch.Size([1, 10])
    

## 训练模型


```python
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch] #pair[1] // ratio：表示元组 pair 中的第二个元素，即该 VGG 块中卷积层的输出通道数，将其除以 ratio 并进行整除操作，得到缩放后的输出通道数。
net = vgg(small_conv_arch)
```


```python
lr, num_epochs, batch_size = 0.05, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```


    
由于我的电脑性能，我并没有跑出完整的结果，但是代码是可以运行成功的。
    



```python

```
