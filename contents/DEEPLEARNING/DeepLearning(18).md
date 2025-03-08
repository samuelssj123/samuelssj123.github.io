# 残差网络（ResNet）


```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```


```python
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv = False, strides = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size = 3, padding = 1, stride = strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size = 3, padding = 1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size = 1, stride = strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```


```python
blk = Residual(3,3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
Y.shape
```




    torch.Size([4, 3, 6, 6])




```python
blk = Residual(3,6, use_1x1conv=True, strides=2)
blk(X).shape
```




    torch.Size([4, 6, 3, 3])



## [**ResNet模型**]


```python
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
def resnet_block(input_channels, num_channels, num_residuals, first_block = False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv = True, strides = 2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block = True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10))
```


```python
#测试和训练
X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```

    Sequential output shape:	 torch.Size([1, 64, 56, 56])
    Sequential output shape:	 torch.Size([1, 64, 56, 56])
    Sequential output shape:	 torch.Size([1, 128, 28, 28])
    Sequential output shape:	 torch.Size([1, 256, 14, 14])
    Sequential output shape:	 torch.Size([1, 512, 7, 7])
    AdaptiveAvgPool2d output shape:	 torch.Size([1, 512, 1, 1])
    Flatten output shape:	 torch.Size([1, 512])
    Linear output shape:	 torch.Size([1, 10])
    


```python
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```


    
此处结果可以运行代码后展示图片。
    


这段代码实现了一个简化版的 ResNet（残差网络）模型，ResNet 是一种深度卷积神经网络，通过引入残差块（Residual Block）解决了深度神经网络中的梯度消失和梯度爆炸问题，使得网络可以训练更深的层次。下面我们逐部分来理解这段代码。

### 1. 定义残差块类 `Residual`
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv = False, strides = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size = 3, padding = 1, stride = strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size = 3, padding = 1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size = 1, stride = strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```
- **`__init__` 方法**：
  - `input_channels`：输入特征图的通道数。
  - `num_channels`：输出特征图的通道数。
  - `use_1x1conv`：是否使用 1x1 卷积来调整输入特征图的通道数和尺寸。
  - `strides`：卷积层的步长。
  - `self.conv1` 和 `self.conv2`：两个 3x3 的卷积层，用于提取特征。
  - `self.conv3`：如果 `use_1x1conv` 为 `True`，则使用 1x1 卷积来调整输入特征图的通道数和尺寸，使其与输出特征图的尺寸和通道数匹配。
  - `self.bn1` 和 `self.bn2`：两个批量归一化层，用于加速模型收敛和提高模型的稳定性。

- **`forward` 方法**：
  - 首先，将输入 `X` 通过 `self.conv1` 卷积层和 `self.bn1` 批量归一化层，然后使用 ReLU 激活函数得到 `Y`。
  - 接着，将 `Y` 通过 `self.conv2` 卷积层和 `self.bn2` 批量归一化层。
  - 如果 `self.conv3` 存在，则将输入 `X` 通过 `self.conv3` 卷积层，使其与 `Y` 的尺寸和通道数匹配。
  - 最后，将 `Y` 和 `X` 相加，再通过 ReLU 激活函数得到最终输出。

### 2. 定义网络的第一个模块 `b1`
```python
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1))
```
- `nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3)`：一个 7x7 的卷积层，输入通道数为 1，输出通道数为 64，步长为 2，填充为 3。
- `nn.BatchNorm2d(64)`：批量归一化层，对 64 个通道的特征图进行归一化。
- `nn.ReLU()`：ReLU 激活函数，增加模型的非线性。
- `nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)`：一个 3x3 的最大池化层，步长为 2，填充为 1，用于减小特征图的尺寸。

### 3. 定义残差块组函数 `resnet_block`
```python
def resnet_block(input_channels, num_channels, num_residuals, first_block = False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv = True, strides = 2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
```
- `input_channels`：输入特征图的通道数。
- `num_channels`：输出特征图的通道数。
- `num_residuals`：残差块的数量。
- `first_block`：是否为第一个残差块组。
- 如果是第一个残差块组或者不是第一个残差块，则使用普通的残差块；否则，使用带有 1x1 卷积的残差块，步长为 2，用于减小特征图的尺寸。

### 4. 定义网络的其余模块 `b2`、`b3`、`b4`、`b5`
```python
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block = True))
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))
```
- `b2`：第一个残差块组，输入通道数为 64，输出通道数为 64，包含 2 个残差块。
- `b3`、`b4`、`b5`：后续的残差块组，通道数逐渐增加，分别为 128、256、512，每个残差块组包含 2 个残差块。

### 5. 定义完整的网络 `net`
```python
net = nn.Sequential(b1, b2, b3, b4, b5, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, 10))
```
- `b1`、`b2`、`b3`、`b4`、`b5`：前面定义的模块。
- `nn.AdaptiveAvgPool2d((1, 1))`：自适应平均池化层，将特征图的尺寸调整为 1x1。
- `nn.Flatten()`：将特征图展平为一维向量。
- `nn.Linear(512, 10)`：全连接层，将 512 维的向量映射到 10 维的输出，用于分类任务。

综上所述，这段代码实现了一个简化版的 ResNet 模型，用于图像分类任务。模型通过残差块解决了深度神经网络中的梯度消失和梯度爆炸问题，使得网络可以训练更深的层次。

在 `Residual` 类所定义的残差块中，并不是简单的先运行 `conv1`、`conv2`，前向传播后再运行 `conv3` 这种顺序。下面详细解释其运行逻辑：

### 残差块的结构和运行流程

残差块的核心思想是构建一个跳跃连接（shortcut connection），让输入能够直接跨越部分层与后续层的输出相加，这样有助于缓解梯度消失问题，使得网络可以训练得更深。

#### 1. 初始化部分
```python
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv = False, strides = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size = 3, padding = 1, stride = strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size = 3, padding = 1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size = 1, stride = strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
```
- `conv1` 和 `conv2` 是两个 3x3 的卷积层，用于提取输入特征图的特征。
- `conv3` 是一个 1x1 的卷积层，它不是一定会被创建的，只有当 `use_1x1conv` 为 `True` 时才会被创建。其作用是调整输入特征图的通道数和尺寸，以保证在残差连接时，输入 `X` 能够和经过 `conv1`、`conv2` 处理后的特征图 `Y` 在形状上匹配。
- `bn1` 和 `bn2` 是批量归一化层，用于加速模型收敛和提高模型的稳定性。

#### 2. 前向传播部分
```python
    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```
- **`conv1` 和 `conv2` 路径**：
  - 输入 `X` 首先经过 `conv1` 卷积层进行特征提取，然后通过 `bn1` 批量归一化层进行归一化处理，接着使用 ReLU 激活函数引入非线性，得到中间结果 `Y`。
  - 中间结果 `Y` 再经过 `conv2` 卷积层和 `bn2` 批量归一化层，得到最终的特征图 `Y`。

- **`conv3` 路径（如果存在）**：
  - 在将 `Y` 和 `X` 相加之前，会检查 `self.conv3` 是否存在。如果 `self.conv3` 存在（即 `use_1x1conv` 为 `True`），说明输入 `X` 的通道数或尺寸与 `Y` 不匹配，需要对输入 `X` 进行调整。此时，输入 `X` 会经过 `conv3` 卷积层，得到调整后的 `X`。

- **残差连接**：
  - 经过上述处理后，将调整后的 `X`（如果经过了 `conv3`）和 `Y` 逐元素相加，得到残差连接的结果。
  - 最后，对相加后的结果使用 ReLU 激活函数，得到最终的输出。

### 总结
`conv1` 和 `conv2` 是用于特征提取的主要路径，而 `conv3` 是为了保证残差连接能够正常进行（即输入 `X` 和经过 `conv1`、`conv2` 处理后的 `Y` 形状匹配）而存在的辅助路径。`conv3` 并不是在 `conv1`、`conv2` 前向传播之后才运行，而是在将 `Y` 和 `X` 相加之前，根据需要对 `X` 进行调整。 


```python

```
