# 批量归一化

为什么需要批量规范化层呢？让我们来回顾一下训练神经网络时出现的一些实际挑战。

首先，数据预处理的方式通常会对最终结果产生巨大影响。
回想一下我们应用多层感知机来预测房价的例子（ :numref:`sec_kaggle_house`）。
使用真实数据时，我们的第一步是标准化输入特征，使其平均值为0，方差为1。
直观地说，这种标准化可以很好地与我们的优化器配合使用，因为它可以将参数的量级进行统一。

第二，对于典型的多层感知机或卷积神经网络。当我们训练时，中间层中的变量（例如，多层感知机中的仿射变换输出）可能具有更广的变化范围：不论是沿着从输入到输出的层，跨同一层中的单元，或是随着时间的推移，模型参数的随着训练更新变幻莫测。
批量规范化的发明者非正式地假设，这些变量分布中的这种偏移可能会阻碍网络的收敛。
直观地说，我们可能会猜想，如果一个层的可变值是另一层的100倍，这可能需要对学习率进行补偿调整。

第三，更深层的网络很复杂，容易过拟合。
这意味着正则化变得更加重要。

## 批量规范化层

回想一下，批量规范化和其他层之间的一个关键区别是，由于批量规范化在完整的小批量上运行，因此我们不能像以前在引入其他层时那样忽略批量大小。
我们在下面讨论这两种情况：全连接层和卷积层，他们的批量规范化实现略有不同。

### 全连接层

通常，我们将批量规范化层置于全连接层中的仿射变换和激活函数之间。
设全连接层的输入为x，权重参数和偏置参数分别为$\mathbf{W}$和$\mathbf{b}$，激活函数为$\phi$，批量规范化的运算符为$\mathrm{BN}$。
那么，使用批量规范化的全连接层的输出的计算详情如下：

$$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

回想一下，均值和方差是在应用变换的"相同"小批量上计算的。

### 卷积层

同样，对于卷积层，我们可以在卷积层之后和非线性激活函数之前应用批量规范化。
当卷积有多个输出通道时，我们需要对这些通道的“每个”输出执行批量规范化，每个通道都有自己的拉伸（scale）和偏移（shift）参数，这两个参数都是标量。
假设我们的小批量包含$m$个样本，并且对于每个通道，卷积的输出具有高度$p$和宽度$q$。
那么对于卷积层，我们在每个输出通道的$m \cdot p \cdot q$个元素上同时执行每个批量规范化。
因此，在计算平均值和方差时，我们会收集所有空间位置的值，然后在给定通道内应用相同的均值和方差，以便在每个空间位置对值进行规范化。

### 预测过程中的批量规范化

正如我们前面提到的，批量规范化在训练模式和预测模式下的行为通常不同。
首先，将训练好的模型用于预测时，我们不再需要样本均值中的噪声以及在微批次上估计每个小批次产生的样本方差了。
其次，例如，我们可能需要使用我们的模型对逐个样本进行预测。
一种常用的方法是通过移动平均估算整个训练数据集的样本均值和方差，并在预测时使用它们得到确定的输出。
可见，和暂退法一样，批量规范化层在训练模式和预测模式下的计算结果也是不一样的。


```python
import torch
from torch import nn
from d2l import torch as d2l
```

 归一化公式：$$\(\hat{x} = \frac{x - \mu_{moving}}{\sqrt{\sigma_{moving}^2 + \epsilon}}\)，其中\(\mu_{moving}\)是移动均值，\(\sigma_{moving}^2\)是移动方差，\(\epsilon\)$$是一个小的常数防止除零

公式：$$\(\mu = \frac{1}{m}\sum_{i = 1}^{m}x_i\)$$，这里m是样本数量，在dim=0上求均值，即对所有样本的每个特征求均值

公式：$$\(\sigma^2 = \frac{1}{m}\sum_{i = 1}^{m}(x_i - \mu)^2\)$$，这里m是样本数量，先计算每个样本的特征与均值的差的平方，再在dim=0上求均值

公式：$$\(\mu = \frac{1}{n \times h \times w}\sum_{i = 1}^{n}\sum_{j = 1}^{h}\sum_{k = 1}^{w}x_{ijk}\)$$，其中n是样本数量，h和w是特征图的高和宽

公式：$$\(\sigma^2 = \frac{1}{n \times h \times w}\sum_{i = 1}^{n}\sum_{j = 1}^{h}\sum_{k = 1}^{w}(x_{ijk} - \mu)^2\)$$

公式：$$\(\mu_{moving}^{new} = \beta \times \mu_{moving}^{old} + (1 - \beta) \times \mu\)，其中\(\beta\)是动量参数，\(\mu_{moving}^{old}\)是旧的移动均值，\(\mu\)$$是当前计算的均值


```python
import torch


def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # 通过is_grad_enabled来判断当前模式是训练模式还是预测模式
    # 在PyTorch中，如果启用了梯度计算（训练模式），is_grad_enabled()返回True；否则（预测模式）返回False
    if not torch.is_grad_enabled():
        # 如果是在预测模式下，直接使用传入的移动平均所得的均值和方差进行归一化
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 断言输入数据X的维度只能是2或4，因为这里只处理全连接层（维度为2）和二维卷积层（维度为4）的情况
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # 使用全连接层的情况，计算特征维上的均值
            # 公式：\(\mu = \frac{1}{m}\sum_{i = 1}^{m}x_i\)，这里m是样本数量，在dim=0上求均值，即对所有样本的每个特征求均值
            mean = X.mean(dim=0)
            # 使用全连接层的情况，计算特征维上的方差
            # 公式：\(\sigma^2 = \frac{1}{m}\sum_{i = 1}^{m}(x_i - \mu)^2\)，这里m是样本数量，先计算每个样本的特征与均值的差的平方，再在dim=0上求均值
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 使用二维卷积层的情况，计算通道维上（axis=1）的均值，keepdim=True保持维度以便后续广播运算
            # 公式：\(\mu = \frac{1}{n \times h \times w}\sum_{i = 1}^{n}\sum_{j = 1}^{h}\sum_{k = 1}^{w}x_{ijk}\)，其中n是样本数量，h和w是特征图的高和宽
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            # 使用二维卷积层的情况，计算通道维上（axis=1）的方差，keepdim=True保持维度以便后续广播运算
            # 公式：\(\sigma^2 = \frac{1}{n \times h \times w}\sum_{i = 1}^{n}\sum_{j = 1}^{h}\sum_{k = 1}^{w}(x_{ijk} - \mu)^2\)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下，用当前计算得到的均值和方差做标准化
        # 公式：\(\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\)，其中\(\mu\)是当前计算的均值，\(\sigma^2\)是当前计算的方差，\(\epsilon\)是一个小的常数防止除零
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值
        # 公式：\(\mu_{moving}^{new} = \beta \times \mu_{moving}^{old} + (1 - \beta) \times \mu\)，其中\(\beta\)是动量参数，\(\mu_{moving}^{old}\)是旧的移动均值，\(\mu\)是当前计算的均值
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        # 更新移动平均的方差
        # 公式：\(\sigma_{moving}^{2, new} = \beta \times \sigma_{moving}^{2, old} + (1 - \beta) \times \sigma^2\)，其中\(\beta\)是动量参数，\(\sigma_{moving}^{2, old}\)是旧的移动方差，\(\sigma^2\)是当前计算的方差
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    # 对归一化后的数据进行缩放和平移
    # 公式：\(y = \gamma \times \hat{x} + \beta\)，其中\(\gamma\)和\(\beta\)是可学习的参数，\(\hat{x}\)是归一化后的数据
    Y = gamma * X_hat + beta
    # 返回归一化并经过缩放和平移后的数据Y，以及更新后的移动均值和移动方差（data属性是为了获取tensor中的数据，避免梯度追踪）
    return Y, moving_mean.data, moving_var.data

```


```python
class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        """
        初始化BatchNorm类。
        :param num_features: 完全连接层的输出数量或卷积层的输出通道数。
        :param num_dims: 2表示完全连接层，4表示卷积层
        """
        # 调用父类nn.Module的构造函数，确保父类的初始化逻辑被执行
        super().__init__()
        # 根据输入的num_dims确定形状
        if num_dims == 2:
            # 对于全连接层，形状为 (1, num_features)
            shape = (1, num_features)
        else:
            # 对于卷积层，形状为 (1, num_features, 1, 1)，这样可以在通道维度上进行批量归一化
            shape = (1, num_features, 1, 1)
        # 参与求梯度和迭代的拉伸和偏移参数，分别初始化成1和0
        # nn.Parameter将张量包装成可训练的参数，会被自动添加到模型的参数列表中，在反向传播时更新
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量初始化为0和1
        # 移动均值和移动方差在训练过程中通过滑动平均更新，不需要反向传播更新，所以不使用nn.Parameter
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
    def forward(self, X):
        """
        前向传播函数，定义了数据在模型中的流动过程。
        :param X: 输入数据
        :return: 经过批量归一化处理后的数据
        """
        # 如果X不在内存上，将moving_mean和moving_var
        # 复制到X所在显存上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 保存更新过的moving_mean和moving_var
        # 调用之前定义的batch_norm函数进行批量归一化操作
        # eps是一个小的常数，防止除零错误，momentum是移动平均的动量参数
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps = 1e-5, momentum = 0.9)
        return Y
```

##  使用批量规范化层的 LeNet


```python
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
```


```python
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```


    
![svg](output_9_0.svg)
    



```python
net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))
```

在批量归一化（Batch Normalization）以及其他涉及指数移动平均（Exponential Moving Average, EMA）的场景中，`momentum` 参数起着重要的作用，下面为你详细介绍其作用和参数设置方法。

### `momentum` 的作用

在批量归一化中，我们需要在训练过程中维护一个移动均值（`moving_mean`）和移动方差（`moving_var`），用于在预测阶段对输入数据进行归一化。`momentum` 就是用于更新这两个统计量的一个超参数，其更新公式如下：

设当前批次计算得到的均值为 $\mu_{batch}$，方差为 $\sigma_{batch}^2$，上一轮的移动均值为 $\mu_{moving}^{old}$，移动方差为 $\sigma_{moving}^{2, old}$，`momentum` 为 $\beta$，则更新后的移动均值 $\mu_{moving}^{new}$ 和移动方差 $\sigma_{moving}^{2, new}$ 计算公式为：

$$
\mu_{moving}^{new} = \beta \times \mu_{moving}^{old} + (1 - \beta) \times \mu_{batch}
$$

$$
\sigma_{moving}^{2, new} = \beta \times \sigma_{moving}^{2, old} + (1 - \beta) \times \sigma_{batch}^2
$$

`momentum` 参数的作用主要体现在以下几个方面：

1. **平滑统计量**：`momentum` 可以对不同批次计算得到的均值和方差进行平滑处理。因为在训练过程中，每个批次的数据可能存在一定的波动，如果直接使用当前批次的统计量进行预测，可能会导致预测结果不稳定。通过指数移动平均的方式，结合 `momentum` 参数，可以让移动均值和移动方差更加稳定，减少批次间的波动影响。

2. **保留历史信息**：较大的 `momentum` 值会让模型更多地保留历史批次的统计信息，因为当前批次的统计量在更新中所占的权重较小；而较小的 `momentum` 值则会让模型更关注当前批次的统计信息，历史信息的权重相对较小。

### `momentum` 参数的设置方法

`momentum` 参数的设置并没有一个固定的最优值，它通常需要根据具体的数据集、模型结构和训练任务进行调整。以下是一些常见的设置建议：

1. **默认值**：在很多深度学习框架中，`momentum` 的默认值通常设置为 0.9 或 0.99。这个值是经过大量实验验证的，在大多数情况下可以取得较好的效果。例如，在 PyTorch 的 `nn.BatchNorm` 系列模块中，`momentum` 的默认值就是 0.1（这里的实现是 $1 - \beta$ 的形式）。

2. **数据集大小**：
    - 如果数据集较大，每个批次的数据能够较好地代表整个数据集的分布，此时可以适当增大 `momentum` 值，让模型更多地保留历史信息，减少当前批次数据波动的影响。
    - 如果数据集较小，每个批次的数据可能无法很好地代表整个数据集的分布，此时可以适当减小 `momentum` 值，让模型更关注当前批次的统计信息，以便更快地适应数据的变化。

3. **训练阶段**：
    - 在训练初期，数据的分布可能还不稳定，此时可以适当减小 `momentum` 值，让模型更快地适应数据的变化。
    - 在训练后期，数据的分布逐渐稳定，此时可以适当增大 `momentum` 值，让模型更多地保留历史信息，提高移动均值和移动方差的稳定性。



```python

```

