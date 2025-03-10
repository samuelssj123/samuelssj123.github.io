# 多GPU训练

一般来说，$k$个GPU并行训练过程如下：

* 在任何一次训练迭代中，给定的随机的小批量样本都将被分成$k$个部分，并均匀地分配到GPU上；
* 每个GPU根据分配给它的小批量子集，计算模型参数的损失和梯度；
* 将$k$个GPU中的局部梯度聚合，以获得当前小批量的随机梯度；
* 聚合梯度被重新分发到每个GPU中；
* 每个GPU使用这个小批量随机梯度，来更新它所维护的完整的模型参数集。


```python
%matplotlib inline
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

## [**简单网络**]


```python
# 初始化模型参数
scale = 0.01
W1 = torch.randn(size=(20, 1, 3, 3)) * scale
b1 = torch.zeros(20)
W2 = torch.randn(size=(50, 20, 5, 5)) * scale
b2 = torch.zeros(50)
W3 = torch.randn(size=(800, 128)) * scale
b3 = torch.zeros(128)
W4 = torch.randn(size=(128, 10)) * scale
b4 = torch.zeros(10)
params = [W1, b1, W2, b2, W3, b3, W4, b4]

# 定义模型
def lenet(X, params):
    h1_conv = F.conv2d(input=X, weight=params[0], bias=params[1])
    h1_activation = F.relu(h1_conv)
    h1 = F.avg_pool2d(input=h1_activation, kernel_size=(2, 2), stride=(2, 2))
    h2_conv = F.conv2d(input=h1, weight=params[2], bias=params[3])
    h2_activation = F.relu(h2_conv)
    h2 = F.avg_pool2d(input=h2_activation, kernel_size=(2, 2), stride=(2, 2))
    h2 = h2.reshape(h2.shape[0], -1)
    h3_linear = torch.mm(h2, params[4]) + params[5]
    h3 = F.relu(h3_linear)
    y_hat = torch.mm(h3, params[6]) + params[7]
    return y_hat

# 交叉熵损失函数
loss = nn.CrossEntropyLoss(reduction='none')
```

## 数据同步


```python
# 给到参数，知道分发去哪个GPU上
def get_params(params, device):
    new_params = [p.clone().to(device) for p in params]
    for p in new_params:
        p.requires_grad_() #做梯度
    return new_params

new_params = get_params(params, d2l.try_gpu(0))
print('bl weight:', new_params[1])
print('bl grad:', new_params[1].grad)
```

    bl weight: tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           requires_grad=True)
    bl grad: None
    

`allreduce`函数将所有向量相加，并将结果广播给所有GPU.相加必须在同一个GPU上，然后再返回去。


```python
# 定义 allreduce 函数，用于在多个 GPU 设备之间进行数据的聚合和广播操作
def allreduce(data):
    # 第一步：将所有设备上的数据累加到第一个设备上
    for i in range(1, len(data)):
        # 将 data[i] 移动到 data[0] 所在的设备上，并累加到 data[0] 中
        data[0][:] += data[i].to(data[0].device)
    # 第二步：将第一个设备上累加后的数据广播到其他所有设备上
    for i in range(1, len(data)):
        # 将 data[0] 移动到 data[i] 所在的设备上，并赋值给 data[i]
        data[i][:] = data[0].to(data[i].device)

# 创建一个包含两个张量的列表 data，每个张量在不同的 GPU 设备上，且值分别为 1 和 2
data = [torch.ones((1, 2), device = d2l.try_gpu(i)) * (i + 1) for i in range(2)]
# 打印 allreduce 操作之前的数据
print('allreduce之前：\n', data[0], '\n', data[1])
# 调用 allreduce 函数进行数据的聚合和广播操作
allreduce(data)
# 打印 allreduce 操作之后的数据
print('allreduce之后：\n', data[0], '\n', data[1])
```

    allreduce之前：
     tensor([[1., 1.]]) 
     tensor([[2., 2.]])
    allreduce之后：
     tensor([[3., 3.]]) 
     tensor([[3., 3.]])
    

## 数据分发


```python
data = torch.arange(20).reshape(4, 5)
devices = [torch.device('cuda:0'), torch.device('cuda:1')]
split = nn.parallel.scatter(data, devices)
print('input :', data)
print('load into', devices)
print('output:', split)
```


```python
def split_batch(X, y, devices):
    """将X和y拆分到多个设备上"""
    assert X.shape[0] == y.shape[0]
    return (nn.parallel.scatter(X, devices),
            nn.parallel.scatter(y, devices))
```

## 训练


```python
# 在一个小批量上实现多 GPU 训练的函数
def train_batch(X, y, device_params, devices, lr):
    # 将输入数据 X 和标签 y 分割到不同的 GPU 设备上
    X_shards, y_shards = split_batch(X, y, devices)
    # 在每个 GPU 上分别计算损失
    ls = [loss(lenet(X_shard, device_W), y_shard).sum()
          for X_shard, y_shard, device_W in zip(
              X_shards, y_shards, device_params)]
    # 反向传播在每个 GPU 上分别执行
    for l in ls:
        l.backward()
    # 将每个 GPU 的所有梯度相加，并将其广播到所有 GPU
    with torch.no_grad():
        # 遍历每个模型参数
        for i in range(len(device_params[0])):
            # 收集所有 GPU 上对应参数的梯度
            allreduce(
                [device_params[c][i].grad for c in range(len(devices))])
    # 在每个 GPU 上分别更新模型参数
    for param in device_params:
        # 使用随机梯度下降法更新参数，这里使用全尺寸的小批量
        d2l.sgd(param, lr, X.shape[0]) 
```

[**定义训练函数**]。
与前几章中略有不同：训练函数需要分配GPU并将所有模型参数复制到所有设备。
显然，每个小批量都是使用`train_batch`函数来处理多个GPU。
我们只在一个GPU上计算模型的精确度，而让其他GPU保持空闲，尽管这是相对低效的，但是使用方便且代码简洁。


```python
# 训练函数，用于在多个 GPU 上训练模型
def train(num_gpus, batch_size, lr):
    # 加载 Fashion-MNIST 数据集的训练集和测试集
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # 获取指定数量的可用 GPU 设备
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]
    # 将模型参数复制到 num_gpus 个 GPU 上
    device_params = [get_params(params, d) for d in devices]
    # 训练的总轮数
    num_epochs = 10
    # 创建一个动画对象，用于可视化测试准确率的变化
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])
    # 创建一个计时器对象，用于记录训练时间
    timer = d2l.Timer()
    # 开始训练循环
    for epoch in range(num_epochs):
        # 开始计时
        timer.start()
        # 遍历训练集的每个小批量
        for X, y in train_iter:
            # 为单个小批量执行多 GPU 训练
            train_batch(X, y, device_params, devices, lr)
            # 同步所有 GPU 设备，确保所有操作完成
            torch.cuda.synchronize()
        # 停止计时
        timer.stop()
        # 在 GPU0 上评估模型
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(
            lambda x: lenet(x, device_params[0]), test_iter, devices[0]),))
    # 打印测试精度、每轮训练的平均时间和使用的 GPU 设备
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
          f'在{str(devices)}')
```


```python
train(num_gpus=1, batch_size=256, lr=0.2)
```


```python
train(num_gpus=2, batch_size=256, lr=0.2)
```

# 简洁实现


```python
def resnet18(num_classes, in_channels=1):
    """稍加修改的ResNet-18模型"""
    def resnet_block(in_channels, out_channels, num_residuals,
                     first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(d2l.Residual(in_channels, out_channels,
                                        use_1x1conv=True, strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 该模型使用了更小的卷积核、步长和填充，而且删除了最大汇聚层
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU())
    net.add_module("resnet_block1", resnet_block(
        64, 64, 2, first_block=True))
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    net.add_module("resnet_block4", resnet_block(256, 512, 2))
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1,1)))
    net.add_module("fc", nn.Sequential(nn.Flatten(),
                                       nn.Linear(512, num_classes)))
    return net
```


```python
net = resnet18(10)
# 获取GPU列表
devices = d2l.try_all_gpus()
# 我们将在训练代码实现中初始化网络
```

## [**训练**]

如前所述，用于训练的代码需要执行几个基本功能才能实现高效并行：

* 需要在所有设备上初始化网络参数；
* 在数据集上迭代时，要将小批量数据分配到所有设备上；
* 跨设备并行计算损失及其梯度；
* 聚合梯度，并相应地更新参数。


```python
# 定义训练函数，用于在多个 GPU 上训练指定的神经网络模型
def train(net, num_gpus, batch_size, lr):
    # 加载 Fashion-MNIST 数据集的训练集和测试集迭代器，batch_size 为每个小批量的数据样本数量
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # 获取指定数量的可用 GPU 设备列表，这里 num_gpus 表示要使用的 GPU 数量
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]

    # 定义一个初始化权重的函数，用于对网络中的线性层（nn.Linear）和卷积层（nn.Conv2d）进行权重初始化
    def init_weights(m):
        # 检查当前层是否为线性层或卷积层
        if type(m) in [nn.Linear, nn.Conv2d]:
            # 使用正态分布初始化该层的权重，标准差为 0.01
            nn.init.normal_(m.weight, std=0.01)

    # 对网络中的所有层应用权重初始化函数
    net.apply(init_weights)

    # 使用 PyTorch 的 nn.DataParallel 模块将模型并行化到多个 GPU 上
    # device_ids 参数指定要使用的 GPU 设备列表
    net = nn.DataParallel(net, device_ids=devices)

    # 定义优化器，使用随机梯度下降（SGD）算法，对网络的所有参数进行优化，学习率为 lr
    trainer = torch.optim.SGD(net.parameters(), lr)

    # 定义损失函数，使用交叉熵损失，适用于多分类问题
    loss = nn.CrossEntropyLoss()

    # 创建一个计时器对象，用于记录训练时间
    timer = d2l.Timer()
    # 定义训练的总轮数
    num_epochs = 10

    # 创建一个动画对象，用于可视化测试准确率随训练轮数的变化
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])

    # 开始训练循环，共进行 num_epochs 轮训练
    for epoch in range(num_epochs):
        # 将模型设置为训练模式，启用一些在训练时需要的特殊层，如 Dropout 等
        net.train()
        # 开始计时当前轮次的训练时间
        timer.start()

        # 遍历训练集的每个小批量数据
        for X, y in train_iter:
            # 清零优化器中的梯度信息，避免梯度累积
            trainer.zero_grad()
            # 将输入数据 X 和标签 y 移动到第一个 GPU 设备上
            X, y = X.to(devices[0]), y.to(devices[0])
            # 前向传播，计算模型的输出，并通过损失函数计算损失值
            l = loss(net(X), y)
            # 反向传播，计算梯度
            l.backward()
            # 根据计算得到的梯度，使用优化器更新模型的参数
            trainer.step()

        # 停止计时当前轮次的训练时间
        timer.stop()
        # 在测试集上评估模型的准确率，并将结果添加到动画对象中，用于后续可视化
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))

    # 打印最终的测试精度、每轮训练的平均时间以及使用的 GPU 设备列表
    print(f'测试精度：{animator.Y[0][-1]:.2f}，{timer.avg():.1f}秒/轮，'
          f'在{str(devices)}')
```


```python
train(net, num_gpus=1, batch_size=256, lr=0.1)
```


```python
train(net, num_gpus=2, batch_size=512, lr=0.2)
```
