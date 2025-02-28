# 多层感知机
本部分内容的包在d2l中没有，需要调用上节课自己写的函数才能跑通。


```python
import torch
from torch import nn
from d2l import torch as d2l
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

## 初始化模型参数
实现一个但隐藏层，包含256个隐藏单元


```python
num_inputs, num_outputs, num_hiddens = 784, 10, 256
# num_hiddens：表示隐藏层的神经元数量，这里设置为 256。隐藏层用于学习输入数据的特征表示。
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
#W1：第一层的权重矩阵，它是一个形状为 (num_inputs, num_hiddens) 的二维张量，即 (784, 256)。torch.randn 函数用于生成一个服从标准正态分布（均值为 0，标准差为 1）的随机张量，然后将其乘以 0.01 来缩小权重的初始值，以避免梯度爆炸或梯度消失问题。
# nn.Parameter 是 PyTorch 中的一个特殊类，用于将张量标记为模型的可训练参数，requires_grad=True 表示需要对这些参数进行梯度计算，以便在训练过程中更新它们。
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True)) #第一层
#b1：第一层的偏置向量，它是一个形状为 (num_hiddens,) 的一维张量，即 (256,)。初始值全部设置为 0。同样，使用 nn.Parameter 标记为可训练参数。
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True)) #输出层

params = [W1, b1, W2, b2]
```

## ReLU 激活函数


```python
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)
```

## 实现模型


```python
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X@W1 + b1) # 这里“@”代表矩阵乘法
    return (H@W2 + b2)
```


```python
loss = nn.CrossEntropyLoss(reduction='none')
```

## 训练模型


```python
## 这是上一章的代码，然而在d2l中似乎没有封装进去，先加载下面代码再运行微调后的代码，可以通过。

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: 
        #len(y_hat.shape) > 1 检查 y_hat 是否为二维数组； 确认 y_hat 的第二维（通常是类别数）大于1，意味着有多个类别
        y_hat = y_hat.argmax(axis = 1)
        # 使用 argmax(axis=1) 找到每一行中最大值的索引，这个索引对应预测的类别。argmax 返回的是每行最大值的列索引，即预测的类别标签。
    cmp = y_hat.type(y.dtype) == y #逐元素比较预测类别与真实类别，生成一个布尔数组。由于等式运算符“==”对数据类型很敏感， 因此我们将y_hat的数据类型转换为与y的数据类型一致。
    return float(cmp.type(y.dtype).sum())  # 结果是一个包含0（错）和1（对）的张量。
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2) #初始化为包含两个元素的列表 [0.0, 0.0]。这两个元素分别用于累加正确预测的数量和总样本数量。
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel()) #y.numel()：获取当前批次的样本数量。
        #metric.add(...)：将当前批次的准确率和样本数量累加到 metric 中
    return metric[0] / metric[1]  #将累加的正确预测数量除以总样本数量，得到整个数据集的平均准确率
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n #初始化一个包含 n 个 0.0 的列表，用于存储累积的值
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)] #对每一对 (a, b)，将 a（累加器中的当前值）和 b（新值）相加，并将结果转换成浮点数。
    def reset(self):
        self.data = [0.0] * len(self.data) #将 self.data 中的所有元素重置为 0.0，以便重新开始累积
    def __getitem__(self, idx):
        return self.data[idx] #允许使用索引访问 Accumulator 中的特定元素，例如 metric[0] 访问第一个累积值。

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):  
        net.train()  #net.train() 将模型设置为训练模式
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            # 在训练神经网络时，需要通过反向传播计算损失函数对模型参数的梯度，然后使用优化器根据这些梯度更新参数。优化器决定了每次更新的方向和幅度，从而影响模型的收敛速度和最终性能。
            updater.zero_grad()
            l.mean().backward()
            updater.step()
    #如果 updater 是 torch.optim.Optimizer 的实例，使用PyTorch内置的优化器进行梯度清零、反向传播和参数更新。
    #否则，使用自定义的 updater 函数进行反向传播和参数更新。
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())  #累加当前批次的损失总和、正确预测的数量和样本总数。
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型"""
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])  #用于可视化训练过程，显示每个epoch的训练损失、训练准确率和测试准确率。
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    #断言：确保训练损失和准确率在合理范围内
    assert train_loss < 0.5, train_loss  #用于检查某个条件是否为真，如果条件为假，则抛出 AssertionError 异常
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y) #将真实标签转换为可读的字符串形式
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1)) #计算模型的预测结果并转换为可读的字符串形式。
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)] #生成每个样本的真实标签和预测标签的组合。
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])  #显示前 n 个样本的图像及其真实标签和预测标签
```


```python
from torch import nn
from d2l import torch as d2l
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
```


    
![svg](2.1-1output_11_0.svg)
    



```python
predict_ch3(net, test_iter)
```


    
![svg](2.1-2output_12_0.svg)
    


# 简洁实现


```python
import torch
from torch import nn
from d2l import torch as d2l
```


```python
# 模型
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10))
#nn.Sequential：是一个容器，用于按顺序堆叠多个神经网络层。
#nn.Flatten()：将输入的多维张量展平为一维张量，因为全连接层的输入需要是一维的。在 Fashion-MNIST 数据集中，输入图像的尺寸是 28x28，展平后得到长度为 784 的一维向量。
#nn.Linear(784, 256)：全连接层，将输入的 784 维向量映射到 256 维的隐藏层。
#nn.ReLU()：ReLU 激活函数，用于引入非线性，增强模型的表达能力。
#nn.Linear(256, 10)：全连接层，将 256 维的隐藏层输出映射到 10 维的输出层，对应 Fashion-MNIST 数据集中的 10 个类别。
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
#init_weights 函数：用于初始化模型的权重。对于模型中的每个线性层（nn.Linear），使用 nn.init.normal_ 函数将其权重初始化为均值为 0，标准差为 0.01 的正态分布。
#net.apply(init_weights)：将 init_weights 函数应用到模型 net 的所有层上，实现权重的初始化。
```


```python
# 实现
batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=lr)
#nn.CrossEntropyLoss(reduction='none')：交叉熵损失函数，用于计算模型预测结果与真实标签之间的损失。reduction='none' 表示不对损失进行降维操作，返回每个样本的损失值。
#torch.optim.SGD(net.parameters(), lr=lr)：随机梯度下降（SGD）优化器，用于更新模型的参数。net.parameters() 表示模型的所有可训练参数，lr 是学习率。
```


```python
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
#d2l.train_ch3：调用 d2l 库中的训练函数，传入模型 net、训练集迭代器 train_iter、测试集迭代器 test_iter、损失函数 loss、训练轮数 num_epochs 和优化器 trainer，开始训练模型。
#在训练过程中，该函数会在每个训练轮结束后评估模型在测试集上的性能，并输出训练损失、训练准确率和测试准确率等信息。
```


    
![svg](2.1-3output_17_0.svg)
    



```python

```
