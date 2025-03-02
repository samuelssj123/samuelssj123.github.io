# 层和块

## 块

块的基本功能：

1. 将输入数据作为其前向传播函数的参数。
1. 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个全连接的层接收一个20维的输入，但是返回一个维度为256的输出。
1. 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
1. 存储和访问前向传播计算所需的参数。
1. 根据需要初始化模型参数。

步骤：

自定义的块包含一个多层感知机，其具有256个隐藏单元的隐藏层和一个10维输出层。
注意，下面的`MLP`类继承了表示块的类。
我们的实现只需要提供我们自己的构造函数（Python中的`__init__`函数）和前向传播函数。


```python
import torch
from torch import nn
from torch.nn import functional as F
```


```python
class MLP(nn.Module): # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        super().__init__() # 调用MLP的父类Module的构造函数来执行必要的初始化。 # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        self.hidden = nn.Linear(20, 256)  # 隐藏层
        self.out = nn.Linear(256, 10) # 输出层
    def forward(self, X): # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
        return self.out(F.relu(self.hidden(X)))

#实例化多层感知机的层，然后在每次调用前向传播函数时调用这些层
net = MLP()
X = torch.rand(2, 20)
net(X)  
```




    tensor([[ 0.0449,  0.1696,  0.4724,  0.1419,  0.0487,  0.0049,  0.2130,  0.0896,
              0.0034,  0.1126],
            [-0.0162,  0.2894,  0.4408,  0.1272,  0.0722,  0.0030,  0.0974, -0.0155,
              0.0711,  0.1646]], grad_fn=<AddmmBackward0>)



## 顺序块

`Sequential`的设计是为了把其他模块串起来。
为了构建我们自己的简化的`MySequential`，
我们只需要定义两个关键函数：

1. 一种将块逐个追加到列表中的函数；
1. 一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。

`_modules`的主要优点是：
在模块的参数初始化过程中，
系统知道在`_modules`字典中查找需要初始化参数的子块。


```python
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        for block in self._modules.values(): # OrderedDict保证了按照成员添加的顺序遍历它们
            X = block(X)
        return X
```


```python
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)
```




    tensor([[ 0.0906, -0.1660, -0.2350,  0.2038, -0.0606, -0.1721, -0.1209,  0.1682,
             -0.1134,  0.1745],
            [-0.0593, -0.0611, -0.1282,  0.1752, -0.0794, -0.1417, -0.1194,  0.0529,
             -0.0876,  0.0832]], grad_fn=<AddmmBackward0>)



## 在前向传播函数中执行代码

有时我们可能希望合并既不是上一层的结果也不是可更新参数的项，我们称之为*常数参数*（constant parameter）。
例如，我们需要一个计算函数
$f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^\top \mathbf{x}$的层，
其中$\mathbf{x}$是输入，
$\mathbf{w}$是参数，
$c$是某个在优化过程中没有更新的指定常量。


在这个`FixedHiddenMLP`模型中，我们实现了一个隐藏层，
其权重（`self.rand_weight`）在实例化时被随机初始化，之后为常量。
这个权重不是一个模型参数，因此它永远不会被反向传播更新。
然后，神经网络将这个固定层的输出通过一个全连接层。

注意，在返回输出之前，模型做了一些不寻常的事情：
它运行了一个while循环，在$L_1$范数大于$1$的条件下，
将输出向量除以$2$，直到它满足条件为止。
最后，模型返回了`X`中所有项的和。
注意，此操作可能不会常用于在任何实际任务中，
我们只展示如何将任意代码集成到神经网络计算的流程中。


```python
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad = False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1) # 使用创建的常量参数以及relu和mm函数
        X = self.linear(X) # 复用全连接层。这相当于两个全连接层共享参数
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()
        #while X.abs().sum() > 1: 和 X /= 2：使用一个 while 循环检查 X 中所有元素的绝对值之和是否大于 1。如果大于 1，则将 X 中的所有元素都除以 2，直到绝对值之和小于等于 1。
        #这一步是为了对 X 的值进行缩放，避免数值过大。
net = FixedHiddenMLP()
net(X)
```




    tensor(-0.0681, grad_fn=<SumBackward0>)




```python
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)
    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
chimera(X)
```




    tensor(-0.1706, grad_fn=<SumBackward0>)



# 参数管理


```python
import torch
from torch import nn
```

## 参数访问


```python
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
net(X)
print(net[2].state_dict())
```

    OrderedDict([('weight', tensor([[ 0.1800,  0.3440, -0.1547,  0.0388,  0.3353,  0.2302,  0.0796, -0.1810]])), ('bias', tensor([-0.2753]))])
    

## 目标参数

首先我们需要访问底层的数值，下面的代码从第二个全连接层（即第三个神经网络层）提取偏置， 提取后返回的是一个参数类实例，并进一步访问该参数的值。


```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
```

    <class 'torch.nn.parameter.Parameter'>
    Parameter containing:
    tensor([-0.2753], requires_grad=True)
    tensor([-0.2753])
    


```python
net[2].weight.grad == None
```




    True



## 一次性访问所有参数


```python
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])
```

    ('weight', torch.Size([8, 4])) ('bias', torch.Size([8]))
    ('0.weight', torch.Size([8, 4])) ('0.bias', torch.Size([8])) ('2.weight', torch.Size([1, 8])) ('2.bias', torch.Size([1]))
    


```python
net.state_dict()['2.bias'].data # 网络参数的访问
```




    tensor([-0.2753])



## 从嵌套块收集参数


```python
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())
def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
```




    tensor([[-0.0935],
            [-0.0935]], grad_fn=<AddmmBackward0>)




```python
print(rgnet)
```

    Sequential(
      (0): Sequential(
        (block 0): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
        (block 1): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
        (block 2): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
        (block 3): Sequential(
          (0): Linear(in_features=4, out_features=8, bias=True)
          (1): ReLU()
          (2): Linear(in_features=8, out_features=4, bias=True)
          (3): ReLU()
        )
      )
      (1): Linear(in_features=4, out_features=1, bias=True)
    )
    


```python
rgnet[0][1][0].bias.data #层是分层嵌套的，所以我们也可以像通过嵌套列表索引一样访问它们
```




    tensor([ 0.4905, -0.2165,  0.2255,  0.1872, -0.2191, -0.1004, -0.2786, -0.3786])



## 参数初始化


```python
# 内置初始化  下面的代码将所有权重参数初始化为标准差为0.01的高斯随机变量， 且将偏置参数设置为0。
def init_normal(m):  #
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean = 0, std = 0.1)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]

# 下面我们使用Xavier初始化方法初始化第一个神经网络层， 然后将第三个神经网络层初始化为常量值42。
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

    tensor([ 0.4112, -0.1844,  0.0477, -0.4106])
    tensor([[42., 42., 42., 42., 42., 42., 42., 42.]])
    

### 自定义初始化
有时，深度学习框架没有提供我们需要的初始化方法。
在下面的例子中，我们使用以下的分布为任意权重参数$w$定义初始化方法：

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ 可能性 } \frac{1}{4} \\
            0    & \text{ 可能性 } \frac{1}{2} \\
        U(-10, -5) & \text{ 可能性 } \frac{1}{4}
    \end{cases}
\end{aligned}
$$



```python
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

    Init weight torch.Size([8, 4])
    Init weight torch.Size([1, 8])
    




    tensor([[5.7185, 7.3152, 7.6288, -0.0000],
            [5.5001, -0.0000, -0.0000, -0.0000]], grad_fn=<SliceBackward0>)



### 参数绑定
有时我们希望在多个层间共享参数： 我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。


```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

    tensor([True, True, True, True, True, True, True, True])
    tensor([True, True, True, True, True, True, True, True])
    

# 自定义层

## 不带参数的层


```python
import torch
import torch.nn.functional as F
from torch import nn


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
layer(torch.FloatTensor([1, 2, 3, 4, 5]))
```




    tensor([-2., -1.,  0.,  1.,  2.])




```python
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
```


```python
Y = net(torch.rand(4, 8))
Y.mean()
```




    tensor(6.5193e-09, grad_fn=<MeanBackward0>)



## 带参数的层

自定义版本的全连接层，该层需要两个参数，一个用于表示权重，另一个用于表示偏置项。
在此实现中，我们使用修正线性单元作为激活函数。
该层需要输入参数：`in_units`和`units`，分别表示输入数和输出数。


```python
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)
```


```python
# 实例化
linear = MyLinear(5, 3)
linear.weight
```




    Parameter containing:
    tensor([[ 0.1664, -0.2702, -0.2802],
            [ 0.2520, -1.4697, -0.1221],
            [ 0.1085,  1.3945,  1.0281],
            [ 0.0270, -0.0160, -0.0964],
            [ 2.3701,  0.1581, -0.3037]], requires_grad=True)




```python
linear(torch.rand(2, 5))
```




    tensor([[1.5973, 0.0000, 0.0000],
            [0.3006, 0.0000, 0.0000]])




```python
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
net(torch.rand(2, 64))
```




    tensor([[0.],
            [0.]])



# 读写文件

## 加载和保存张量


```python
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')
```


```python
x2 = torch.load('x-file')
x2
```




    tensor([0, 1, 2, 3])




```python
y = torch.zeros(4)
torch.save([x, y],'x-files')
x2, y2 = torch.load('x-files')
(x2, y2)
```




    (tensor([0, 1, 2, 3]), tensor([0., 0., 0., 0.]))




```python
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
mydict2
```




    {'x': tensor([0, 1, 2, 3]), 'y': tensor([0., 0., 0., 0.])}



## 加载和保存模型参数


```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
```


```python
torch.save(net.state_dict(), 'mlp.params')
```


```python
clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
clone.eval()
```




    MLP(
      (hidden): Linear(in_features=20, out_features=256, bias=True)
      (output): Linear(in_features=256, out_features=10, bias=True)
    )




```python
Y_clone = clone(X)
Y_clone == Y
```




    tensor([[True, True, True, True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True, True, True, True]])




```python

```
