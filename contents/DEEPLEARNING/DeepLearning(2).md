# Softmax回归+损失函数+图像分类数据集

## 图像分类数据集
选择图像分类中广泛使用的数据集：Fashion-MNIST数据集


```python
%matplotlib inline
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
```

#### 读取数据集
通过框架中的内置函数将Fashion-MNIST数据集下载并读取到内存中。
Fashion-MNIST由10个类别的图像组成，
每个类别由*训练数据集*（train dataset）中的6000张图像
和*测试数据集*（test dataset）中的1000张图像组成。
因此，训练集和测试集分别包含60000和10000张图像。
测试数据集不会用于训练，只用于评估模型性能。
为了简洁起见，本书将高度$h$像素、宽度$w$像素图像的形状记为$h \times w$或（$h$,$w$）。


```python
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
```


```python
mnist_train[0][0].shape
```




    torch.Size([1, 28, 28])



Fashion-MNIST中包含的10个类别，分别为t-shirt（T恤）、trouser（裤子）、pullover（套衫）、dress（连衣裙）、coat（外套）、sandal（凉鞋）、shirt（衬衫）、sneaker（运动鞋）、bag（包）和ankle boot（短靴）。
以下函数用于在数字标签索引及其文本名称之间进行转换。



```python
def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

### 可视化


```python
# 可视化
def show_images(imgs, num_rows, num_cols, titles = None, scale = 1.5):
    figsize = (num_cols * scale, num_rows * scale) # 计算子图布局的整体图像大小
    # 创建一个包含多个子图的图形对象
    # num_rows 和 num_cols 分别指定子图的行数和列数
    # figsize 是图形的整体大小
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize = figsize)
    # 将二维的子图数组扁平化，方便后续遍历
    axes = axes.flatten()
    # 遍历每个子图和对应的图像
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img): # 判断图像是否为 PyTorch 张量
            ax.imshow(img.numpy()) # 如果是张量，将其转换为 NumPy 数组后显示
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False) # 隐藏子图的 x 轴刻度
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```


```python
# 从 mnist_train 数据集中获取一个批次的图像和标签
# batch_size=18 表示每个批次包含 18 个样本
X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
# X.reshape(18, 28, 28) 将图像数据从 (18, 784) 重塑为 (18, 28, 28)，以适应图像显示
# get_fashion_mnist_labels(y) 用于获取每个图像对应的标签名称
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y));
```


    
![svg](output_10_0.svg)
    


### 读取小批量
为了使我们在读取训练集和测试集时更容易，我们使用内置的数据迭代器，而不是从零开始创建。
回顾一下，在每次迭代中，数据加载器每次都会[**读取一小批量数据，大小为`batch_size`**]。
通过内置数据迭代器，我们可以随机打乱了所有样本，从而无偏见地读取小批量。


```python
# 读取小批量
batch_size = 256
def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4
train_iter = data.DataLoader(mnist_train, batch_size, shuffle = True, num_workers = get_dataloader_workers())

timer = d2l.Timer()
for X, y in train_iter:
    continue
f'{timer.stop():.2f} sec'
```




    '8.13 sec'



### 封装数据集
现在我们[**定义`load_data_fashion_mnist`函数**]，用于获取和读取Fashion-MNIST数据集。
这个函数返回训练集和验证集的数据迭代器。
此外，这个函数还接受一个可选参数`resize`，用来将图像大小调整为另一种形状。


```python
def load_data_fashion_mnist(batch_size, resize=None):  
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()] # 定义数据转换操作列表，初始包含将数据转换为张量的操作
#将输入的图像数据（通常是 PIL 图像，即 Python Imaging Library 中的图像对象）转换为 PyTorch 中的张量（Tensor）。
    #并且在转换过程中，会将图像像素值的范围从原来的 [0, 255]（对于 8 位图像）归一化到 [0, 1]。

#如果 resize 有具体的值（通常是一个整数，表示要将图像调整为的尺寸，比如 224 表示将图像调整为 224x224 的大小），
    #则说明用户希望对图像进行大小调整的预处理操作。
    if resize: #如果指定了图像大小调整的尺寸
        trans.insert(0, transforms.Resize(resize)) # 在转换操作列表的开头插入调整图像大小的操作
#如果 resize 参数为真（即指定了图像大小调整的尺寸），就在 trans 列表的开头插入 transforms.Resize(resize) 这个调整图像大小的转换操作。
    #这样，在后续对图像进行处理时，会先调整图像大小，然后再将其转换为张量。
    
    trans = transforms.Compose(trans) # 将多个数据转换操作组合成一个复合操作
#前面已经将需要的转换操作（如调整图像大小和转换为张量）添加到了 trans 列表中，这里通过 transforms.Compose(trans) 将列表 trans 中的所有转换操作组合起来，
    #形成一个单一的复合转换操作对象。

    
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))
```


```python
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:   
    print(X.shape, X.dtype, y.shape, y.dtype)
    break
```

    torch.Size([32, 1, 64, 64]) torch.float32 torch.Size([32]) torch.int64
    

## Softmax回归从零开始

**为了避免函数间的干扰，运行下面代码时请Restart！**


```python
import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

### 初始化模型参数
在softmax回归中，我们的输出与类别一样多。
(**因为我们的数据集有10个类别，所以网络输出维度为10**)。
因此，权重将构成一个$784 \times 10$的矩阵，
偏置将构成一个$1 \times 10$的行向量。


```python
num_inputs = 784
num_outputs = 10

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
```

### 定义softmax操作
如果`X`是一个形状为`(2, 3)`的张量，我们对列进行求和，
 则结果将是一个具有形状`(3,)`的向量。
 当调用`sum`运算符时，我们可以指定保持在原始张量的轴数，而不折叠求和的维度。
 这将产生一个具有形状`(1, 3)`的二维张量。


```python
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
X.sum(0, keepdim=True), X.sum(1, keepdim=True)
```




    (tensor([[5., 7., 9.]]),
     tensor([[ 6.],
             [15.]]))



[**实现softmax**]由三个步骤组成：

1. 对每个项求幂（使用`exp`）；
1. 对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数；
1. 将每一行除以其规范化常数，确保结果的和为1。

表达式：

(**
$$
\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(\mathbf{X}_{ij})}{\sum_k \exp(\mathbf{X}_{ik})}.
$$
**)


```python
#任何随机输入，我们将每个元素变成一个非负数。 此外，依据概率原理，每行总和为1。
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制

#keepdims=True 参数确保结果的维度与 X_exp 保持一致，即结果仍然是一个二维数组，每行对应一个求和结果。
```


```python
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)
#矩阵中的非常大或非常小的元素可能造成数值上溢或下溢，但我们没有采取措施来防止这点。
```




    (tensor([[0.2239, 0.1927, 0.0382, 0.4539, 0.0913],
             [0.0145, 0.0093, 0.7928, 0.1357, 0.0476]]),
     tensor([1.0000, 1.0000]))



### 定义模型


```python
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)
```

### 定义损失函数
交叉熵采用真实标签的预测概率的负对数似然。
我们创建一个数据样本y_hat，其中包含2个样本在3个类别的预测概率， 以及它们对应的标签y。 有了y，我们知道在第一个样本中，第一类是正确的预测； 而在第二个样本中，第三类是正确的预测。 然后使用y作为y_hat中概率的索引， 我们选择第一个样本中第一个类的概率和第二个样本中第三个类的概率。


```python
y = torch.tensor([0, 2]) # 真实标签：第一个样本属于类别0，第二个样本属于类别2
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]]) # 预测概率
y_hat[[0, 1], y] #使用 y 作为索引，从 y_hat 中选择每个样本对应真实类别的预测概率。
```




    tensor([0.1000, 0.5000])



交叉熵损失用于衡量预测概率分布与真实分布之间的差异。对于多分类问题，交叉熵损失的计算公式为：

$$
\text{Cross Entropy} = -\sum_{i=1}^{N} \log(p_i)
$$

其中：
-  N  是样本数量。
-  p_i 是第  i 个样本对应真实类别的预测概率。



```python
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])
    #运行顺序是：生成一个从0到样本数减1的序列；选择了每个样本对应真实类别的预测概率；对这些概率取自然对数并取负值，得到每个样本的交叉熵损失。

cross_entropy(y_hat, y)
```




    tensor([2.3026, 0.6931])



### 分类精度


```python
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: 
        #len(y_hat.shape) > 1 检查 y_hat 是否为二维数组； 确认 y_hat 的第二维（通常是类别数）大于1，意味着有多个类别
        y_hat = y_hat.argmax(axis = 1)
        # 使用 argmax(axis=1) 找到每一行中最大值的索引，这个索引对应预测的类别。argmax 返回的是每行最大值的列索引，即预测的类别标签。
    cmp = y_hat.type(y.dtype) == y #逐元素比较预测类别与真实类别，生成一个布尔数组。由于等式运算符“==”对数据类型很敏感， 因此我们将y_hat的数据类型转换为与y的数据类型一致。
    return float(cmp.type(y.dtype).sum())  # 结果是一个包含0（错）和1（对）的张量。
```


```python
accuracy(y_hat, y) / len(y)
```




    0.5




```python
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
```


```python
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n #初始化一个包含 n 个 0.0 的列表，用于存储累积的值
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)] #对每一对 (a, b)，将 a（累加器中的当前值）和 b（新值）相加，并将结果转换成浮点数。
    def reset(self):
        self.data = [0.0] * len(self.data) #将 self.data 中的所有元素重置为 0.0，以便重新开始累积
    def __getitem__(self, idx):
        return self.data[idx] #允许使用索引访问 Accumulator 中的特定元素，例如 metric[0] 访问第一个累积值。
```


```python
evaluate_accuracy(net, test_iter)
```




    0.1154



### 训练


```python
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
```


```python
class Animator:  
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
#**xlabel**: X 轴标签，默认为 None。
#**ylabel**: Y 轴标签，默认为 None。
#**legend**: 图例列表，默认为空列表 []。
#**xlim**: X 轴范围，默认为 None（自动调整）。
#**ylim**: Y 轴范围，默认为 None（自动调整）。
#**xscale**: X 轴比例类型，默认为 'linear'（线性）。
#**yscale**: Y 轴比例类型，默认为 'linear'（线性）。
#**fmts**: 绘制线条的格式列表，默认为 ('-', 'm--', 'g-.', 'r:')，分别表示实线、点划线、点线等。
#**nrows**: 子图的行数，默认为 1。
#**ncols**: 子图的列数，默认为 1。
#**figsize**: 图表的大小，默认为 (3.5, 2.5) 英寸。
        # 增量地绘制多条线
        if legend is None:
            legend = [] #如果未提供图例，则初始化为空列表
        d2l.use_svg_display() #使用 SVG 格式显示图表，适用于 Jupyter Notebook 等环境
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
    #使用 matplotlib 创建一个包含多个子图的图表。如果只有一个子图，axes 将被调整为单元素列表，方便后续操作。
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    #使用 lambda 函数封装坐标轴配置，以便后续调用。d2l.set_axes 是一个自定义函数，用于设置坐标轴的各种属性。
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
    # 确保 x 和 y 都是列表形式。如果 y 不是列表，则将其转换为包含单个元素的列表。如果 x 不是列表，则将其复制 n 次，以匹配 y 的长度。
        
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
    #如果 self.X 或 self.Y 尚未初始化，则创建对应长度的空列表，用于存储每一条线的 X 和 Y 数据。
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        #遍历 x 和 y 的每一对数据点，如果两者都不为 None，则将其分别添加到对应的 self.X 和 self.Y 列表中。
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
    #cla() 方法清除当前子图的内容。使用 zip 将 self.X、self.Y 和 self.fmts 对应起来，逐条绘制每一条线的图形。
        self.config_axes() #调用 self.config_axes() 方法，根据初始化时设置的参数配置坐标轴。
        display.display(self.fig) #使用 display.display(self.fig) 显示图表。
        display.clear_output(wait=True) #使用 display.clear_output(wait=True) 清除之前的输出，以避免图表重叠。wait=True 表示在新的输出生成前等待，防止闪烁。
```


```python
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
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
```


```python
lr = 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)
#学习率：lr = 0.1 定义了学习率。
# 返回一个 d2l.sgd 函数，用于执行随机梯度下降（SGD）更新。d2l.sgd 是一个自定义的SGD实现，接受参数列表、学习率和批次大小。
```


```python
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```


    
![svg](output_42_0.svg)
    


## 预测


```python
def predict_ch3(net, test_iter, n=6):  #@save
    """预测标签"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y) #将真实标签转换为可读的字符串形式
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1)) #计算模型的预测结果并转换为可读的字符串形式。
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)] #生成每个样本的真实标签和预测标签的组合。
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])  #显示前 n 个样本的图像及其真实标签和预测标签

predict_ch3(net, test_iter)
```


    
![svg](output_44_0.svg)
    


# Softmax的简洁实现


```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```


```python
## 初始化模型参数
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);
```

softmax函数$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$，
其中$\hat y_j$是预测的概率分布。
$o_j$是未规范化的预测$\mathbf{o}$的第$j$个元素。
如果$o_k$中的一些数值非常大，
那么$\exp(o_k)$可能大于数据类型容许的最大数字，即*上溢*（overflow）。
这将使分母或分子变为`inf`（无穷大），
最后得到的是0、`inf`或`nan`（不是数字）的$\hat y_j$。
在这些情况下，我们无法得到一个明确定义的交叉熵值。

解决这个问题的一个技巧是：
在继续softmax计算之前，先从所有$o_k$中减去$\max(o_k)$。
这里可以看到每个$o_k$按常数进行的移动不会改变softmax的返回值：

$$
\begin{aligned}
\hat y_j & =  \frac{\exp(o_j - \max(o_k))\exp(\max(o_k))}{\sum_k \exp(o_k - \max(o_k))\exp(\max(o_k))} \\
& = \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}.
\end{aligned}
$$


在减法和规范化步骤之后，可能有些$o_j - \max(o_k)$具有较大的负值。
由于精度受限，$\exp(o_j - \max(o_k))$将有接近零的值，即*下溢*（underflow）。
这些值可能会四舍五入为零，使$\hat y_j$为零，
并且使得$\log(\hat y_j)$的值为`-inf`。
反向传播几步后，我们可能会发现自己面对一屏幕可怕的`nan`结果。

尽管我们要计算指数函数，但我们最终在计算交叉熵损失时会取它们的对数。
通过将softmax和交叉熵结合在一起，可以避免反向传播过程中可能会困扰我们的数值稳定性问题。
如下面的等式所示，我们避免计算$\exp(o_j - \max(o_k))$，
而可以直接使用$o_j - \max(o_k)$，因为$\log(\exp(\cdot))$被抵消了。

$$
\begin{aligned}
\log{(\hat y_j)} & = \log\left( \frac{\exp(o_j - \max(o_k))}{\sum_k \exp(o_k - \max(o_k))}\right) \\
& = \log{(\exp(o_j - \max(o_k)))}-\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)} \\
& = o_j - \max(o_k) -\log{\left( \sum_k \exp(o_k - \max(o_k)) \right)}.
\end{aligned}
$$


```python
loss = nn.CrossEntropyLoss(reduction='none')
```


```python
## 优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.1)
```


```python
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)

## 这段代码报错，d2l似乎没有封装这个函数，如果要成功运行，还是要把涉及到的参数和函数在完整版里面重新运行一遍，才能跑通。
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[41], line 2
          1 num_epochs = 10
    ----> 2 d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    

    AttributeError: module 'd2l.torch' has no attribute 'train_ch3'



```python

```
