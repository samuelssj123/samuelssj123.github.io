# 线性回归 + 基础优化算法

# 线性回归从零开始实现


```python
%matplotlib inline
import random
import torch
from d2l import torch as d2l
```

### 生成数据集
我们使用线性模型参数$\mathbf{w} = [2, -3.4]^\top$、$b = 4.2$
和噪声项$\epsilon$生成数据集及其标签：

$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$


```python
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) #该张量的形状信息
    return X, y.reshape((-1, 1)) #使用 reshape((-1, 1)) 将 y 转换为列向量的形式

true_w = torch.tensor([2, -3.4]) # true_w 是一个包含两个元素的张量，分别对应两个特征的权重
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 10000)
```

注意，[**`features`中的每一行都包含一个二维数据样本，
`labels`中的每一行都包含一维标签值（一个标量）**]。


```python
print('features:', features[0], '\nlabel:', labels[0])
d2l.set_figsize() #设置绘图的图形大小
d2l.plt.scatter(features[:,(1)].detach().numpy(), labels.detach().numpy(), 1);
#scatter 散点图
#: 表示选取所有行，(1) 表示选取第二列（在 Python 中索引从 0 开始计数），所以这会获取 features 张量中所有样本的第二个特征值，作为散点图的 x 轴数据。
#.detach() 是 PyTorch 中张量的方法，用于返回一个新的张量，这个新张量和原张量共享数据，但会从计算图中分离出来，即不再计算梯度。这在将张量数据转换为 numpy 数组时很常用，因为 numpy 数组不支持计算梯度。
```

    features: tensor([-1.5131, -0.5141]) 
    label: tensor([2.9240])
    


    
![svg](output_6_1.svg)
    


# 读取数据集
[**定义一个`data_iter`函数，
该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为`batch_size`的小批量**]。
每个小批量包含一组特征和标签。


```python
def data_iter(batch_size, features, labels):
    """
    定义一个数据迭代器函数，用于按批次生成数据。

    参数:
    batch_size (int): 每个批次的数据样本数量。
    features (torch.Tensor): 特征矩阵，形状为 (样本数量, 特征维度)。
    labels (torch.Tensor): 标签向量，形状为 (样本数量, 1) 或 (样本数量,)。

    返回:
    一个生成器，每次迭代返回一个批次的特征和标签。
    """
    num_examples = len(features)
    indices = list(range(num_examples)) # 创建一个包含所有样本索引的列表
    random.shuffle(indices) # 随机打乱样本索引列表，shuffle用于随机打乱列表中的元素顺序
    for i in range(0, num_examples, batch_size):  # 按批次遍历样本索引
        batch_indices = torch.tensor(indices[i: min(i+ batch_size, num_examples)]) # 获取当前批次的样本索引张量
                        #min(i + batch_size, num_examples) 用于确保切片不会超出样本总数的范围
        yield features[batch_indices], labels[batch_indices] # 生成当前批次的特征和标签
```


```python
batch_size = 10 #小批量运算
for X,y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break  #避免输出太多，打印完第一个看看就行了。
```

    tensor([[-0.4380,  1.1015],
            [ 0.0894,  0.0131],
            [-0.1448, -1.9261],
            [-1.9886, -1.1573],
            [-0.2117, -0.2338],
            [ 1.1591,  1.1163],
            [-1.2666, -0.7379],
            [ 0.5283, -0.3690],
            [-0.4135, -0.1274],
            [-0.4790,  0.8121]]) 
     tensor([[-0.4330],
            [ 4.3530],
            [10.4584],
            [ 4.1571],
            [ 4.5582],
            [ 2.7293],
            [ 4.1896],
            [ 6.5156],
            [ 3.8044],
            [ 0.4767]])
    

## 初始化模型参数和定义模型


```python
w = torch.normal(0, 1, size = (2,1), requires_grad = True)
#size=(2, 1)：指定生成张量的形状，这里是 2 行 1 列
#requires_grad=True：是一个布尔值参数，设置为 True 表示这个张量在计算过程中需要计算梯度，以便后续进行反向传播更新参数。
#只有设置了 requires_grad=True 的张量才会被计算梯度，未设置的张量会被忽略
b = torch.zeros(1, requires_grad = True)
#创建一个全零张量，创建一个全零张量
```


```python
def linreg(X, w, b):
    return torch.matmul(X, w) + b #matmul 矩阵乘法
```

## 定义损失函数、优化算法


```python
def squared_loss(y_hat, y):
    """均方损失（Mean Squared Error, MSE）"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
#reshape:将 y 的形状重塑为与 y_hat 相同，以便进行逐元素的减法运算
```


```python
def sgd(params, lr, batch_size): 
    #params：一个包含需要更新的参数（如 w 和 b）的列表
    #lr：学习率（learning rate）
    #batch_size：批次大小
    """随机梯度下降（Stochastic Gradient Descent, SGD）"""
    with torch.no_grad():
#torch.no_grad() 是 PyTorch 中的上下文管理器，在这个上下文块内的操作不会记录梯度，即不会为操作创建计算图，
        #这样可以节省内存和计算资源，并且在不需要计算梯度的情况下（如参数更新过程）使用。
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()  #用于将张量的梯度清零。在每次参数更新后，需要将梯度清零，以便下一次计算梯度时不会受到之前梯度的影响。
```

## 训练

在每次迭代中，我们读取一小批量训练样本，并通过我们的模型来获得一组预测。
计算完损失后，我们开始反向传播，存储每个参数的梯度。
最后，我们调用优化算法`sgd`来更新模型参数。

* 初始化参数
* 重复以下训练，直到完成
    * 计算梯度$\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
    * 更新参数$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$


```python
# 设置学习率，控制每次参数更新的步长
# 学习率过大可能导致模型无法收敛，过小则会使训练速度变慢
lr = 0.003  #可以把学习率改为3、0.01尝试一下
# 设置训练的轮数，即整个数据集被模型遍历的次数
# 增加轮数可以让模型有更多机会学习数据中的模式，但也可能导致过拟合
num_epochs = 5 #可以把学习率改为10、100尝试一下

# 外层循环，控制训练的轮数
for epoch in range(num_epochs):
    # 内层循环，使用 data_iter 函数按批次遍历数据集
    # data_iter 函数会将数据集打乱并按批次返回特征 X 和标签 y
    for X, y in data_iter(batch_size, features, labels):
        # 计算当前批次数据的预测值
        # linreg 函数实现了线性回归模型，根据输入特征 X、权重 w 和偏置 b 计算预测值
        y_hat = linreg(X, w, b)
        
        # 计算当前批次数据的损失
        # squared_loss 函数计算预测值 y_hat 与真实标签 y 之间的均方损失
        l = squared_loss(y_hat, y)

        # 将当前批次的损失求和，得到一个标量损失值
        # 因为 PyTorch 的反向传播需要一个标量值作为起点
        loss_sum = l.sum()

        # 进行反向传播，计算损失关于模型参数（w 和 b）的梯度
        # 由于 w 和 b 在定义时设置了 requires_grad=True，PyTorch 会自动跟踪它们的操作并计算梯度
        loss_sum.backward()

        # 使用随机梯度下降算法更新模型参数
        # sgd 函数接受模型参数列表 [w, b]、学习率 lr 和批次大小 batch_size 作为输入
        # 它会根据计算得到的梯度更新参数，并将梯度清零，为下一次迭代做准备
        sgd([w, b], lr, batch_size)
        
    # 在每个轮次结束时，计算整个训练集上的损失
    # 使用 torch.no_grad() 上下文管理器，禁用梯度计算，因为在计算训练集损失时不需要更新参数
    with torch.no_grad():
        train_l = squared_loss(linreg(features, w, b), labels) # 计算整个训练集的预测值、损失
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

    epoch 1, loss 0.036595
    epoch 2, loss 0.000138
    epoch 3, loss 0.000051
    epoch 4, loss 0.000051
    epoch 5, loss 0.000051
    


```python
print(f' w的估计误差：{true_w - w.reshape(true_w.shape)}')
print(f' b的估计误差：{true_b - b}')
```

     w的估计误差：tensor([-4.7469e-04, -1.5736e-05], grad_fn=<SubBackward0>)
     b的估计误差：tensor([-0.0008], grad_fn=<RsubBackward1>)
    

# 线性回归的简洁实现


```python
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l
```


```python
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```


```python
def load_array(data_arrays, batch_size, is_train = True):
    """构造一个PyTorch数据迭代器"""
    
```
