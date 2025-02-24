# Linear Regression and Basic Optimization Algorithms

## 线性模型

线性模型可以看作单层神经网络。Input Layer是各个因素，Output layer是要预测的结果。

神经网络源于神经科学，神经元：输入-计算-输出-下一层。

对于特征集合$\mathbf{X}$，预测值$\hat{\mathbf{y}} \in \mathbb{R}^n$
可以通过矩阵-向量乘法表示为：

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b$$

衡量预测的质量：

当样本$i$的预测值为$\hat{y}^{(i)}$，其相应的真实标签为$y^{(i)}$时，
平方误差可以定义为以下公式：

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$

