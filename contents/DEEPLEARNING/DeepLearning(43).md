# 自然语言推断与数据集


```python
import os
import re
import torch
from torch import nn
from d2l import torch as d2l

#d2l.DATA_HUB['SNLI'] = (
#    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
#    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = '..\\data\\snli_1.0'  # 替换为你的实际路径
file_name = os.path.join(data_dir, 'snli_1.0_train.txt')
print("File exists:", os.path.exists(file_name))  # 检查文件是否存在
```

    File exists: True
    


```python
def read_snli(data_dir, is_train):
    """将SNLI数据集解析为前提、假设和标签"""
    def extract_text(s):
        # 删除我们不会使用的信息
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # 用一个空格替换两个或多个连续的空格
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] \
                in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```


```python
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('前提：', x0)
    print('假设：', x1)
    print('标签：', y)
```

    前提： A person on a horse jumps over a broken down airplane .
    假设： A person is training his horse for a competition .
    标签： 2
    前提： A person on a horse jumps over a broken down airplane .
    假设： A person is at a diner , ordering an omelette .
    标签： 1
    前提： A person on a horse jumps over a broken down airplane .
    假设： A person is outdoors , on a horse .
    标签： 0
    


```python
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

    [183416, 183187, 182764]
    [3368, 3237, 3219]
    


```python
class SNLIDataset(torch.utils.data.Dataset):
    """用于加载SNLI数据集的自定义数据集"""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + \
                all_hypothesis_tokens, min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```


```python
def load_data_snli(batch_size, num_steps=50):
    """下载SNLI数据集并返回数据迭代器和词表"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = '..\\data\\snli_1.0'
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```


```python
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

    read 549367 examples
    read 9824 examples
    




    18678




```python
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

# 针对序列级和词元级应用微调BERT


```python
import json
import multiprocessing
import os
import torch
from torch import nn
from d2l import torch as d2l
```


```python
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')
```


```python
import torch
import torch.nn as nn
import d2l

def get_tokens_and_segments(tokens_a, tokens_b=None):
    """
    处理输入序列的词元并生成片段索引。
    
    参数:
    tokens_a (list): 第一个句子的词元列表。
    tokens_b (list, optional): 第二个句子的词元列表（可选，用于句子对任务）。
    
    返回:
    tuple: 包含两个元素：
        - tokens (list): 处理后的词元序列，包含特殊标记<cls>和<SEP>。
        - segments (list): 片段索引列表，0表示属于第一个句子，1表示属于第二个句子。
    """
    # 添加句子开始标记<cls>和第一个句子结束标记<SEP>
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 初始化片段索引，0表示第一个句子及其前后的特殊标记
    segments = [0] * (len(tokens_a) + 2)
    # 如果存在第二个句子，则添加<SEP>和第二个句子的词元，并标记片段索引为1
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

class PositionWiseFFN(nn.Module):
    """
    逐位置前馈神经网络（Position-wise Feed-Forward Network）。
    对序列中的每个位置独立应用相同的全连接神经网络，用于特征变换。
    """
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        """
        初始化逐位置前馈网络。
        
        参数:
        ffn_num_input (int): 输入特征维度。
        ffn_num_hiddens (int): 隐藏层维度。
        ffn_num_outputs (int): 输出特征维度。
        **kwargs: 其他关键字参数，传递给父类。
        """
        super(PositionWiseFFN, self).__init__(**kwargs)
        # 第一个全连接层，将输入映射到隐藏层
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        # ReLU激活函数，引入非线性变换
        self.relu = nn.ReLU()
        # 第二个全连接层，将隐藏层映射到输出
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        """
        前向传播。
        
        参数:
        X (torch.Tensor): 输入张量，形状为(batch_size, seq_len, ffn_num_input)。
        
        返回:
        torch.Tensor: 输出张量，形状为(batch_size, seq_len, ffn_num_outputs)。
        """
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """
    残差连接与层归一化（Add & Norm）模块。
    用于将输入与子层的输出相加，并应用层归一化。
    """
    def __init__(self, normalized_shape, dropout, **kwargs):
        """
        初始化AddNorm模块。
        
        参数:
        normalized_shape (tuple): 层归一化的形状。
        dropout (float): Dropout概率，用于防止过拟合。
        **kwargs: 其他关键字参数，传递给父类。
        """
        super(AddNorm, self).__init__(**kwargs)
        # Dropout层，用于在子层输出中随机丢弃部分神经元
        self.dropout = nn.Dropout(dropout)
        # 层归一化层，对最后一个维度进行归一化
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        """
        前向传播。
        
        参数:
        X (torch.Tensor): 输入张量。
        Y (torch.Tensor): 子层的输出张量。
        
        返回:
        torch.Tensor: 应用残差连接和层归一化后的输出。
        """
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
    """
    BERT编码器块，包含多头注意力和逐位置前馈网络。
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        """
        初始化编码器块。
        
        参数:
        key_size (int): 键的特征维度。
        query_size (int): 查询的特征维度。
        value_size (int): 值的特征维度。
        num_hiddens (int): 隐藏层维度。
        norm_shape (tuple): 层归一化的形状。
        ffn_num_input (int): 前馈网络的输入维度。
        ffn_num_hiddens (int): 前馈网络的隐藏层维度。
        num_heads (int): 多头注意力的头数。
        dropout (float): Dropout概率。
        use_bias (bool): 是否在多头注意力中使用偏置。
        **kwargs: 其他关键字参数，传递给父类。
        """
        super(EncoderBlock, self).__init__(**kwargs)
        # 多头注意力层，用于捕捉序列中的上下文信息
        self.attention = d2l.MultiHeadAttention(
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            dropout=dropout,
            use_bias=use_bias
        )
        # 第一个AddNorm模块，用于多头注意力后的残差连接和层归一化
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # 逐位置前馈网络，用于进一步特征变换
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        # 第二个AddNorm模块，用于前馈网络后的残差连接和层归一化
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        """
        前向传播。
        
        参数:
        X (torch.Tensor): 输入张量，形状为(batch_size, seq_len, num_hiddens)。
        valid_lens (torch.Tensor): 有效长度张量，用于遮蔽填充部分。
        
        返回:
        torch.Tensor: 编码器块的输出张量。
        """
        # 多头注意力计算，并通过AddNorm处理残差连接和层归一化
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        # 前馈网络计算，并通过AddNorm处理残差连接和层归一化
        return self.addnorm2(Y, self.ffn(Y))

class BERTEncoder(nn.Module):
    """
    BERT编码器，包含嵌入层、位置编码和多个编码器块。
    """
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        """
        初始化BERT编码器。
        
        参数:
        vocab_size (int): 词汇表大小。
        num_hiddens (int): 隐藏层维度。
        norm_shape (tuple): 层归一化的形状。
        ffn_num_input (int): 前馈网络的输入维度。
        ffn_num_hiddens (int): 前馈网络的隐藏层维度。
        num_heads (int): 多头注意力的头数。
        num_layers (int): 编码器块的数量。
        dropout (float): Dropout概率。
        max_len (int): 最大序列长度，用于位置编码。
        key_size (int): 键的特征维度。
        query_size (int): 查询的特征维度。
        value_size (int): 值的特征维度。
        **kwargs: 其他关键字参数，传递给父类。
        """
        super(BERTEncoder, self).__init__(**kwargs)
        # 词元嵌入层，将词元索引转换为词向量
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        # 片段嵌入层，标记句子A和句子B（0或1）
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        # 按顺序添加多个编码器块
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 可学习的位置嵌入参数，用于捕捉序列的位置信息
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        """
        前向传播。
        
        参数:
        tokens (torch.Tensor): 词元索引张量，形状为(batch_size, seq_len)。
        segments (torch.Tensor): 片段索引张量，形状为(batch_size, seq_len)。
        valid_lens (torch.Tensor): 有效长度张量，形状为(batch_size,)。
        
        返回:
        torch.Tensor: 编码器的输出张量，形状为(batch_size, seq_len, num_hiddens)。
        """
        # 词元嵌入、片段嵌入和位置嵌入相加
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        # 依次通过每个编码器块
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

class MaskLM(nn.Module):
    """
    掩蔽语言模型（Masked Language Model）任务模块。
    用于预测输入中被遮蔽的词元。
    """
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        """
        初始化MaskLM模块。
        
        参数:
        vocab_size (int): 词汇表大小。
        num_hiddens (int): 隐藏层维度。
        num_inputs (int): 输入特征维度。
        **kwargs: 其他关键字参数，传递给父类。
        """
        super(MaskLM, self).__init__(**kwargs)
        # 多层感知机（MLP），用于将输入特征映射到词汇表维度
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size)
        )

    def forward(self, X, pred_positions):
        """
        前向传播。
        
        参数:
        X (torch.Tensor): 编码器的输出张量，形状为(batch_size, seq_len, num_inputs)。
        pred_positions (torch.Tensor): 被遮蔽位置的张量，形状为(batch_size, num_preds)。
        
        返回:
        torch.Tensor: 预测的词元概率分布，形状为(batch_size, num_preds, vocab_size)。
        """
        # 获取预测位置的数量和批次大小
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        # 生成批次索引，用于提取遮蔽位置的特征
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        # 提取遮蔽位置的特征
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        # 通过MLP预测被遮蔽的词元
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

class NextSentencePred(nn.Module):
    """
    下一句预测（Next Sentence Prediction）任务模块。
    用于判断两个句子是否为连续的上下文。
    """
    def __init__(self, num_inputs, **kwargs):
        """
        初始化NextSentencePred模块。
        
        参数:
        num_inputs (int): 输入特征维度（通常为BERT编码器输出的隐藏层维度）。
        **kwargs: 其他关键字参数，传递给父类。
        """
        super(NextSentencePred, self).__init__(**kwargs)
        # 全连接层，将输入特征映射到两个类别（是/否为下一句）
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        """
        前向传播。
        
        参数:
        X (torch.Tensor): 输入张量，形状为(batch_size, num_inputs)。
        
        返回:
        torch.Tensor: 预测的概率分布，形状为(batch_size, 2)。
        """
        return self.output(X)
```


```python
import torch
import torch.nn as nn

class BERTModel(nn.Module):
    """
    BERT模型，整合了编码器、掩蔽语言模型（MLM）和下一句预测（NSP）任务。
    
    BERT通过无监督预训练学习通用的文本表示，可通过微调适应多种下游任务。
    """
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        """
        初始化BERT模型。
        
        参数:
        vocab_size (int): 词汇表大小，决定词元嵌入的维度。
        num_hiddens (int): 编码器的隐藏层维度，即特征向量的维度。
        norm_shape (tuple): 层归一化的形状，通常为(num_hiddens,)。
        ffn_num_input (int): 前馈网络的输入维度。
        ffn_num_hiddens (int): 前馈网络的隐藏层维度。
        num_heads (int): 多头注意力的头数，用于并行捕捉不同子空间的信息。
        num_layers (int): 编码器块的数量，决定模型的深度。
        dropout (float): Dropout概率，用于防止过拟合。
        max_len (int): 输入序列的最大长度，用于位置嵌入的初始化。
        key_size (int): 多头注意力中键的特征维度。
        query_size (int): 多头注意力中查询的特征维度。
        value_size (int): 多头注意力中值的特征维度。
        hid_in_features (int): 下一句预测任务隐藏层的输入维度。
        mlm_in_features (int): 掩蔽语言模型任务的输入特征维度。
        nsp_in_features (int): 下一句预测任务的输入特征维度。
        """
        super(BERTModel, self).__init__()
        # 初始化BERT编码器，处理词元、片段和位置嵌入，并通过多层编码器块提取上下文特征
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        # 下一句预测任务的隐藏层，将编码器输出的CLS标记映射到num_hiddens维度
        self.hidden = nn.Sequential(
            nn.Linear(hid_in_features, num_hiddens),  # 线性变换
            nn.Tanh()  # Tanh激活函数引入非线性
        )
        # 掩蔽语言模型（MLM）模块，用于预测输入中被遮蔽的词元
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        # 下一句预测（NSP）模块，用于判断两个句子是否为连续的上下文
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        """
        BERT模型的前向传播。
        
        参数:
        tokens (torch.Tensor): 词元索引张量，形状为(batch_size, seq_len)。
        segments (torch.Tensor): 片段索引张量，0和1分别标记句子A和B，形状同上。
        valid_lens (torch.Tensor): 有效长度张量，用于遮蔽填充部分，形状为(batch_size,)。
        pred_positions (torch.Tensor): 被遮蔽词元的位置张量，形状为(batch_size, num_preds)。
        
        返回:
        tuple: 包含三个元素：
            - encoded_X (torch.Tensor): 编码器的输出，形状为(batch_size, seq_len, num_hiddens)。
            - mlm_Y_hat (torch.Tensor): MLM任务的预测结果，形状为(batch_size, num_preds, vocab_size)。
            - nsp_Y_hat (torch.Tensor): NSP任务的预测结果，形状为(batch_size, 2)。
        """
        # 通过BERT编码器获取上下文表示
        encoded_X = self.encoder(tokens, segments, valid_lens)
        # 如果提供了被遮蔽位置，则计算MLM任务的预测结果
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 提取CLS标记的特征（序列的第一个位置），用于下一句预测任务
        # 先通过隐藏层进行变换，再输入到NSP模块
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```


```python
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    """
    加载预训练的BERT模型，并适配当前配置。
    
    参数:
    pretrained_model (str): 预训练模型的名称或路径。
    num_hiddens (int): 模型隐藏层维度。
    ffn_num_hiddens (int): 前馈网络隐藏层维度。
    num_heads (int): 多头注意力的头数。
    num_layers (int): 编码器层数。
    dropout (float): Dropout概率。
    max_len (int): 输入序列的最大长度。
    devices (list): 可用的计算设备（如GPU列表）。
    
    返回:
    tuple: 包含加载了预训练参数的BERT模型和对应的词汇表。
    """
    # 下载并解压预训练模型的数据（假设d2l.download_extract已实现）
    data_dir = d2l.download_extract(pretrained_model)
    
    # 加载词汇表：从预训练模型的vocab.json文件中读取词表
    vocab = d2l.Vocab()
    # 读取JSON文件，将索引到词的映射加载到vocab.idx_to_token
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    # 构建词到索引的反向映射
    vocab.token_to_idx = {token: idx for idx, token in enumerate(vocab.idx_to_token)}
    
    # 初始化BERT模型，确保参数与预训练模型结构匹配
    bert = BERTModel(
        len(vocab),               # 词汇表大小，决定词元嵌入维度
        num_hiddens=num_hiddens,  # 隐藏层维度
        norm_shape=[256],         # 层归一化的形状，需与预训练模型一致
        ffn_num_input=256,        # 前馈网络输入维度
        ffn_num_hiddens=ffn_num_hiddens,  # 前馈网络隐藏层维度
        num_heads=4,              # 多头注意力的头数
        num_layers=2,             # 编码器层数
        dropout=0.2,              # Dropout概率
        max_len=max_len,          # 最大序列长度，用于位置嵌入
        key_size=256,             # 多头注意力中键的维度
        query_size=256,           # 多头注意力中查询的维度
        value_size=256,           # 多头注意力中值的维度
        hid_in_features=256,      # 下一句预测任务隐藏层的输入维度
        mlm_in_features=256,      # 掩蔽语言模型任务的输入特征维度
        nsp_in_features=256       # 下一句预测任务的输入特征维度
    )
    
    # 加载预训练参数文件（假设文件名为pretrained.params）
    pretrained_path = os.path.join(data_dir, 'pretrained.params')
    pretrained_dict = torch.load(pretrained_path)
    
    # 获取当前模型的状态字典（即模型各层的参数）
    model_dict = bert.state_dict()
    
    # 过滤预训练参数字典：只保留当前模型中存在的键
    # 这一步解决因模型结构差异导致的参数不匹配问题
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    
    # 将过滤后的预训练参数合并到当前模型的状态字典中
    model_dict.update(pretrained_dict)
    
    # 加载参数到模型中，strict=False表示忽略预训练中存在但当前模型没有的键
    # 同时，模型中存在但预训练中没有的键会被随机初始化
    bert.load_state_dict(model_dict, strict=False)
    
    return bert, vocab

# 获取可用的计算设备（如GPU）
devices = d2l.try_all_gpus()
# 调用函数加载预训练模型
bert, vocab = load_pretrained_model(
    'bert.small', 
    num_hiddens=256, 
    ffn_num_hiddens=512, 
    num_heads=4,
    num_layers=2, 
    dropout=0.1, 
    max_len=512, 
    devices=devices
)
```


```python
import torch
import torch.utils.data as data
import multiprocessing
import d2l

class SNLIBERTDataset(data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        """
        初始化SNLI/BERT数据集类，用于处理文本蕴含任务的数据。
        
        参数:
        dataset (list): 包含前提（premise）、假设（hypothesis）和标签的数据集。
                        假设格式为：[前提句子列表, 假设句子列表, 标签列表]
        max_len (int): 输入序列的最大长度，用于截断和填充。
        vocab (Vocab, optional): 词汇表对象，用于将词元转换为索引。
        """
        # 1. 对前提和假设句子进行分词处理
        # 使用d2l.tokenize对句子列表进行分词，转为小写
        # zip(*[...]) 将前提和假设的分词结果按顺序配对
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]

        # 2. 初始化标签为张量
        self.labels = torch.tensor(dataset[2])
        
        # 3. 保存词汇表和最大长度
        self.vocab = vocab
        self.max_len = max_len
        
        # 4. 预处理所有前提-假设对，生成词元索引、片段索引和有效长度
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        
        # 打印加载的样本数量
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        """
        多进程预处理所有前提-假设对，生成模型所需的输入格式。
        
        参数:
        all_premise_hypothesis_tokens (list): 包含前提和假设词元对的列表。
        
        返回:
        tuple: 包含词元索引张量、片段索引张量和有效长度张量。
        """
        # 1. 创建多进程池（4个工作进程）以加速处理
        pool = multiprocessing.Pool(4)
        
        # 2. 并行处理每个前提-假设对
        # 使用pool.map将_mp_worker函数应用到所有输入对上
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        
        # 3. 解包处理结果
        all_token_ids = [token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        
        # 4. 将结果转换为张量
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        """
        多进程工作函数，处理单个前提-假设对。
        
        参数:
        premise_hypothesis_tokens (tuple): 包含前提和假设词元列表的元组。
        
        返回:
        tuple: 处理后的词元索引、片段索引和有效长度。
        """
        # 1. 解包前提和假设的词元列表
        p_tokens, h_tokens = premise_hypothesis_tokens
        
        # 2. 截断词元对，确保总长度不超过max_len - 3
        # （预留3个位置给<cls>、<sep>和<SEP>）
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        
        # 3. 生成带特殊标记的词元序列和片段索引
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        
        # 4. 转换为词元索引并填充到max_len长度
        # 使用词汇表将词元转换为索引，不足部分用<pad>填充
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] * (self.max_len - len(tokens))
        
        # 5. 填充片段索引，不足部分用0填充
        segments = segments + [0] * (self.max_len - len(segments))
        
        # 6. 记录有效长度（不包含填充部分）
        valid_len = len(tokens)
        
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        """
        截断前提和假设的词元列表，使其总长度不超过max_len - 3。
        
        参数:
        p_tokens (list): 前提的词元列表。
        h_tokens (list): 假设的词元列表。
        """
        # 循环删除较长句子的末尾词元，直到总长度符合要求
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()  # 删除前提的最后一个词元
            else:
                h_tokens.pop()  # 删除假设的最后一个词元

    def __getitem__(self, idx):
        """
        根据索引获取数据样本。
        
        参数:
        idx (int): 样本索引。
        
        返回:
        tuple: 包含模型输入（词元索引、片段索引、有效长度）和标签。
        """
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        """
        获取数据集的样本数量。
        
        返回:
        int: 样本数量。
        """
        return len(self.all_token_ids)
```


```python
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = '..\\data\\snli_1.0' 
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```


```python
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))

net = BERTClassifier(bert)
```


```python
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```


```python

```
