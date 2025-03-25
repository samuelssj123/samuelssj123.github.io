# BERT


```python
import torch
from torch import nn
from d2l import torch as d2l
```


```python
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0和1分别标记片段A和B
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```


```python
class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```


```python
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```


```python
class EncoderBlock(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
             norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
             dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
    # 假设 MultiHeadAttention 只需要 num_hiddens, num_heads, dropout, use_bias
        self.attention = d2l.MultiHeadAttention(
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            dropout=dropout,
            use_bias=use_bias
        )
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
```


```python
class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```


```python
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)
```


```python
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```




    torch.Size([2, 8, 768])




```python
class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```


```python
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```




    torch.Size([2, 3, 10000])




```python
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```




    torch.Size([6])




```python
class NextSentencePred(nn.Module):
    """BERT的下一句预测任务"""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # X的形状：(batchsize,num_hiddens)
        return self.output(X)
```


```python
encoded_X = torch.flatten(encoded_X, start_dim=1)
# NSP的输入形状:(batchsize，num_hiddens)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```




    torch.Size([2, 2])




```python
nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```




    torch.Size([2])




```python
class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

# 用于预训练BERT的数据集


```python
import os
import random
import torch
from d2l import torch as d2l
```


```python
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r', encoding='utf-8') as f:  # 指定编码格式为 utf-8
        lines = f.readlines()
    # 大写字母转换为小写字母
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    return paragraphs
```


```python
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # paragraphs是三重列表的嵌套
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```


```python
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    """
    从段落中生成用于BERT下一句预测（NSP）任务的训练数据。
    
    参数:
    paragraph (list): 当前段落的句子列表，每个元素是一个句子的词元列表。
    paragraphs (list): 所有段落的列表，用于生成负样本（随机替换句子对）。
    vocab (Vocab): 词汇表，用于将词元转换为索引。
    max_len (int): 输入序列的最大长度，超过此长度的句子对将被忽略。
    
    返回:
    list: 包含三元组的列表，每个三元组为 (tokens, segments, is_next)，分别表示词元序列、片段索引和是否为下一句的标签。
    """
    # 初始化一个空列表，用于存储生成的NSP数据
    nsp_data_from_paragraph = []
    
    # 遍历当前段落中的每个句子，除了最后一个（因为需要与下一个句子配对）
    for i in range(len(paragraph) - 1):
        # 调用_get_next_sentence函数，生成当前句子对的词元和是否为下一句的标签
        # tokens_a: 第一个句子的词元列表
        # tokens_b: 第二个句子的词元列表（可能是下一个句子或随机选择的句子）
        # is_next: 标签，1表示tokens_b是tokens_a的下一句，0表示随机替换的句子
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        
        # 检查合并后的句子长度是否超过max_len
        # 加3是因为需要添加'<cls>'（1个）和两个'<sep>'（2个）词元
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            # 如果超过最大长度，跳过当前句子对
            continue
        
        # 调用get_tokens_and_segments函数，处理词元和生成片段索引
        # tokens: 合并后的词元序列，格式为 ['<cls>'] + tokens_a + ['<sep>'] + tokens_b + ['<sep>']
        # segments: 片段索引列表，0表示属于第一个句子（tokens_a），1表示属于第二个句子（tokens_b）
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        
        # 将处理后的词元、片段索引和标签添加到结果列表中
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    
    # 返回生成的NSP数据
    return nsp_data_from_paragraph
```


```python
import random

def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    """
    为BERT的遮蔽语言模型（MLM）任务生成遮蔽后的输入词元和对应的标签。
    
    参数:
    tokens (list): 原始词元列表，例如 ["hello", "world", "."]。
    candidate_pred_positions (list): 候选的遮蔽位置列表，通常为非特殊词元（如<cls>、<sep>）的索引。
    num_mlm_preds (int): 需要遮蔽并预测的词元数量。
    vocab (Vocab): 词汇表，用于生成随机替换的词元。
    
    返回:
    tuple: 包含两个元素：
        - mlm_input_tokens (list): 遮蔽或替换后的词元列表，用于模型输入。
        - pred_positions_and_labels (list): 包含元组的列表，每个元组为 (遮蔽位置, 原始词元)，作为模型的训练标签。
    """
    # 复制原始词元列表，避免修改原始数据
    mlm_input_tokens = [token for token in tokens]
    # 初始化一个列表，用于存储被遮蔽的位置及其对应的原始词元（标签）
    pred_positions_and_labels = []
    
    # 打乱候选遮蔽位置的顺序，确保随机性
    random.shuffle(candidate_pred_positions)
    
    # 遍历每个候选遮蔽位置
    for mlm_pred_position in candidate_pred_positions:
        # 如果已经达到需要预测的词元数量，提前终止循环
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        
        masked_token = None
        # 按照BERT的MLM策略，以不同概率处理当前位置的词元：
        # 1. 80%的概率：将词元替换为"<mask>"
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 2. 10%的概率：保持词元不变
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 3. 10%的概率：用词汇表中的随机词元替换
            else:
                masked_token = random.choice(vocab.idx_to_token)
        
        # 更新遮蔽后的词元列表
        mlm_input_tokens[mlm_pred_position] = masked_token
        # 记录遮蔽位置和对应的原始词元（标签）
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    
    # 返回遮蔽后的输入词元和标签信息
    return mlm_input_tokens, pred_positions_and_labels
```


```python
def _get_mlm_data_from_tokens(tokens, vocab):
    """
    从词元列表中生成BERT掩蔽语言模型（MLM）任务的训练数据。
    
    参数:
    tokens (list): 原始词元列表，例如 ["hello", "world", "."]。
    vocab (Vocab): 词汇表，用于将词元转换为索引。
    
    返回:
    tuple: 包含三个元素：
        - mlm_input_ids (list): 遮蔽或替换后的词元索引列表，用于模型输入。
        - pred_positions (list): 被遮蔽词元的位置列表。
        - mlm_labels (list): 被遮蔽词元的原始索引标签列表。
    """
    # 初始化一个空列表，用于存储可被遮蔽的候选位置
    candidate_pred_positions = []
    # 遍历每个词元及其索引，筛选出非特殊词元的位置
    for i, token in enumerate(tokens):
        # 在MLM任务中，特殊词元（如'<cls>'和'<sep>'）不参与预测，因此跳过
        if token in ['<cls>', '<sep>']:
            continue
        # 将非特殊词元的位置添加到候选列表，供后续随机遮蔽
        candidate_pred_positions.append(i)
    
    # 计算需要遮蔽的词元数量：通常为词元总数的15%，但至少遮蔽1个
    # max(1, ...) 确保当序列过短时（如长度为0），仍有至少1个词元被遮蔽
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    
    # 调用_replace_mlm_tokens函数生成遮蔽后的词元列表和对应的标签信息
    # mlm_input_tokens: 遮蔽或替换后的词元列表（包含'<mask>'或随机词元）
    # pred_positions_and_labels: 包含元组的列表，每个元组为 (遮蔽位置, 原始词元)
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    
    # 按遮蔽位置从小到大排序，确保后续处理时位置顺序一致
    # 例如，若遮蔽位置为[3, 1]，排序后变为[1, 3]
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    
    # 从排序后的结果中提取遮蔽位置和对应的原始词元
    # pred_positions: 存储被遮蔽词元在原始tokens中的索引
    # mlm_pred_labels: 存储被遮蔽词元的原始值（字符串形式）
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    
    # 将遮蔽后的词元列表和标签词元转换为词汇表索引
    # vocab[mlm_input_tokens] 将词元列表（如["<mask>", "world"]）转换为对应的索引列表
    # vocab[mlm_pred_labels] 将原始词元标签（如["hello"]）转换为索引
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```


```python
def _get_mlm_data_from_tokens(tokens, vocab):
    """
    从词元列表中生成BERT掩蔽语言模型（MLM）任务的训练数据。
    
    参数:
    tokens (list): 原始词元列表，例如 ["hello", "world", "."]。
    vocab (Vocab): 词汇表，用于将词元转换为索引。
    
    返回:
    tuple: 包含三个元素：
        - mlm_input_ids (list): 遮蔽或替换后的词元索引列表，用于模型输入。
        - pred_positions (list): 被遮蔽词元的位置列表。
        - mlm_labels (list): 被遮蔽词元的原始索引标签列表。
    """
    # 初始化一个空列表，用于存储可被遮蔽的候选位置
    candidate_pred_positions = []
    # 遍历每个词元及其索引，筛选出非特殊词元的位置
    for i, token in enumerate(tokens):
        # 在MLM任务中，特殊词元（如'<cls>'和'<sep>'）不参与预测，因此跳过
        if token in ['<cls>', '<sep>']:
            continue
        # 将非特殊词元的位置添加到候选列表，供后续随机遮蔽
        candidate_pred_positions.append(i)
    
    # 计算需要遮蔽的词元数量：通常为词元总数的15%，但至少遮蔽1个
    # max(1, ...) 确保当序列过短时（如长度为0），仍有至少1个词元被遮蔽
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    
    # 调用_replace_mlm_tokens函数生成遮蔽后的词元列表和对应的标签信息
    # mlm_input_tokens: 遮蔽或替换后的词元列表（包含'<mask>'或随机词元）
    # pred_positions_and_labels: 包含元组的列表，每个元组为 (遮蔽位置, 原始词元)
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    
    # 按遮蔽位置从小到大排序，确保后续处理时位置顺序一致
    # 例如，若遮蔽位置为[3, 1]，排序后变为[1, 3]
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    
    # 从排序后的结果中提取遮蔽位置和对应的原始词元
    # pred_positions: 存储被遮蔽词元在原始tokens中的索引
    # mlm_pred_labels: 存储被遮蔽词元的原始值（字符串形式）
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    
    # 将遮蔽后的词元列表和标签词元转换为词汇表索引
    # vocab[mlm_input_tokens] 将词元列表（如["<mask>", "world"]）转换为对应的索引列表
    # vocab[mlm_pred_labels] 将原始词元标签（如["hello"]）转换为索引
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```


```python
import torch
import d2l

class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        """
        初始化WikiText数据集类，用于为BERT模型准备训练数据。

        参数:
        paragraphs (list): 包含多个段落的列表，每个段落是一个句子字符串列表。
        max_len (int): 输入序列的最大长度，用于填充和截断数据。
        """
        # 对每个段落中的句子进行分词处理，将句子字符串转换为词元列表
        # 输入的paragraphs[i]是句子字符串列表，处理后paragraphs[i]变为句子词元列表
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        # 将所有段落中的句子合并为一个大的句子列表
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        # 根据合并后的句子列表构建词汇表
        # min_freq=5表示只保留出现频率至少为5的词元
        # reserved_tokens指定了保留的特殊词元，这些词元在后续任务中有特殊用途
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # 初始化一个空列表，用于存储下一句子预测（NSP）任务的数据样本
        examples = []
        # 遍历每个段落，为每个段落生成NSP任务的数据样本
        # _get_nsp_data_from_paragraph函数会返回一个包含多个样本的列表
        # 每个样本是一个三元组 (tokens, segments, is_next)
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # 对每个样本进行处理，获取遮蔽语言模型（MLM）任务的数据
        # _get_mlm_data_from_tokens函数会返回一个三元组 (mlm_input_ids, pred_positions, mlm_labels)
        # 将MLM任务的数据与原有的NSP任务数据中的segments和is_next合并
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # 对所有样本进行填充处理，使输入数据的长度统一
        # _pad_bert_inputs函数会返回多个列表，分别存储填充后的不同类型的数据
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        """
        根据索引获取数据集中的一个样本。

        参数:
        idx (int): 样本的索引。

        返回:
        tuple: 包含多个张量的元组，分别是填充后的词元索引、片段索引、有效长度、
               遮蔽位置、遮蔽位置权重、MLM标签和NSP标签。
        """
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        """
        获取数据集的样本数量。

        返回:
        int: 数据集的样本数量，即填充后的词元索引列表的长度。
        """
        return len(self.all_token_ids)
```


```python
def load_data_wiki(batch_size, max_len):
    """加载WikiText-2数据集"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```


```python
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```


```python
len(vocab)
```

# 预训练BERT


```python
import torch
import os
from torch import nn
from d2l import torch as d2l
```


```python
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)
```


```python
import torch
import torch.nn as nn

# 假设 BERTEncoder、MaskLM、NextSentencePred 类已经定义
class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat

# 假设 vocab 已经定义
net = BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                num_layers=2, dropout=0.2, key_size=128, query_size=128,
                value_size=128, hid_in_features=128, mlm_in_features=128,
                nsp_in_features=128)

# 假设 d2l 工具包已经正确导入
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()
```


```python
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # 前向传播
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # 计算遮蔽语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # 计算下一句子预测任务的损失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```


```python
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    if len(devices) > 0:
        # 如果有可用的 GPU 设备
        net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    else:
        # 如果没有可用的 GPU 设备，使用 CPU
        print("No GPU devices found. Using CPU for training.")
        net = net.to('cpu')
    #net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # 遮蔽语言模型损失的和，下一句预测任务损失的和，句子对的数量，计数
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```


```python
train_bert(train_iter, net, loss, len(vocab), devices, 50)
```

    No GPU devices found. Using CPU for training.
    


```python
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```


```python
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# 词元：'<cls>','a','crane','is','flying','<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```


```python
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# 词元：'<cls>','a','crane','driver','came','<sep>','he','just',
# 'left','<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
```


```python

```
