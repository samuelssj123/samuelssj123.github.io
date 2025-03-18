# 文本预处理
1. 将文本作为字符串加载到内存中。
1. 将字符串拆分为词元（如单词和字符）。
1. 建立一个词表，将拆分的词元映射到数字索引。
1. 将文本转换为数字索引序列，方便模型操作。


```python
import collections
import re
from d2l import torch as d2l
```


```python
# 将数据集读取到由多条文本行组成的列表中
```


```python
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  
    """将时间机器数据集加载到文本行的列表中"""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()

# 词元化
def tokenize(lines, token='word'):  
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

    Downloading ../data\timemachine.txt from http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt...
    ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
    []
    []
    []
    []
    ['i']
    []
    []
    ['the', 'time', 'traveller', 'for', 'so', 'it', 'will', 'be', 'convenient', 'to', 'speak', 'of', 'him']
    ['was', 'expounding', 'a', 'recondite', 'matter', 'to', 'us', 'his', 'grey', 'eyes', 'shone', 'and']
    ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
    


```python
# 定义词汇表类，用于将单词（token）映射到索引（idx）或反向映射
class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """
        初始化词汇表对象

        参数说明:
        - tokens: 输入的文本数据，可以是单词列表或嵌套列表（如分句后的单词列表），默认为空列表
        - min_freq: 最小词频阈值，出现次数低于该值的单词不会被加入词汇表，默认保留所有词
        - reserved_tokens: 保留的标记列表（如'<unk>', '<pad>'等），这些标记会强制加入词汇表，默认为空列表

        功能:
        1. 统计所有token的频率
        2. 按词频排序并构建索引与token的映射关系
        3. 过滤低频词，保留高频词和保留标记
        """
        # 处理参数默认值
        if tokens is None:
            tokens = []  # 确保tokens不为None
        if reserved_tokens is None:
            reserved_tokens = []  # 确保保留标记列表不为None

        # 统计词频（调用count_corpus函数）
        counter = count_corpus(tokens)
        
        # 按词频降序排序，得到元组列表（token, freq）
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        
        # 初始化索引到token的映射列表：
        # 1. 第0位固定为未知词标记'<unk>'
        # 2. 随后添加用户定义的保留标记
        self.idx_to_token = ['<unk>'] + reserved_tokens
        
        # 初始化token到索引的字典映射（反向映射）
        # 使用推导式生成字典，初始包含'<unk>'和保留标记的映射
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        
        # 遍历排序后的词频列表，构建完整词汇表
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break  # 遇到低频词立即停止（因已排序，后续词频必定更低）
            if token not in self.token_to_idx:  # 防止重复添加保留标记
                self.idx_to_token.append(token)  # 添加token到索引映射列表
                self.token_to_idx[token] = len(self.idx_to_token) - 1  # 更新反向映射字典

    def __len__(self):
        """返回词汇表的总长度（含保留标记）"""
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """
        将单词（token）转换为索引（idx）
        
        参数说明:
        - tokens: 单个单词或单词列表/元组
        
        返回:
        - 单个索引或索引列表
        """
        # 处理单个token的情况
        if not isinstance(tokens, (list, tuple)):
            # 使用get方法获取索引，若不存在则返回unk的索引（0）
            return self.token_to_idx.get(tokens, self.unk())
        # 递归处理列表/元组类型的多个token
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """
        将索引（indices）转换回单词（token）
        
        参数说明:
        - indices: 单个索引或索引列表/元组
        
        返回:
        - 单个token或token列表
        """
        # 处理单个索引的情况
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        # 处理多个索引的情况
        return [self.idx_to_token[index] for index in indices]

    def unk(self):
        """返回未知词（<unk>）的索引（固定为0）"""
        return 0

def count_corpus(tokens):
    """
    统计词频的辅助函数
    
    参数说明:
    - tokens: 输入数据，可以是1D列表（单个句子）或2D列表（多个句子组成的语料库）
    
    返回:
    - Counter对象，包含每个token的频率统计
    
    实现原理:
    1. 如果检测到输入是嵌套列表（如[[token1, token2], [token3]]），则展平为一维列表
    2. 使用collections.Counter进行高效词频统计
    """
    # 展平嵌套列表：如果tokens非空且第一个元素是列表，则展开所有子列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    # 返回Counter对象（自动统计词频）
    return collections.Counter(tokens)
```


```python
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
```

    [('<unk>', 0), ('the', 1), ('i', 2), ('and', 3), ('of', 4), ('a', 5), ('to', 6), ('was', 7), ('in', 8), ('that', 9)]
    


```python
for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])
```

    文本: ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
    索引: [1, 19, 50, 40, 2183, 2184, 400]
    文本: ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
    索引: [2186, 3, 25, 1044, 362, 113, 7, 1421, 3, 1045, 1]
    


```python
def load_corpus_time_machine(max_tokens=-1): 
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```




    (170580, 28)




```python

```
