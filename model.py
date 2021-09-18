import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import DEVICE

def clones(module, N):
    """
    深拷贝克隆，克隆的单元之间参数不共享
    """    
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embeddings(nn.Module):
    """
    Transformer Encoder的词向量编码层
    """
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        # 返回x的词向量, 乘以math.sqrt(embed_dim)
        return self.embeddings(x) * math.sqrt(self.embed_dim)

class PositionalEncoding(nn.Module):
    """
    Transformer Encoder的位置编码层
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        # 位置编码矩阵，维度[max_len, embedding_dim]

        pe = torch.zeros(max_len, d_model, device=DEVICE)
        # 单词位置
        position = torch.arange(0.0, max_len, device=DEVICE)
        position.unsqueeze_(1)

        # 使用exp和log实现幂运算
        div_term = torch.exp(torch.arange(0.0, d_model, 2, device=DEVICE) * 
            (- math.log(1e4) / d_model))
        div_term.unsqueeze_(0)

        # 计算单词位置沿词向量维度的纹理值
        pe[:, 0 : : 2] = torch.sin(torch.mm(position, div_term))
        pe[:, 1 : : 2] = torch.cos(torch.mm(position, div_term))

        # 增加批次维度，[1, max_len, embedding_dim]
        pe.unsqueeze_(0)

        # 将位置编码矩阵注册为buffer(不参加训练)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将一个批次中语句所有词向量与位置编码相加
        # 注意，位置编码不参与训练，因此detach
        x += self.pe[:, : x.size(1), :]
        x = x.detach()
        return self.dropout(x)

def attention(query, key, value, mask=None, dropout=None):
    """
    Scaled Dot-Product Attention
    """   
    # q、k、v向量维度为d_k
    d_k = query.size(-1) 

    # 矩阵乘法实现q、k点积注意力，除以sqrt(d_k)归一化
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 注意力掩码机制
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)  

    # 注意力矩阵softmax归一化
    p_attn = F.softmax(scores, dim=-1)  

    if dropout is not None:
        p_attn = dropout(p_attn)

    # 注意力对v加权
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    """
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        """
        `h`：注意力头的数量
        `d_model`：词向量维数
        """
        # 确保词向量维数能整除注意力头数
        assert d_model % h == 0
        # q、k、v向量维数
        self.d_k = d_model // h
        # 头的数量
        self.h = h
        # WQ、WK、WV矩阵及多头注意力拼接变换矩阵WO
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        # 批次样本个数
        nbatches = query.size(0)

        # WQ、WK、WV分别对词向量线性变换，并将结果拆成h块
        query, key, value = [line(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                            for line, x in zip(self.linears, (query, key, value))]
        # 注意力加权
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 多头注意力加权拼接
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        # 对多头注意力加权拼接结果线性变换
        return self.linears[-1](x)

class LayerNorm(nn.Module):
    """
    层归一化
    """
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # α、β分别初始化为1、0
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 平滑项
        self.eps = eps

    def forward(self, x):
        # 沿词向量方向计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        # 沿词向量和语句序列方向计算均值和方差
        # mean = x.mean(dim=[-2, -1], keepdim=True)
        # std = x.std(dim=[-2, -1], keepdim=True)

        # 归一化
        x = (x - mean) / torch.sqrt(std ** 2 + self.eps) 
        return self.a_2 * x + self.b_2


class SublayerConnection(nn.Module):
    """
    通过层归一化和残差连接，连接Multi-Head Attention和Feed Forward
    """
    def __init__(self, size, dropout):
        super().__init__()           
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 层归一化
        x_ = self.norm(x)
        x_ = sublayer(x_)
        x_ = self.dropout(x_)
        # 残差连接
        return x + x_

class PointwiseFeedForward(nn.Module):
    """
    FeedForwad前馈神经网络
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout= nn.Dropout(dropout)

    def forward(self, x):
        # 先经过一个全连接层
        x = self.w_1(x)

        # 再通过relu激活函数进行非线性变换
        x = F.relu(x)
        x = self.dropout(x)
        # 先经过一个全连接层
        x = self.w_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # SublayerConnection作用连接multi和ffn
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

        # d_model
        self.size = size

    def forward(self, x, mask):
        # 将embedding层进行Multi head Attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        # attn的结果直接作为下一层输入
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        """
        layer = EncoderLayer
        """
        super(Encoder, self).__init__()
        # 复制N个编码器基本单元
        self.layers = clones(layer, N)
        # 层归一化
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        """
        循环编码器基本单元N次
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        # d_model
        self.size = size
        #自注意力机制
        self.self_attn = self_attn
        # 上下文注意力机制
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # memory为编码器输出隐表示
        m = memory

        # 自注意力机制，q、k、v均来自解码器隐表示
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        # 上下文注意力机制：q为来自解码器隐表示，而k、v为编码器隐表示
        x = self.sublayer[1](x, lambda x: self.self_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        循环解码器基本单元N次
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Generator(nn.Module):
    """
    解码器输出经线性变换和softmax函数映射为下一时刻预测单词的概率分布
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        # decode后的结果，先进入一个全连接层变为词典大小的向量
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 然后再进行log_softmax操作(在softmax结果上再做多一次log运算)
        return F.log_softmax(self.proj(x), dim=-1)

class Transformer(nn.Module):
    def __init__(self, 
                    encoder,     # 编码器
                    decoder,     # 解码器
                    src_embed,   # 编码器embedding
                    tgt_embed,   # 解码器embedding
                    generator):  # 解码器输出经softmax预测下一个词
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # encoder的结果作为decoder的memory参数传入，进行decode
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

def make_model(src_vocab, 
                tgt_vocab, 
                N=6, 
                d_model=512, 
                d_ff=2048, 
                h = 8, 
                dropout=0.1):                
    cp = copy.deepcopy

    # 实例化Self Attention对象
    attn = MultiHeadAttention(h, d_model).to(DEVICE)

    # 实例化feedforword对象
    ff = PointwiseFeedForward(d_model, d_ff).to(DEVICE)

    # 实例化位置编码对象
    position = PositionalEncoding(d_model, dropout).to(DEVICE)

    # 实例化Transformer对象
    model = Transformer(Encoder(EncoderLayer(d_model, cp(attn), cp(ff), dropout). \
                            to(DEVICE), N).to(DEVICE),
                        Decoder(DecoderLayer(d_model, cp(attn), cp(attn), cp(ff), dropout). \
                            to(DEVICE), N).to(DEVICE),
                        nn.Sequential(Embeddings(d_model, src_vocab).to(DEVICE), cp(position)),
                        nn.Sequential(Embeddings(d_model, tgt_vocab).to(DEVICE), cp(position)),
                        Generator(d_model, tgt_vocab)).to(DEVICE)

    # xavier_uniform_初始化参数
    for param in model.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

    print("****** Transform encoder layers ******\n")  
    print(model.encoder)

    print("****** Transform decoder layers ******\n")
    print(model.decoder)

    return model.to(DEVICE)