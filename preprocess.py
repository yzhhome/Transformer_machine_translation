import numpy as np
import torch
from collections import Counter
from langconv import Converter
from nltk import word_tokenize
import config


def seq_padding(X, padding=config.PAD):
    """
    按批次（batch）对数据填充、长度对齐
    """
    # 计算该批次各条样本语句长度
    L = [len(x) for x in X]
    # 获取该批次样本中语句长度最大值
    ML = max(L)
    # 遍历该批次样本，如果语句长度小于最大长度，则用padding填充
    return np.array([np.concatenate([x, [padding] * (ML - len(x))]) \
        if len(x) < ML else x for x in X])

def cht_to_chs(sent):
    """
    繁体转换为简体
    """
    sent = Converter("zh-hans").convert(sent)
    sent.encode("utf-8")
    return sent

class PrepareData(object):
    def __init__(self, train_file, dev_file):
        # 读取数据、分词
        self.train_en, self.train_cn = self.load_data(train_file)
        self.dev_en, self.dev_cn = self.load_data(dev_file)

        # 构建词表
        self.en_word_dict, self.en_total_words, self.en_index_dict = \
            self.build_dict(self.train_en)
        self.cn_word_dict, self.cn_total_words, self.cn_index_dict = \
            self.build_dict(self.train_cn)

         # 单词映射为索引
        self.train_en, self.train_cn = self.word2id(self.train_en, 
            self.train_cn, self.en_word_dict, self.cn_word_dict)
        self.dev_en, self.dev_cn = self.word2id(self.dev_en, 
            self.dev_cn, self.en_word_dict, self.cn_word_dict)

        # 划分批次、填充、掩码
        self.train_data = self.split_batch(self.train_en, self.train_cn, config.BATCH_SIZE)
        self.dev_data = self.split_batch(self.dev_en, self.dev_cn, config.BATCH_SIZE)           

    def load_data(self, path):
        """
        读取英文、中文数据
        对每条样本分词并构建包含起始符和终止符的单词列表
        形式如：en = [['BOS', 'i', 'love', 'you', 'EOS'], ['BOS', 'me', 'too', 'EOS'], ...]
                cn = [['BOS', '我', '爱', '你', 'EOS'], ['BOS', '我', '也', '是', 'EOS'], ...]
        """
        en = []
        cn = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                # 英文和中文按\t分隔
                sent_en, sent_cn = line.strip().split('\t') 

                sent_en = sent_en.lower()
                sent_cn = cht_to_chs(sent_cn)
                
                sent_en = ['BOS'] + word_tokenize(sent_en) + ['EOS']

                # 中文直接按字切分，不用分词
                sent_cn = ['BOS'] + [char for char in sent_cn] + ['EOS']

                en.append(sent_en)
                cn.append(sent_cn)
        return en, cn

    def build_dict(self, sentences, max_word=5e4):
        """
        构造分词后的列表数据
        构建单词-索引映射（key为单词，value为id值）
        """        
        # 统计数据集中单词词频
        word_count = Counter([word for sent in sentences for word in sent])

        # 按词频保留前max_words个单词构建词典
        ls = word_count.most_common(int(max_word))
        total_words = len(ls) + 2
                
        word_dict = {w[0] : index for index, w in enumerate(ls)}
        word_dict['PAD'] = config.PAD
        word_dict['UNK'] = config.UNK

        index_dict = {v: k for k, v in  word_dict.items()}

        return word_dict, total_words, index_dict

    def word2id(self, en, cn, en_dict, cn_dict, sort=True):
        """
        将英文、中文单词列表转为单词索引列表
        `sort=True`表示以英文语句长度排序，以便按批次填充时，同批次语句填充尽量少
        """
        out_en_ids = [[en_dict.get(word, config.UNK) for word in sent] for sent in en]
        out_cn_ids = [[cn_dict.get(word, config.UNK) for word in sent] for sent in cn]

        def len_argsort(seq):
            """
            传入一系列语句数据(分好词的列表形式)，
            按照语句长度排序后，返回排序后原来各语句在数据中的索引下标
            """
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        if sort:
            sort_index = len_argsort(out_en_ids)

            # 按排序后所在原数据中的索引获取ids
            out_en_ids = [out_en_ids[idx] for idx in sort_index]
            out_cn_ids = [out_cn_ids[idx] for idx in sort_index]

        return out_en_ids, out_cn_ids

    def split_batch(self, en, cn, batch_size, shuffle=True):
        """
        划分批次
        shuffle=True 表示对所有批次的顺序随机打乱
        """
         # 每隔batch_size生成一个数字作为每个batch数据的起始索引
        idx_list = np.arange(0, len(en), batch_size)

        if shuffle:
            np.random.shuffle(idx_list)

        batch_indexs = []
        # 生成每个batch中数据的索引列表
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(en))))

        batches = []
        for bat_index in batch_indexs:
            # 根据索引获取每个batch中的数据
            batch_en = [en[index] for index in bat_index]
            batch_cn = [cn[index] for index in bat_index]

            batch_en = seq_padding(batch_en)
            batch_cn = seq_padding(batch_cn)

            # 将当前批次添加到批次列表
            # Batch类用于实现注意力掩码
            batches.append(Batch(batch_en, batch_cn))

        return batches


def subsequent_mask(size):
    "Mask out subsequent positions."
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0

class Batch(object):
    """
    批次类
        1. 输入序列（源）
        2. 输出序列（目标）
        3. 构造掩码
    """
    def __init__(self, src, trg=None, pad=config.PAD):

        # 将输入、输出单词id表示的数据规范成整数类型
        src = torch.from_numpy(src).to(config.DEVICE).long()
        trg = torch.from_numpy(trg).to(config.DEVICE).long()
        self.src = src

        # 对于当前输入的语句非空部分进行判断，bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)

        # 如果输出目标不为空，则需要对解码器使用的目标语句进行掩码
        if trg is not None:
            # 解码器使用的目标输入部分
            self.trg = trg[:, : -1]

            # 解码器训练时应预测输出的目标结果
            self.trg_y = trg[:, 1 :]

            # 将目标输入部分进行注意力掩码
            self.trg_mask = self.make_std_mask(self.trg, pad)

            # 将应输出的目标结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()
    
    # 掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)      
        
        return tgt_mask