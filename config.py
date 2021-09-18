import os
import torch

# 项目根目录路径
root_path = os.path.dirname(os.path.abspath(__file__))

# 初始化参数设置
PAD = 0                             # padding占位符的索引
UNK = 1                             # 未登录词标识符的索引
BATCH_SIZE = 128                    # 批次大小
EPOCHS = 20                         # 训练轮数
LAYERS = 6                          # transformer中encoder、decoder层数
H_NUM = 8                           # 多头注意力个数
D_MODEL = 256                       # 输入、输出词向量维数
D_FF = 1024                         # feed forward全连接层维数
DROPOUT = 0.1                       # dropout比例
MAX_LENGTH = 60                     # 语句最大长度

TRAIN_FILE = root_path + '/data/train.txt'  # 训练集
DEV_FILE = root_path + "/data/dev.txt"      # 验证集
TEST_FILE = root_path + "/data/test.txt"    # 测试集
SAVE_FILE = root_path + '/save/model.pt'    # 模型保存路径

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")