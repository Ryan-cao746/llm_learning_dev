from tiktoken import Encoding
from torch.utils.data import Dataset
import torch

class GPTDatasetV1(Dataset):

    def __init__(self, txt:str, tokenizer:Encoding, max_length:int, stride):
        self.txt = txt
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride    # 步幅

        self.input_ids = []
        self.target_ids = []

        token_ids = self.tokenizer.encode(self.txt) # 将整个文本分词、

        for i in range(0, len(token_ids) - self.max_length, self.stride):
            # 使用滑动窗口按token将书籍分块
            input_chunk = token_ids[i:i + self.max_length]  # 输入窗口
            target_chunk = token_ids[i+1:i + self.max_length+1]     # 预测目标词的窗口

            # 从窗口创建一维张量
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """返回数据集的总行数"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """从数据集中返回指定行"""
        return self.input_ids[idx], self.target_ids[idx]