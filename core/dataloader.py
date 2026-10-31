import tiktoken
from torch.utils.data import DataLoader
from core.dataset import GPTDatasetV1
from core.tokenizer import get_raw_text


def create_dataloader_v1(
        txt:str,
        batch_size = 4,
        max_length = 256,
        stride = 128,   # 滑窗分词的步幅
        shuffle = True,
        drop_last = True,
        num_workers = 0,
):
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    # 创建GPTDatasetV1类
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # DataLoader类负责把 Dataset 中的单个样本组织成 batch，并提供自动批处理、打乱、多进程并行加载、自定义整理等功能
    data_loader = DataLoader(
        dataset,    # 这里数据集是自定义DataSet
        batch_size=batch_size,  # 每个batch(批次)的样本数
        shuffle=shuffle,    # 是否在每 epoch(轮) 打乱数据，默认 False。
        drop_last=drop_last,    # 是否丢弃最后一个不足 batch_size 的 batch，默认 False。
        num_workers=num_workers,    # 加载数据的子进程数，默认 0，表示在主进程加载。
    )

    return data_loader

if __name__ == "__main__":

    raw_text = get_raw_text()

    dataloader = create_dataloader_v1(
        raw_text,
        batch_size = 1,
        max_length = 8,
        stride = 2,
        shuffle = False,
    )

    data_iter = iter(dataloader)    # 从可迭代对象获取迭代器，迭代每个batch
    first_batch = next(data_iter)   # 第一批次
    print(first_batch)
    second_batch = next(data_iter)
    print(second_batch)     # 这里可以看到第二个批次的token_id向右移了一个位置。步幅shuffle决定了批次间移动的位置数。、
    # 根本上步幅是写在range()里的，就是range()的步幅参数
    # 同样可以看到每个张量的长度是max_length
    # 这些效果可以在自定义DataSet中实现