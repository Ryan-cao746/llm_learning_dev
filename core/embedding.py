import torch

from core.dataloader import create_dataloader_v1
from core.tokenizer import get_raw_text


def create_bpe_embedding(raw_text:str):
    """创建适用于本项目里用的BPE分词器的嵌入层"""
    vocab_size = 50257  # 词表大小
    output_dim = 256    # 词嵌入向量维度
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)  # 创建词嵌入层

    max_length = 4
    data_loader = create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )

    data_iter = iter(data_loader)
    inputs, targets = next(data_iter)

    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    token_embeddings = token_embedding_layer(inputs)    # 将这些token ID 转换为 256 维的向量
    print(token_embeddings.shape)
    # 现在是8个batch，每个batch 4个token，每个token用256维向量表示，是torch.Size([8, 4, 256])

    # 绝对位置嵌入与序列中的特定位置直接相关。
    # 对于输入序列中的每个位置，都会将一个唯一的绝对位置嵌入向量添加到token的嵌入向量中，以传达其确切位置。
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    print(pos_embeddings.shape)
    # pos_embeddings 的输入通常是一个占位符向量torch.arange(context_length)，它包含一个从0到最大输入长度-1的数字序列。c
    # torch.arange(context_length)生成位置编号
    # 比如torch.arange(5)生成tensor([0, 1, 2, 3, 4])
    # 在这里，我们设置它与输入文本的最大长度相同。在实际应用中，输入文本可能会超过支持的上下文长度，此时我们需要对文本进行截断。

    # Transformer没有天然的序列顺序概念，所以需要把位置信息注入到token_embedding中。
    # 这两个向量维数不同，但最后一维相同，所以可以相加。pos_embedding没有batch维，表示所有样本共享一套位置向量
    # 相加时token_embedding的每个batch加上pos_embedding
    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)

    return input_embeddings

if __name__ == "__main__":
    text = get_raw_text()
    create_bpe_embedding(text)