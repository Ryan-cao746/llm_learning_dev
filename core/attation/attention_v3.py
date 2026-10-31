# 实现带有可训练权重的自注意力机制。第一个简单示例，逐步计算。

import torch

# 已被转化为token嵌入的句子
inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your     (x^1)
        [0.55, 0.87, 0.66],  # journey  (x^2)
        [0.57, 0.85, 0.64],  # starts   (x^3)
        [0.22, 0.58, 0.33],  # with     (x^4)
        [0.77, 0.25, 0.10],  # one      (x^5)
        [0.05, 0.80, 0.55]   # step     (x^6)
    ]
)

x_2 = inputs[1] # 取第二个输入元素
d_in = inputs.shape[1]  # 输入维度是inputs的第二个维度的shape，即3
d_out = 2   # 输出维度，2

# 初始化三个权重矩阵
# torch.randn(d_in, d_out)生成一个形状为 (d_in, d_out) 的矩阵，元素服从标准正态分布，均值 0，方差 1
# torch.nn.Parameter把普通张量包装成“模型参数”。这样它会被注册到模型里，调用 model.parameters() 时能拿到它，优化器也能更新它。
# requires_grad=True表示这个参数需要计算梯度，训练时通过反向传播更新。
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)
W_key = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)
W_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)

# 计算query key value向量
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

# 获得所有元素的key，和value
# 尽管我们当前的目标仅仅是计算一个上下文向量 z(2)，但仍然需要获取所有输入元素的 key 和 value 向量，
# 因为它们将参与与查询向量 q(2) 一起计算注意力权重的过程
keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# 计算注意力得分
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# 应用到第2元素的所有注意力得分计算
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

# 通过将注意力得分除以keys嵌入维度的平方根来进行缩放，然后归一化
# 缩放是为了将点积结果缩放到适当的范围，避免softmax函数进入梯度平缓区
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

# 现在我们通过值向量的加权和来计算上下文向量
# 注意力权重作为加权因子，用于衡量每个值向量的重要性
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)