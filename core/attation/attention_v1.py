# 简单的注意力实例程序

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

# 第二个输入token被用作查询向量
query = inputs[1]

# torch.empty 用来创建一个指定形状的张量，但只分配内存，不初始化里面的值。
# inputs.shape[0]表示取 inputs 这个张量或数组的第 0 维的大小，也就是最外层维度的长度。这里是token数
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    # 我们通过计算查询 x(2) 与每个其他输入 token 的点积来确定查询 token 与每个输入 token 之间的中间注意力得分
    attn_scores_2[i] = torch.dot(x_i, query)

print(attn_scores_2)

# 归一化注意力权重
# dim参数表示沿着哪一个维度做归一化，得该维度上的所有元素经过 softmax 后加起来等于 1。这里是对张量的零维归一化
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# 上下文向量context_vec_2是所有输入向量的加权和.它的计算方法为将每个输入向量与对应的注意力权重相乘后相加
# torch.zeros 用来创建一个指定形状、全部元素为 0 的张量。
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i

print(context_vec_2)