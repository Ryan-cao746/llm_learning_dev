# 为所有输入的token计算注意力权重的示例程序

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

# 计算所有输入对之间的点积
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, y_j in enumerate(inputs):
        attn_scores[i][j] = torch.dot(x_i, y_j)
print(attn_scores)
# 以上张量中的每个元素都表示每对输入之间的注意力得分

# 然而，for循环通常较慢，我们可以通过矩阵乘法实现相同的结果
# 每行结果是每行query与每个其他输入 token 的点积
attn_scores_v2 = inputs @ inputs.T
print(attn_scores_v2)
# 结果与之前一致

# 归一化
# 通过设置 dim=-1，我们指示 softmax 函数沿着 attn_scores 张量的最后一个维度进行归一化操作。
# 如果 attn_scores 是一个二维张量（例如，形状为 [行数, 列数]），则 dim=-1 将沿列方向进行归一化，
# 使得每一行的值（沿列方向求和）之和等于 1。
attn_weights = torch.softmax(attn_scores_v2, dim=-1)
print(attn_weights)

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

# 使用这些注意力权重通过矩阵乘法的方式来并行计算所有的上下文向量
# 实际上这意味着将注意力权重矩阵的行作为attention_v1的权重张量，也就是将每个输入向量与对应的注意力权重相乘后相加
# 也就是每行的结果就是inputs中每行向量作为query算出的上下文向量
all_attn_weights = attn_weights @ inputs
print(all_attn_weights)
