# 实现一个简洁的自注意力机制类
import torch
import torch.nn as nn

class SelfAttentionV1(nn.Module):
    """nn.Module 是 PyTorch 模型的基础组件，提供了创建和管理模型层所需的必要功能。"""

    def __init__(self, d_in, d_out):
        """__init__ 方法初始化了用于计算查询（query）、键（key）和值（value）的可训练权重矩阵（W_query、W_key 和 W_value），
        每个矩阵都将输入维度 d_in 转换为输出维度 d_out。
        """
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)
        self.W_key = nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)
        self.W_value = nn.Parameter(torch.randn(d_in, d_out), requires_grad=True)

    def forward(self, x):
        """
        前向传播过程在 forward 方法中实现，我们通过将查询（query）和键（key）相乘来计算注意力得分（attn_scores），
        并使用 softmax 对这些得分进行归一化。
        最后，我们使用这些归一化的注意力得分对值（value）加权，生成上下文向量。
        :param x:
        :return:
        """
        # 根据input计算三个向量
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        # 计算注意力得分
        # QK^T的第(i,j)个元素正好是第i个query向量和第j个key向量的点积
        attn_scores = queries @ keys.T
        # 归一化
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec
