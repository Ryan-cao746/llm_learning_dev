"""
使用 PyTorch 的 nn.Linear 层来进一步改进 SelfAttention_v1 的实现。
当禁用偏置单元时，nn.Linear 层可以有效地执行矩阵乘法。
此外，使用 nn.Linear 替代手动实现的 nn.Parameter(torch.rand(...)) 的一个显著优势在于，
nn.Linear 具有优化的权重初始化方案，从而有助于实现更稳定和更高效的模型训练。
"""

import torch

class SelfAttentionV2(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # nn.Linear是线性变换，它只对输入的最后一维做线性变换
        # nn.Linear 就是 PyTorch 里的全连接层，用来把输入特征从 in_features 维映射到 out_features 维。
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        attn_scores = queries @ keys.T  # 这个scores矩阵是d_in行d_in列的
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec