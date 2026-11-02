# 使用因果注意力机制来屏蔽后续词

import torch

class SelfAttentionV3(torch.nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        # linear具有优化的初始权重方案
        self.W_query = torch.nn.Linear(d_in, d_out)
        self.W_key = torch.nn.Linear(d_in, d_out)
        self.W_value = torch.nn.Linear(d_in, d_out)

    def forward(self, x):

        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1], dim=-1)
        print(attn_weights)

        # 应用因果注意力掩码，意思是用weight时加权不考虑当前向量后面的向量。
        # 比如你掩码后weight的n行第1-n列元素有值，其余为零，那么之后weight左乘value时加权就不带n之后的value的行向量了
        content_length = attn_weights.shape[0]  # 这个是d_in
        mask_simple = torch.tril(torch.ones(content_length, content_length))
        # torch.ones生成一个长和宽都为content_length的全1矩阵
        # torch.tril取这个矩阵的下三角部分，保留主对角线及其左下方的元素，其余是0.
        # 即得到一个下三角全1矩阵
        print(mask_simple)
        masked_simple = attn_weights * mask_simple  # 注意这里不是做矩阵乘法，是逐元素乘法，即两个张量对应位置相乘。这样才能起到掩码的作用
        print(masked_simple)

        # 重新归一化，让每行值为1
        row_sums = masked_simple.sum(dim=1, keepdim=True)   # 沿第二维(列)加起来，默认求和后将列的维度压掉。这里keepdim保留该求和维度为1
        masked_simple_norm = masked_simple / row_sums   # 两个矩阵每个位置做除法
        print(masked_simple_norm)

        context_vec = attn_weights @ values
        return context_vec

if __name__ == "__main__":
    attention = SelfAttentionV3(6, 6)
    attention.forward(torch.randn(6, 6))