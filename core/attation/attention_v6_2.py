# 使用因果注意力机制来屏蔽后续词的第二种做法
# 用softmax会将负无穷视作零的特性，直接做掩码。这种方法就不用先softmax一遍再做掩码再归一化了

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

        content_length = attn_scores.shape[0]
        mask = torch.triu(torch.ones(content_length, content_length), diagonal=-1)  # 取上三角，不取主对角线
        masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
        # 把 attn_scores 中 mask 为 True 的位置替换成 负无穷 -inf，返回一个新的张量 masked，不会原地修改 attn_scores。
        print(masked)

        attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
        print(attn_weights)

        context_vec = attn_weights @ values
        return context_vec

if __name__ == "__main__":
    attention = SelfAttentionV3(6, 6)
    attention.forward(torch.randn(6, 6))