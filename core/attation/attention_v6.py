
import torch

class SelfAttentionV3(torch.nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

        self.W_query = torch.nn.Linear(d_in, d_out)
        self.W_key = torch.nn.Linear(d_in, d_out)
        self.W_value = torch.nn.Linear(d_in, d_out)

    def forward(self, x):

        keys = self.w_key(x)
        values = self.w_value(x)
        queries = self.W_query(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1], dim=-1)

        context_vec = attn_weights @ values
        return context_vec