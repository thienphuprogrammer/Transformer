import math
from torch import nn


class ScaleDotProductAttention(nn.Module):
    def __init(self):
        super(ScaleDotProductAttention, self).__init()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # q: (batch_size, num_heads, seq_len, d_k)
        # k: (batch_size, num_heads, seq_len, d_k)
        # v: (batch_size, num_heads, seq_len, d_v)
        # mask: (batch_size, num_heads, seq_len, seq_len)

        # input is 4 dimensional tensor
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # (batch_size, num_heads, d_k, seq_len)
        score = (q @ k_t) / math.sqrt(d_tensor)

        # 2. Apply mask
        if mask is not None:
            score = score.masked_fill(mask == 0, -e)

        # 3. Apply softmax
        score = self.softmax(score)
        v = score @ v

        return v, score
