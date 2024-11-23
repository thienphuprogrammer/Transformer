from numpy.ma.core import masked
from torch import nn

from models.layers.layer_norm import LayerNorm
from models.layers.position_wise_feed_forward import PositionWiseFeedForward
from models.layers.multi_head_attention import MultiHeadAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_out=drop_prob)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. Compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # 2. Add norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # Compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)

            # 4. Add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. position wise feed forward network
        _x = x
        x = self.ffn(x)

        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x