"""Masked pre-norm transformer block over the state token and the observed slots."""
from torch import nn


class JointTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, heads, ff_dim, dropout):
        super().__init__()
        self.attn_ln = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.ffn_ln = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask):
        h = self.attn_ln(x)
        x = x + self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return x + self.ffn(self.ffn_ln(x))
