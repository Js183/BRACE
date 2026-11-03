"""Stage 1: quality-consistent anchor construction."""
import torch
from torch import nn

from ..constants import MODALITIES
from .slots import SlotProjector
from .transformer import JointTransformerBlock


class SlotFusionEncoder(nn.Module):
    """Slots per modality, quality-weighted state token and joint transformer.
    Produces the fused proxy and its raw sentiment score."""

    def __init__(self, cfg, input_dims, present):
        super().__init__()
        d, num_slots = cfg["hidden_dim"], cfg["num_slots"]
        self.register_buffer("present", torch.tensor(present, dtype=torch.bool), persistent=False)
        self.slot_projector = nn.ModuleDict({m: SlotProjector(input_dims[m], d, num_slots) for m in MODALITIES})
        self.slot_embeddings = nn.ParameterDict(
            {m: nn.Parameter(torch.randn(1, num_slots, d) * 0.02) for m in MODALITIES}
        )
        self.quality_proj = nn.Linear(d, d)
        self.quality_out = nn.Linear(d, 1, bias=False)
        self.type_embeddings = nn.Parameter(torch.randn(len(MODALITIES) + 1, d) * 0.02)
        self.joint_transformer = nn.ModuleList(
            [JointTransformerBlock(d, cfg["heads"], cfg["ffn_dim"], cfg["dropout"]) for _ in range(cfg["joint_depth"])]
        )
        self.raw_score_head = nn.Linear(d, 1)

    @property
    def width(self):
        return self.raw_score_head.in_features

    def forward(self, inputs):
        slots = [self.slot_projector[m](x) + self.slot_embeddings[m] for m, x in zip(MODALITIES, inputs)]
        summaries = [s.mean(dim=1) for s in slots]
        quality = torch.cat([self.quality_out(torch.tanh(self.quality_proj(s))) for s in summaries], dim=1)
        batch = quality.size(0)
        present = self.present.view(1, -1).expand(batch, -1).to(dtype=quality.dtype)
        alpha = torch.softmax(quality.masked_fill(present < 0.5, -1e4), dim=1) * present
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp(min=1e-6)
        state = 0.0
        for i, s in enumerate(summaries):
            state = state + alpha[:, i:i + 1] * s

        tokens = [state.unsqueeze(1) + self.type_embeddings[0].view(1, 1, -1)]
        padding = [torch.zeros(batch, 1, device=quality.device, dtype=torch.bool)]
        num_slots = slots[0].size(1)
        for i, s in enumerate(slots):
            tokens.append((s + self.type_embeddings[i + 1].view(1, 1, -1)) * present[:, i:i + 1].unsqueeze(-1))
            padding.append((present[:, i:i + 1] < 0.5).expand(-1, num_slots))
        x = torch.cat(tokens, dim=1)
        key_padding_mask = torch.cat(padding, dim=1)
        for block in self.joint_transformer:
            x = block(x, key_padding_mask)
        proxy = x[:, 0]
        return proxy, self.raw_score_head(proxy)
