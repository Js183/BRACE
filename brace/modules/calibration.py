"""Stage 3: bipolar reference-aware calibration."""
import torch
from torch import nn

from .memory import MemoryBank


class RetrievalEditor(nn.Module):
    """Bipolar reference-aware calibration (Stage 3). Built from the encoder: the
    refined state is scored by the encoder's own prediction head g, widths come from
    the encoder, and the memory bank is created around this editor's key projection."""

    def __init__(self, encoder, cfg):
        super().__init__()
        d = encoder.width
        self.num_candidates = cfg["num_candidates"]
        self.neighbor_margin = cfg["neighbor_margin"]
        self.neighbor_temp = cfg["neighbor_temp"]
        self.direction_margin = cfg["direction_margin"]
        self.score_head = encoder.raw_score_head
        self.key_proj = nn.Linear(d, cfg["key_dim"])
        self.edit_gate = nn.Linear(4 * d, d)
        self.edit_delta = nn.Linear(4 * d, d)
        self.bank = MemoryBank(self.key_proj, d, cfg)

    def retrieve(self, proxy, raw_score):
        """Higher- and lower-label centroids c+ / c- around the current prediction (Eq. 8)."""
        bank = self.bank
        fallback_plus = bank.fallback_proto(True, proxy)
        fallback_minus = bank.fallback_proto(False, proxy)
        count = int(bank.count.item())
        if count == 0:
            return fallback_plus, fallback_minus

        keys = bank.keys[:count].to(dtype=proxy.dtype)
        values = bank.values[:count].to(dtype=proxy.dtype)
        labels = bank.labels[:count].to(dtype=proxy.dtype)
        top_sim, top_idx = torch.topk(bank.query(proxy) @ keys.t(), k=min(self.num_candidates, count), dim=-1)
        batch, k = top_idx.shape
        cand_labels = labels.index_select(0, top_idx.reshape(-1)).view(batch, k)
        cand_values = values.index_select(0, top_idx.reshape(-1)).view(batch, k, -1)
        raw_col = raw_score.view(batch, 1)
        logits = self.neighbor_temp * top_sim

        def centroid(mask):
            w = torch.softmax(logits.masked_fill(~mask, -1e4), dim=1) * mask.to(dtype=logits.dtype)
            w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=1e-12)
            return torch.sum(w.unsqueeze(-1) * cand_values, dim=1)

        upper = cand_labels >= raw_col + self.neighbor_margin
        lower = cand_labels <= raw_col - self.neighbor_margin
        return (
            torch.where(upper.any(dim=1, keepdim=True), centroid(upper), fallback_plus),
            torch.where(lower.any(dim=1, keepdim=True), centroid(lower), fallback_minus),
        )

    def forward(self, proxy, raw_score):
        """Gated signed update (Eq. 9): z = [s; c+; c-; c+ - c-], s' = s + sigma(phi_b(z)) * phi_d(z), r' = g(s')."""
        c_plus, c_minus = self.retrieve(proxy, raw_score)
        z = torch.cat([proxy, c_plus, c_minus, c_plus - c_minus], dim=-1)
        refined = proxy + torch.sigmoid(self.edit_gate(z)) * self.edit_delta(z)
        return self.score_head(refined)

    def direction_loss(self, raw, edited, labels):
        """Push the edit towards the label, balanced between up- and down-moves."""
        d = torch.sign(labels - raw)
        hinge = torch.relu(self.direction_margin - d * (edited - raw))
        pos, neg = (d > 0).float(), (d < 0).float()
        has_pos, has_neg = pos.sum() > 0, neg.sum() > 0
        pos_loss = torch.sum(hinge * pos) / torch.clamp(pos.sum(), min=1.0)
        neg_loss = torch.sum(hinge * neg) / torch.clamp(neg.sum(), min=1.0)
        if has_pos and has_neg:
            return 0.5 * (pos_loss + neg_loss)
        if has_pos:
            return pos_loss
        if has_neg:
            return neg_loss
        return hinge.new_tensor(0.0)
