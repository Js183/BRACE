"""Stage 2: global state-label memory."""
import torch
import torch.nn.functional as F
from torch import nn

from ..constants import LABEL_MAX, LABEL_MIN


class MemoryBank(nn.Module):
    """FIFO bank of (key, proxy, label) plus positive/negative prototypes.
    Keys are produced by the editor's key projection, which the bank shares."""

    def __init__(self, key_proj, value_dim, cfg):
        super().__init__()
        self.key_proj = key_proj
        self.size = cfg["bank_size"]
        self.prototype_momentum = cfg["prototype_momentum"]
        self.prototype_quantile = cfg["prototype_quantile"]
        self.register_buffer("keys", torch.zeros(self.size, key_proj.out_features, dtype=torch.float16))
        self.register_buffer("values", torch.zeros(self.size, value_dim, dtype=torch.float16))
        self.register_buffer("labels", torch.zeros(self.size, dtype=torch.float16))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("proto_pos", torch.zeros(value_dim))
        self.register_buffer("proto_neg", torch.zeros(value_dim))
        self.register_buffer("proto_pos_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("proto_neg_count", torch.zeros(1, dtype=torch.long))
        self.register_buffer("ready", torch.zeros((), dtype=torch.bool))

    def query(self, proxy):
        return F.normalize(self.key_proj(proxy), dim=-1)

    def fallback_proto(self, positive, like):
        proto = self.proto_pos if positive else self.proto_neg
        count = int((self.proto_pos_count if positive else self.proto_neg_count).item())
        if count == 0:
            return like
        return proto.to(device=like.device, dtype=like.dtype).unsqueeze(0).expand_as(like)

    @torch.no_grad()
    def rebuild(self, encoder, loader, device):
        """Fill the bank from the training set with the given encoder; editing is active from then on."""
        encoder.eval()
        for buf in (self.keys, self.values, self.labels, self.ptr, self.count,
                    self.proto_pos, self.proto_neg, self.proto_pos_count, self.proto_neg_count):
            buf.zero_()
        for *inputs, labels in loader:
            proxy, _ = encoder([x.to(device) for x in inputs])
            self.write(proxy, labels.to(device))
        self.ready.fill_(True)
        encoder.train()

    @torch.no_grad()
    def write(self, proxy, labels):
        values = proxy.detach()
        keys = self.query(values).to(dtype=torch.float16)
        labels = labels.view(-1)
        n = keys.size(0)
        if n >= self.size:
            keys, values_h, labels_h = keys[-self.size:], values[-self.size:], labels[-self.size:]
            n = self.size
        else:
            values_h, labels_h = values, labels
        values_h = values_h.to(dtype=torch.float16)
        labels_h = labels_h.detach().to(dtype=torch.float16)
        ptr = int(self.ptr.item())
        first = min(n, self.size - ptr)
        self.keys[ptr:ptr + first] = keys[:first]
        self.values[ptr:ptr + first] = values_h[:first]
        self.labels[ptr:ptr + first] = labels_h[:first]
        if first < n:
            self.keys[:n - first] = keys[first:]
            self.values[:n - first] = values_h[first:]
            self.labels[:n - first] = labels_h[first:]
        self.ptr[0] = (ptr + n) % self.size
        self.count[0] = min(self.size, int(self.count.item()) + n)

        span = LABEL_MAX - LABEL_MIN
        for positive, mask in (
            (True, labels >= LABEL_MIN + (1.0 - self.prototype_quantile) * span),
            (False, labels <= LABEL_MIN + self.prototype_quantile * span),
        ):
            if not mask.any():
                continue
            mean = values[mask].mean(dim=0)
            proto = self.proto_pos if positive else self.proto_neg
            count_buf = self.proto_pos_count if positive else self.proto_neg_count
            count = int(count_buf.item())
            if count == 0:
                proto.copy_(mean)
            else:
                proto.mul_(self.prototype_momentum).add_(mean * (1.0 - self.prototype_momentum))
            count_buf[0] = count + int(mask.sum().item())
