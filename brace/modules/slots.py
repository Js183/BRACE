"""Per-modality slot projection."""
from torch import nn


class SlotProjector(nn.Module):
    """Projects a pre-extracted feature vector ([B, 1, D]) into num_slots slots."""

    def __init__(self, in_dim, hidden_dim, num_slots):
        super().__init__()
        self.num_slots = num_slots
        self.conv = nn.Conv1d(in_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv(x.transpose(1, 2))
        return x.expand(-1, -1, self.num_slots).transpose(1, 2).contiguous()
