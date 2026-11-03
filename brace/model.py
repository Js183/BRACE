"""BRACE: slot encoding, quality-weighted joint transformer, and
memory-bank retrieval editing of the fused proxy.

The three parts are built on top of each other and share instances:
the editor is constructed from the encoder and takes all of its widths from
it, and the memory bank is constructed by the editor around the editor's own
key projection, so keys are written and queried through one shared layer.
"""
import torch
from torch import nn

from .constants import LAMBDA_DIR
from .modules import RetrievalEditor, SlotFusionEncoder


class BRACE(nn.Module):
    def __init__(self, cfg, input_dims, present):
        super().__init__()
        self.encoder = SlotFusionEncoder(cfg, input_dims, present)
        self.editor = RetrievalEditor(self.encoder, cfg)

    def build_memory_bank(self, loader, device):
        self.editor.bank.rebuild(self.encoder, loader, device)

    def forward(self, inputs):
        proxy, raw_score = self.encoder(inputs)
        return self.editor(proxy, raw_score) if self.editor.bank.ready else raw_score

    def loss(self, inputs, labels):
        """Training objective. Once the memory bank exists, the edited score is trained
        with L1 plus the direction loss and the batch is written into the bank."""
        proxy, raw_score = self.encoder(inputs)
        if not self.editor.bank.ready:
            return torch.mean(torch.abs(raw_score - labels))
        pred = self.editor(proxy, raw_score)
        self.editor.bank.write(proxy, labels)
        return torch.mean(torch.abs(pred - labels)) + LAMBDA_DIR * self.editor.direction_loss(raw_score, pred, labels)
