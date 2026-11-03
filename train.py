"""Train BRACE on CMU-MOSI or CMU-MOSEI with one fixed observed modality subset.

    python train.py --dataset mosei --subset lva --output runs/mosei/lva
"""
import argparse
import json
import os
from pathlib import Path

import torch

from brace import BANK_WARMUP_EPOCHS, BRACE, MODALITIES, SUBSETS
from brace.data import build_loaders
from brace.engine import evaluate, train_one_epoch
from brace.utils import load_config, lr_factor, set_seed

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=SUBSETS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="mosei", choices=("mosi", "mosei"))
    parser.add_argument("--config", help="defaults to configs/<dataset>.yaml")
    args = parser.parse_args()

    cfg, train_cfg, model_cfg = load_config(args.config or ROOT / "configs" / f"{args.dataset}.yaml", args.subset)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg["seed"])
    device = torch.device("cuda")
    loaders, present = build_loaders(ROOT / cfg["data"], args.subset, train_cfg["batch_size"])
    input_dims = {m: loaders["train"].dataset.tensors[i].shape[-1] for i, m in enumerate(MODALITIES)}
    model = BRACE(model_cfg, input_dims, present).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    n_epochs = train_cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: lr_factor(e, n_epochs))
    scaler = torch.amp.GradScaler("cuda")

    log_path = out_dir / "metrics.jsonl"
    log_path.write_text("")
    for epoch in range(1, n_epochs + 1):
        if epoch == BANK_WARMUP_EPOCHS + 1:
            model.build_memory_bank(loaders["train"], device)
        record = {
            "epoch": epoch,
            "train_loss": train_one_epoch(model, loaders["train"], optimizer, scaler, device),
            "valid": evaluate(model, loaders["valid"], device),
            "test": evaluate(model, loaders["test"], device),
        }
        scheduler.step()
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"epoch {epoch:3d} loss {record['train_loss']:.4f} | "
            + " | ".join(f"{s} " + " ".join(f"{k}={v * 100:.2f}" for k, v in record[s].items() if k != "mae")
                         for s in ("valid", "test")),
            flush=True,
        )


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
