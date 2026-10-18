"""09_config.py — 設定を dataclass にまとめ、チェックポイントで再開できる学習器。

学ぶこと:
- ハイパーパラメータを 1 つの dataclass + コマンドライン引数に集約する（再現性と実験管理の基礎）。
- 乱数シードの固定。
- チェックポイント（モデル・optimizer・step）の保存と再開。

使い方:
    python 09_config.py --max_iters 2000 --n_layer 4 --lr 3e-4
    python 09_config.py --resume ckpt.pt        # 途中から再開
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict, fields

import torch

from data_utils import get_batch, load_dataset
from gpt import GPT, GPTConfig


@dataclass
class TrainConfig:
    max_iters: int = 2000
    eval_interval: int = 500
    batch_size: int = 32
    lr: float = 3e-4
    seed: int = 1337
    block_size: int = 64
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    out: str = "ckpt.pt"


def parse_args() -> tuple[TrainConfig, str | None]:
    p = argparse.ArgumentParser(description="小さな GPT の学習（設定とチェックポイント）")
    for f in fields(TrainConfig):
        p.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    p.add_argument("--resume", type=str, default=None, help="再開するチェックポイント")
    a = p.parse_args()
    cfg = TrainConfig(**{f.name: getattr(a, f.name) for f in fields(TrainConfig)})
    return cfg, a.resume


def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    tcfg, resume = parse_args()
    set_seed(tcfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data, val_data, tok = load_dataset()

    mcfg = GPTConfig(vocab_size=tok.vocab_size, block_size=tcfg.block_size,
                     n_layer=tcfg.n_layer, n_head=tcfg.n_head,
                     n_embd=tcfg.n_embd, dropout=tcfg.dropout)
    model = GPT(mcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr)

    start_iter = 0
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_iter = ckpt["iter"]
        print(f"{resume} から再開（step {start_iter}）")

    print(f"device={device}  params={model.num_params():,}  config={asdict(tcfg)}")
    for it in range(start_iter, tcfg.max_iters + 1):
        if it % tcfg.eval_interval == 0:
            x, y = get_batch(val_data, mcfg.block_size, tcfg.batch_size, device)
            with torch.no_grad():
                _, vloss = model(x, y)
            print(f"step {it:4d}  val_loss {vloss.item():.4f}")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "iter": it, "config": asdict(mcfg)}, tcfg.out)
        x, y = get_batch(train_data, mcfg.block_size, tcfg.batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"チェックポイントを保存: {tcfg.out}")


if __name__ == "__main__":
    main()
