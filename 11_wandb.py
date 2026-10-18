"""11_wandb.py — Weights & Biases で実験を記録する。

学ぶこと:
- 学習曲線（loss・perplexity）と実験条件（config）を自動で記録・比較する。
- 生成テキストのサンプルを wandb.Table に残し、数値だけでなく「出力の質」も追う。
- compute ノードは外部ネットに出られないことがあるので、その場合は offline にする。

準備:
    uv pip install wandb
    wandb login                      # 初回のみ
TSUBAME の計算ノードでネットに出られない場合:
    export WANDB_MODE=offline        # ジョブ後に `wandb sync` でアップロード

使い方:  python 11_wandb.py
"""
from __future__ import annotations

import math
import os

import torch
import wandb

from data_utils import get_batch, load_dataset
from gpt import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

WANDB_ENTITY = "rioyokotalab"
WANDB_PROJECT = "kenkyu_project"

max_iters, eval_interval, batch_size, lr = 3000, 500, 32, 3e-4
cfg = GPTConfig(block_size=128, n_layer=4, n_head=4, n_embd=192, dropout=0.1)


@torch.no_grad()
def eval_loss(model, data, iters=50):
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = get_batch(data, cfg.block_size, batch_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    train_data, val_data, tok = load_dataset()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT,
               config={**cfg.__dict__, "lr": lr, "batch_size": batch_size})
    samples = wandb.Table(columns=["step", "sample"])

    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            tr, va = eval_loss(model, train_data), eval_loss(model, val_data)
            start = torch.zeros((1, 1), dtype=torch.long, device=device)
            text = tok.decode(model.generate(start, 200, temperature=0.8, top_k=50)[0].tolist())
            samples.add_data(it, text)
            wandb.log({"train_loss": tr, "val_loss": va,
                       "val_ppl": math.exp(min(va, 20)), "step": it})
            print(f"step {it:4d}  train {tr:.4f}  val {va:.4f}")
        x, y = get_batch(train_data, cfg.block_size, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    wandb.log({"samples": samples})
    wandb.finish()


if __name__ == "__main__":
    main()
