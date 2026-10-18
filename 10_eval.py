"""10_eval.py — 損失だけでなく perplexity（当てづらさ）で評価し、進捗を見やすく表示する。

学ぶこと:
- 言語モデルの標準指標 perplexity = exp(cross entropy)。小さいほど良い。
- AverageMeter / ProgressMeter（旧講義から継続）で、移動平均や [i/N] 表示を整える。
- train/val を分けて定期評価する（過学習の検知）。

使い方:  python 10_eval.py
"""
from __future__ import annotations

import math

import torch

from data_utils import get_batch, load_dataset
from gpt import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
max_iters, eval_interval, batch_size, lr = 3000, 500, 32, 3e-4
cfg = GPTConfig(block_size=128, n_layer=4, n_head=4, n_embd=192, dropout=0.1)


class AverageMeter:
    """旧講義と同じ移動平均メータ。name と表示書式を持つ。"""

    def __init__(self, name: str, fmt: str = ":.4f"):
        self.name, self.fmt = name, fmt
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name} {self.val:{self.fmt[1:]}} (avg {self.avg:{self.fmt[1:]}})"


@torch.no_grad()
def evaluate(model, data, iters=100):
    model.eval()
    loss_meter = AverageMeter("loss")
    for _ in range(iters):
        x, y = get_batch(data, cfg.block_size, batch_size, device)
        _, loss = model(x, y)
        loss_meter.update(loss.item(), x.size(0))
    model.train()
    ppl = math.exp(min(loss_meter.avg, 20))   # exp のオーバーフロー回避
    return loss_meter.avg, ppl


def main():
    train_data, val_data, tok = load_dataset()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"device={device}  params={model.num_params():,}")

    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            tr_loss, tr_ppl = evaluate(model, train_data, iters=50)
            va_loss, va_ppl = evaluate(model, val_data, iters=50)
            print(f"[{it:4d}/{max_iters}] "
                  f"train loss {tr_loss:.4f} ppl {tr_ppl:.2f} | "
                  f"val loss {va_loss:.4f} ppl {va_ppl:.2f}")
        x, y = get_batch(train_data, cfg.block_size, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


if __name__ == "__main__":
    main()
