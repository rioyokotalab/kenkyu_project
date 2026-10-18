"""10_meter.py — 09 + perplexity と AverageMeter（見やすい進捗表示）。

学ぶこと:
- 言語モデルの標準指標 perplexity = exp(cross entropy)。小さいほど良い。
- 移動平均メータで学習中の loss の推移を見やすくする（旧講義から継続）。
"""
import argparse
import math
import time

import torch

from data_utils import get_batch, load_dataset
from gpt import GPT, GPTConfig

parser = argparse.ArgumentParser()
parser.add_argument("--max_iters", type=int, default=2000)
parser.add_argument("--eval_interval", type=int, default=200)
parser.add_argument("--eval_iters", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--block_size", type=int, default=64)
parser.add_argument("--n_layer", type=int, default=4)
parser.add_argument("--n_head", type=int, default=4)
parser.add_argument("--n_embd", type=int, default=128)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=1337)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
cfg = GPTConfig(block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head,
                n_embd=args.n_embd, dropout=args.dropout)


class AverageMeter:
    """値の移動平均を保持する。update(val) で更新、avg で平均。"""

    def __init__(self, name):
        self.name, self.sum, self.count, self.val = name, 0.0, 0, 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)

    def __str__(self):
        return f"{self.name} {self.val:.4f} (avg {self.avg:.4f})"


@torch.no_grad()
def estimate_loss(model, data):
    model.eval()
    losses = torch.zeros(args.eval_iters)
    for k in range(args.eval_iters):
        x, y = get_batch(data, cfg.block_size, args.batch_size, device)
        _, loss, _ = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    train_data, val_data, tok = load_dataset()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    print(f"device={device}  params={model.num_params():,}  args={vars(args)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loss = AverageMeter("loss")

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0:
            tr, va = estimate_loss(model, train_data), estimate_loss(model, val_data)
            print(f"[{it:5d}/{args.max_iters}]  {train_loss}  "
                  f"train {tr:.4f}  val {va:.4f}  val_ppl {math.exp(va):.2f}")
            train_loss = AverageMeter("loss")
        x, y = get_batch(train_data, cfg.block_size, args.batch_size, device)
        _, loss, _ = model(x, y)
        train_loss.update(loss.item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{args.max_iters} step: {dt:.1f}s  {args.max_iters * args.batch_size * cfg.block_size / dt:,.0f} tokens/sec")

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tok.decode(model.generate(start, 500, temperature=1.0, top_k=50)[0].tolist()))


if __name__ == "__main__":
    main()
