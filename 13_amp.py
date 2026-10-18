"""13_amp.py — 12 + 高速化: bf16 混合精度・Flash Attention・torch.compile・勾配累積・KV-cache。

学ぶこと（横田研究室の「効率化」研究の入口）:
- bf16（autocast）: 精度をほぼ保ったまま高速・省メモリ ＝ 低精度計算の研究。
- Flash Attention（cfg.flash=True → F.scaled_dot_product_attention）: 速くて省メモリ。
- torch.compile: カーネル融合で高速化（初回はコンパイルで遅い）。
- 勾配累積（--grad_accum）: 実効バッチを GPU メモリ以上に。
- KV-cache（generate(use_cache=True)）: 生成時に過去の key/value を使い回して高速化。

試すこと: 12 と tokens/sec を比べる。--grad_accum 4 で実効バッチ 128 にしてみる。
"""
import argparse
import math
import time

import torch
import wandb

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
parser.add_argument("--tokenizer", choices=["char", "bpe"], default="bpe")
parser.add_argument("--data", type=str, default="input.txt")
parser.add_argument("--grad_accum", type=int, default=1)
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
cfg = GPTConfig(block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head,
                n_embd=args.n_embd, dropout=args.dropout, flash=True)
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float32


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
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
            _, loss, _ = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    train_data, val_data, tok = load_dataset(args.data, tokenizer=args.tokenizer)
    cfg.vocab_size = tok.vocab_size
    print(f"tokenizer={args.tokenizer}  vocab_size={tok.vocab_size:,}  train tokens={len(train_data):,}")
    model = GPT(cfg).to(device)
    raw_model = model                                # compile 前のモデル（generate で使う）
    if args.compile:
        model = torch.compile(model)
    print(f"device={device}  bf16={use_bf16}  flash={cfg.flash}  compile={args.compile}  "
          f"params={raw_model.num_params():,}  args={vars(args)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loss = AverageMeter("loss")
    wandb.init(project="kenkyu_project", config=vars(args))
    samples = wandb.Table(columns=["step", "sample"])

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for it in range(args.max_iters + 1):
        if it % args.eval_interval == 0:
            tr, va = estimate_loss(model, train_data), estimate_loss(model, val_data)
            print(f"[{it:5d}/{args.max_iters}]  {train_loss}  "
                  f"train {tr:.4f}  val {va:.4f}  val_ppl {math.exp(va):.2f}")
            wandb.log({"step": it, "train_loss": tr, "val_loss": va, "val_ppl": math.exp(va)})
            start = torch.zeros((1, 1), dtype=torch.long, device=device)
            samples.add_data(it, tok.decode(raw_model.generate(start, 200, top_k=50, use_cache=True)[0].tolist()))
            train_loss = AverageMeter("loss")
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):             # 勾配累積: 小さなバッチを何回か足す
            x, y = get_batch(train_data, cfg.block_size, args.batch_size, device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                _, loss, _ = model(x, y)
            (loss / args.grad_accum).backward()
        train_loss.update(loss.item())
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tokens = args.max_iters * args.grad_accum * args.batch_size * cfg.block_size
    print(f"{args.max_iters} step: {dt:.1f}s  {tokens / dt:,.0f} tokens/sec")

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tok.decode(raw_model.generate(start, 500, temperature=1.0, top_k=50, use_cache=True)[0].tolist()))
    wandb.log({"samples": samples})
    wandb.finish()


if __name__ == "__main__":
    main()
