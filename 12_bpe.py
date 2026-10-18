"""12_bpe.py — 11 + BPE トークナイザ（tiktoken）。日本語も学習できる。

学ぶこと:
- 文字単位（char）ではなく、よく出る部分文字列をまとめる BPE。実際の LLM（GPT・Swallow）はこの方式。
- トークナイザを差し替えるだけで同じコードが動く（vocab_size が 65 → 約10万に変わる）。
- 系列が短くなる代わりに語彙（埋め込み行列）が巨大になる、というトレードオフ。

使い方: python 12_bpe.py --tokenizer bpe
        python 12_bpe.py --tokenizer bpe --data japanese.txt   # 自分の日本語テキストで
準備:   uv pip install tiktoken
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
    train_data, val_data, tok = load_dataset(args.data, tokenizer=args.tokenizer)
    cfg.vocab_size = tok.vocab_size
    print(f"tokenizer={args.tokenizer}  vocab_size={tok.vocab_size:,}  train tokens={len(train_data):,}")
    model = GPT(cfg).to(device)
    print(f"device={device}  params={model.num_params():,}  args={vars(args)}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loss = AverageMeter("loss")
    wandb.init(entity="rioyokotalab", project="kenkyu_project", config=vars(args))
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
            samples.add_data(it, tok.decode(model.generate(start, 200, top_k=50)[0].tolist()))
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
    wandb.log({"samples": samples})
    wandb.finish()


if __name__ == "__main__":
    main()
