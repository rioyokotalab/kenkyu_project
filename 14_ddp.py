"""14_ddp.py — 13 + 分散データ並列（DDP）。課題と sweep の本体。

学ぶこと:
- torchrun で複数 GPU を使う。各 GPU が別のミニバッチを処理し、勾配を平均（all-reduce）する。
- rank 0 だけが表示・記録・保存を行う。

動かし方:
  (A) 1 GPU（sweep の 1 試行。job.sh から wandb agent 経由）:  python 14_ddp.py --lr 3e-4
  (B) 1 ノード 4 GPU:  torchrun --standalone --nproc_per_node=4 14_ddp.py --max_iters 5000
"""
import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

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

ddp = "RANK" in os.environ                           # torchrun が RANK/LOCAL_RANK/WORLD_SIZE を設定する
if ddp:
    dist.init_process_group(backend="nccl")
    rank, local_rank, world_size = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
else:
    rank, local_rank, world_size = 0, 0, 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
master = rank == 0
torch.manual_seed(args.seed + rank)                  # rank ごとに別のミニバッチ
cfg = GPTConfig(block_size=args.block_size, n_layer=args.n_layer, n_head=args.n_head,
                n_embd=args.n_embd, dropout=args.dropout, flash=True)
use_bf16 = device != "cpu" and torch.cuda.is_bf16_supported()
amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
amp_device = "cuda" if device != "cpu" else "cpu"


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
        with torch.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=use_bf16):
            _, loss, _ = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    train_data, val_data, tok = load_dataset(args.data, tokenizer=args.tokenizer)
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    raw_model = model
    if ddp:
        model = DDP(model, device_ids=[local_rank])   # 勾配を GPU 間で平均する
    if args.compile:
        model = torch.compile(model)
    if master:
        print(f"world_size={world_size}  device={device}  bf16={use_bf16}  "
              f"params={raw_model.num_params():,}  args={vars(args)}")
        wandb.init(project="kenkyu_project", config=vars(args))
        samples = wandb.Table(columns=["step", "sample"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_loss = AverageMeter("loss")

    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for it in range(args.max_iters + 1):
        if master and it % args.eval_interval == 0:
            tr, va = estimate_loss(model, train_data), estimate_loss(model, val_data)
            print(f"[{it:5d}/{args.max_iters}]  {train_loss}  "
                  f"train {tr:.4f}  val {va:.4f}  val_ppl {math.exp(va):.2f}")
            wandb.log({"step": it, "train_loss": tr, "val_loss": va, "val_ppl": math.exp(va)})
            start = torch.zeros((1, 1), dtype=torch.long, device=device)
            samples.add_data(it, tok.decode(raw_model.generate(start, 200, top_k=50, use_cache=True)[0].tolist()))
            train_loss = AverageMeter("loss")
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, cfg.block_size, args.batch_size, device)
            with torch.autocast(device_type=amp_device, dtype=torch.bfloat16, enabled=use_bf16):
                _, loss, _ = model(x, y)
            (loss / args.grad_accum).backward()
        train_loss.update(loss.item())
        opt.step()
    if device != "cpu":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tokens = args.max_iters * args.grad_accum * args.batch_size * cfg.block_size * world_size
    if master:
        print(f"{args.max_iters} step: {dt:.1f}s  {tokens / dt:,.0f} tokens/sec (全 GPU 合計)")
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
        print(tok.decode(raw_model.generate(start, 500, temperature=1.0, top_k=50, use_cache=True)[0].tolist()))
        torch.save({"model": raw_model.state_dict(), "args": vars(args)}, "ckpt.pt")
        wandb.log({"samples": samples})
        wandb.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
