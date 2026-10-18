"""14_ddp.py — 分散データ並列（DDP）対応の学習器。課題と sweep の本体。

学ぶこと:
- torchrun による複数 GPU 学習（DDP）。各 GPU がミニバッチを分担し、勾配を平均する。
- これまでの要素（bf16・config・perplexity・wandb・チェックポイント）の総まとめ。

2 つの動かし方:
  (A) 1 GPU（sweep の 1 試行。job.sh から `wandb agent` で呼ばれる）:
        python 14_ddp.py --lr 3e-4 --n_layer 6
  (B) 複数 GPU（1 ノード 4 GPU を torchrun で）:
        torchrun --standalone --nproc_per_node=4 14_ddp.py --max_iters 5000

課題はこのスクリプトをベースに sweep.yaml でハイパーパラメータを探索し、
val perplexity と生成例を wandb に記録して提出する。
"""
from __future__ import annotations

import argparse
import math
import os

import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from data_utils import get_batch, load_dataset
from gpt import GPT, GPTConfig

WANDB_ENTITY = "rioyokotalab"
WANDB_PROJECT = "kenkyu_project"


def is_distributed() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def parse_args():
    p = argparse.ArgumentParser(description="GPT 学習（DDP 対応・sweep 本体）")
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", type=str, default="ckpt.pt")
    return p.parse_args()


@torch.no_grad()
def eval_loss(model, data, cfg, batch_size, device, iters=50):
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = get_batch(data, cfg.block_size, batch_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    args = parse_args()

    if is_distributed():
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = "cuda" if torch.cuda.is_available() else "cpu"

    is_master = rank == 0
    torch.manual_seed(args.seed + rank)   # rank ごとに別の乱数（＝別のミニバッチ）

    train_data, val_data, tok = load_dataset()
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size,
                    n_layer=args.n_layer, n_head=args.n_head,
                    n_embd=args.n_embd, dropout=args.dropout, flash=True)
    model = GPT(cfg).to(device)
    raw_model = model
    if is_distributed():
        model = DDP(model, device_ids=[local_rank])

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_bf16 = device != "cpu" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    if is_master:
        print(f"world_size={world_size} device={device} bf16={use_bf16} "
              f"params={raw_model.num_params():,}")
        wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, config=vars(args))

    for it in range(args.max_iters + 1):
        if is_master and it % args.eval_interval == 0:
            va = eval_loss(model, val_data, cfg, args.batch_size, device)
            tr = eval_loss(model, train_data, cfg, args.batch_size, device)
            print(f"step {it:5d}  train {tr:.4f}  val {va:.4f}  ppl {math.exp(min(va, 20)):.2f}")
            wandb.log({"step": it, "train_loss": tr, "val_loss": va,
                       "val_ppl": math.exp(min(va, 20))})
            torch.save({"model": raw_model.state_dict(), "iter": it,
                        "config": vars(cfg) if hasattr(cfg, "__dict__") else {}}, args.out)
        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, cfg.block_size, args.batch_size, device)
            with torch.autocast(device_type="cuda" if device != "cpu" else "cpu", dtype=amp_dtype):
                _, loss = model(x, y)
            (loss / args.grad_accum).backward()
        opt.step()

    if is_master:
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
        text = tok.decode(raw_model.generate(start, 300, temperature=0.8, top_k=50)[0].tolist())
        print("\n--- 生成例 ---\n" + text)
        wandb.finish()
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
