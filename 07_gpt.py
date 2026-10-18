"""07_gpt.py — 小さな GPT を学習して文章を生成する。

学ぶこと:
- gpt.py の GPT（埋め込み + Transformer ブロック×n_layer + 出力層）を学習する。
- 04 の bigram より、文脈を見る分だけ生成がそれらしくなることを体感する。

試すこと:
- n_layer / n_head / n_embd / block_size を変えて loss と生成文がどう変わるか。
- max_iters を増やす。GPU があれば block_size を伸ばす。

CPU でも動くが遅い。GPU（08 以降）で回すのが前提。
"""
from __future__ import annotations

import torch

from data_utils import get_batch, load_dataset
from gpt import GPT, GPTConfig

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

# CPU でも試せるよう控えめな既定値。GPU では大きくしてよい。
max_iters, eval_interval, lr = 3000, 500, 3e-4
batch_size = 32
cfg = GPTConfig(block_size=64, n_layer=4, n_head=4, n_embd=128, dropout=0.1)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, iters=50):
    model.eval()
    out = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, cfg.block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    train_data, val_data, tok = load_dataset()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    print(f"device={device}  パラメータ数={model.num_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {it:4d}  train {losses['train']:.4f}  val {losses['val']:.4f}")
        x, y = get_batch(train_data, cfg.block_size, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- 生成例 ---")
    print(tok.decode(model.generate(start, 500, temperature=1.0, top_k=50)[0].tolist()))


if __name__ == "__main__":
    main()
