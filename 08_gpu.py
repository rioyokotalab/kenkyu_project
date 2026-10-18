"""08_gpu.py — 07 + GPU。学習速度（tokens/sec）も測る。

学ぶこと:
- device を 'cuda' にして model と data を送れば、計算は自動的に GPU で行われる。
- スループット tokens/sec は研究で最重要の指標。GPU は非同期なので計測の前後で synchronize する。

試すこと: batch_size / block_size を変えて tokens/sec と GPU メモリがどう動くか。
"""
import time

import torch

from data_utils import get_batch, load_dataset
from gpt import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
max_iters, eval_interval, eval_iters = 2000, 200, 50
batch_size, lr = 32, 3e-4
cfg = GPTConfig(block_size=64, n_layer=4, n_head=4, n_embd=128, dropout=0.1)


@torch.no_grad()
def estimate_loss(model, data):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = get_batch(data, cfg.block_size, batch_size, device)
        _, loss, _ = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    train_data, val_data, tok = load_dataset()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    print(f"device={device}  params={model.num_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for it in range(max_iters + 1):
        if it % eval_interval == 0:
            tr, va = estimate_loss(model, train_data), estimate_loss(model, val_data)
            print(f"step {it:5d}  train {tr:.4f}  val {va:.4f}")
        x, y = get_batch(train_data, cfg.block_size, batch_size, device)
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"{max_iters} step: {dt:.1f}s  {max_iters * batch_size * cfg.block_size / dt:,.0f} tokens/sec")
    if device == "cuda":
        print(f"GPU メモリ最大: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(tok.decode(model.generate(start, 500, temperature=1.0, top_k=50)[0].tolist()))


if __name__ == "__main__":
    main()
