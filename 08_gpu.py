"""08_gpu.py — GPU を使い、学習速度（tokens/sec）を測る。

学ぶこと:
- model と data を device('cuda') に送れば、計算は自動的に GPU で行われる。
- スループット（tokens/sec）という、研究で最重要の指標の測り方。
- GPU の同期（torch.cuda.synchronize）を忘れると時間を正しく測れないこと。

試すこと:
- batch_size / block_size を変えて tokens/sec と GPU メモリ使用量がどう動くか。
"""
from __future__ import annotations

import time

import torch

from data_utils import get_batch, load_dataset
from gpt import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

batch_size = 32
cfg = GPTConfig(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.0)


def main():
    train_data, _, tok = load_dataset()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    print(f"device={device}  params={model.num_params():,}")
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name())

    warmup, measure = 5, 20
    for step in range(warmup + measure):
        if step == warmup and device == "cuda":
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        x, y = get_batch(train_data, cfg.block_size, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    else:
        t0 = time.perf_counter() - 0  # CPU 近似
    dt = time.perf_counter() - t0
    tokens = measure * batch_size * cfg.block_size
    print(f"{measure} step: {dt:.3f}s  ->  {tokens / dt:,.0f} tokens/sec")
    if device == "cuda":
        print(f"GPU メモリ最大: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print("ヒント: 時間計測の前後で cuda.synchronize() を呼ぶ（GPU は非同期実行のため）。")


if __name__ == "__main__":
    main()
