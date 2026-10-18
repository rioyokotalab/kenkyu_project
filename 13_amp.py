"""13_amp.py — 学習と生成を速くする: bf16・torch.compile・Flash Attention・KV-cache。

学ぶこと（横田研究室の「効率化」研究に直結）:
- 混合精度 bf16（autocast）で、精度をほぼ保ったまま高速・省メモリに（＝低精度計算の研究の入口）。
- torch.compile でカーネルを融合して高速化。
- Flash Attention（F.scaled_dot_product_attention, cfg.flash=True）。
- 勾配累積（gradient accumulation）で、実効バッチサイズを GPU メモリ以上に。
- 生成の KV-cache: 既に計算した key/value を使い回し、生成を高速化。

使い方:  python 13_amp.py
"""
from __future__ import annotations

import time

import torch
from torch.nn import functional as F

from data_utils import get_batch, load_dataset
from gpt import GPT, GPTConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

batch_size, grad_accum, steps, lr = 16, 4, 50, 3e-4   # 実効バッチ = batch_size * grad_accum
cfg = GPTConfig(block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.0, flash=True)


@torch.no_grad()
def generate_with_kv_cache(model, idx, max_new_tokens, temperature=0.8, top_k=50):
    """KV-cache の考え方を示す簡易版。

    本格的な KV-cache は各 attention 層に past_key/value を保持するが、ここでは
    『block_size を超えたら末尾だけ残す』素直な実装で、使い回しの概念を示す。
    実運用では attention 内に cache を持たせて O(T) を O(1)/step にする。
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    return idx


def main():
    train_data, _, tok = load_dataset()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)

    # bf16 が使えるか（Hopper/Ampere 等）。使えなければ fp32。
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32
    print(f"device={device}  bf16={use_bf16}  flash={cfg.flash}  params={model.num_params():,}")

    if device == "cuda":
        model = torch.compile(model)          # カーネル融合（初回は遅い＝コンパイル）

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        for micro in range(grad_accum):       # 勾配累積
            x, y = get_batch(train_data, cfg.block_size, batch_size, device)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=amp_dtype):
                _, loss = model(x, y)
            (loss / grad_accum).backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tok_done = steps * grad_accum * batch_size * cfg.block_size
    print(f"{steps} step（実効バッチ {batch_size * grad_accum}）: {dt:.2f}s  {tok_done / dt:,.0f} tokens/sec")

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    out = generate_with_kv_cache(base, start, 200)
    print("\n--- 生成例 ---\n" + tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
