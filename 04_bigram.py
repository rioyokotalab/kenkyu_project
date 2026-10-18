"""04_bigram.py — 一番単純な言語モデル（bigram）を一から。

学ぶこと:
- 「次のトークンを当てる」を、埋め込み 1 枚（Embedding）だけで解く最小モデル。
- 学習ループ（loss -> backward -> step）と、文章の生成（自己回帰サンプリング）。

これが言語モデルの "Hello World"。05 以降で self-attention を足して賢くしていく。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F

from data_utils import CharTokenizer, get_batch, load_text

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
block_size, batch_size, steps, lr = 8, 32, 3000, 1e-2


class BigramLM(nn.Module):
    """各トークンから「次のトークンの分布」を直接引くだけのモデル。"""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)  # 行=今のトークン, 列=次のトークンのスコア

    def forward(self, idx, targets=None):
        logits = self.table(idx)                           # (B, T, vocab)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -1:])          # 直前の 1 トークンだけで決まる（bigram）
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def main():
    text = load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    model = BigramLM(tok.vocab_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        x, y = get_batch(train_data, block_size, batch_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"step {step:4d}  train_loss {loss.item():.4f}")

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("\n--- 生成例（bigram なのでまだ支離滅裂）---")
    print(tok.decode(model.generate(start, 300)[0].tolist()))


if __name__ == "__main__":
    main()
