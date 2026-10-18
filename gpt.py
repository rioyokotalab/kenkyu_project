"""gpt.py — 本講義で作る小さな GPT（言語モデル）の正準実装。

05〜07 で「self-attention → Transformer ブロック → GPT」を一から組み立てて理解したあと、
08 以降のスクリプトはこのファイルの GPT を `from gpt import GPT, GPTConfig` で読み込んで使う。
nanoGPT (Andrej Karpathy, MIT License) の設計を本講義向けに簡潔化したもの。

ポイント:
- CausalSelfAttention は学習用の素の実装（05 で導出した式）と、
  `F.scaled_dot_product_attention`（Flash Attention 相当・13 で使う）を切り替えられる。
- generate() は KV-cache 無しの素直な実装。KV-cache 版は 13_amp.py で扱う。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65          # 03 の char-level 語彙（Tiny Shakespeare）を既定にする
    block_size: int = 256         # 文脈長（何トークン先まで見るか）
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384             # n_embd は n_head で割り切れること
    dropout: float = 0.1
    bias: bool = False            # LayerNorm/Linear に bias を使うか（False の方が少し速く安定）
    flash: bool = False           # True なら F.scaled_dot_product_attention を使う（13 で True に）


class CausalSelfAttention(nn.Module):
    """因果的（未来を見ない）マルチヘッド自己注意。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd は n_head で割り切れる必要があります"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout
        self.flash = cfg.flash
        # Q, K, V をまとめて 1 本の Linear で作る（3*n_embd 出力）
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # 素の実装で使う下三角マスク（未来を隠す）。flash=True のときは使わない。
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                1, 1, cfg.block_size, cfg.block_size
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape                       # batch, time(系列長), channel(=n_embd)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim) に並べ替え
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        if self.flash:
            # Flash Attention（PyTorch 2 以降）。速くて省メモリ。is_causal で因果マスクを自動適用。
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # 素の実装（05 で導いた式）: softmax(QK^T / sqrt(d)) V、未来はマスク
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # ヘッドを連結
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """位置ごとの 2 層フィードフォワード（Transformer ブロックの後半）。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Transformer ブロック: 残差接続つきの (LN -> Attention) と (LN -> MLP)。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))   # pre-norm 残差接続
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        # weight tying（入力埋め込みと出力の重みを共有）
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"系列長 {T} が block_size {self.cfg.block_size} を超えています"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.token_embedding(idx) + self.position_embedding(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """素直な自己回帰生成（KV-cache 無し）。13_amp.py で KV-cache 版を扱う。"""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]      # block_size を超えたら末尾だけ使う
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
