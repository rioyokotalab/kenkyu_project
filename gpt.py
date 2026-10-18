"""gpt.py — 本講義で作る小さな GPT（言語モデル）。

05 で self-attention、06 で Transformer ブロックを一から組み立てて理解したあと、
07 以降はこのファイルを `from gpt import GPT, GPTConfig` で読み込んで学習に使う。
nanoGPT (Andrej Karpathy, MIT License) の設計を本講義向けに簡潔化したもの。

- flash=False: 05 で導いた式どおりの素の attention（学習用に読みやすい）
- flash=True : F.scaled_dot_product_attention（Flash Attention。13 で使う）
- generate(use_cache=True): KV-cache 付き生成（13 で使う）
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

# SDPA の cuDNN バックエンドは使わない（Flash Attention カーネルを使う）。
# クラスタでは module が古い cuDNN を LD_LIBRARY_PATH に載せていることがあり、
# pip の torch が同梱する cuDNN と食い違って実行時エラーになるため。
if torch.cuda.is_available() and hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
    torch.backends.cuda.enable_cudnn_sdp(False)


@dataclass
class GPTConfig:
    vocab_size: int = 65      # 03 の char-level 語彙（Tiny Shakespeare）
    block_size: int = 256     # 文脈長（最大で何トークン前まで見るか）
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384         # n_head で割り切れること
    dropout: float = 0.1
    bias: bool = False
    flash: bool = False


class CausalSelfAttention(nn.Module):
    """因果的（未来を見ない）マルチヘッド自己注意。KV-cache に対応。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.n_embd, self.dropout, self.flash = cfg.n_head, cfg.n_embd, cfg.dropout, cfg.flash
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)   # Q,K,V をまとめて作る
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)   # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        if past_kv is not None:                     # KV-cache: 過去の key/value を前に連結
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        present = (k, v) if use_cache else None
        Tk = k.size(2)                              # 鍵の長さ（過去を含む）

        if self.flash:
            # 新しいトークンが 1 個で過去がある場合は全ての過去を見てよいので因果マスク不要
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0,
                is_causal=(past_kv is None))
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)          # (B, nh, T, Tk)
            if past_kv is None:                                      # 通常の学習・prefill
                att = att.masked_fill(self.mask[:, :, :T, :Tk] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y)), present


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Transformer ブロック = 残差つき (LN → Attention) と (LN → MLP)。"""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, past_kv=None, use_cache=False):
        a, present = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + a                       # 残差接続
        x = x + self.mlp(self.ln_2(x))
        return x, present


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
        self.token_embedding.weight = self.lm_head.weight   # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, past_kvs=None, use_cache=False):
        B, T = idx.shape
        past_len = 0 if past_kvs is None else past_kvs[0][0].size(2)
        assert past_len + T <= self.cfg.block_size, "block_size を超えています"
        pos = torch.arange(past_len, past_len + T, device=idx.device)   # cache 分だけ位置をずらす
        x = self.drop(self.token_embedding(idx) + self.position_embedding(pos))
        presents = []
        for i, block in enumerate(self.blocks):
            x, present = block(x, past_kv=None if past_kvs is None else past_kvs[i], use_cache=use_cache)
            presents.append(present)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss, (presents if use_cache else None)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=False):
        """自己回帰生成。use_cache=True なら KV-cache で毎ステップ 1 トークン分だけ計算する。"""
        self.eval()
        past = None
        inp = idx
        for _ in range(max_new_tokens):
            if use_cache:
                # 2 回目以降は最後の 1 トークンだけ入力し、過去は cache から
                logits, _, past = self(inp, past_kvs=past, use_cache=True)
            else:
                logits, _, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
            inp = idx_next
            if use_cache and idx.size(1) >= self.cfg.block_size:   # 文脈長の上限で cache を捨てて再開
                past, inp = None, idx[:, -self.cfg.block_size + 1:]
        return idx
