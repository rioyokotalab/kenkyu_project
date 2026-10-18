"""06_block.py — マルチヘッド注意 + FFN + 残差 + LayerNorm = Transformer ブロック。

学ぶこと:
- 1 ヘッド（05）を複数並べて連結する＝マルチヘッド注意。
- その後に位置ごとの 2 層 FFN。どちらも残差接続（x + f(x)）と pre-LayerNorm で包む。
- このブロックを n_layer 段重ねると GPT 本体になる。

ここでは gpt.py の Block を 1 つ作り、入出力の形が保たれることを確認する。
"""
import torch

from gpt import Block, GPTConfig

torch.manual_seed(0)
cfg = GPTConfig(vocab_size=65, block_size=16, n_layer=1, n_head=4, n_embd=32, dropout=0.0)

block = Block(cfg)
x = torch.randn(2, cfg.block_size, cfg.n_embd)   # (B, T, n_embd)
y = block(x)

print(f"ブロック入力:  {tuple(x.shape)}")
print(f"ブロック出力:  {tuple(y.shape)}   <- 形は変わらない（だから何段でも積める）")
print(f"ヘッド数 n_head={cfg.n_head}, 1 ヘッドの次元 head_dim={cfg.n_embd // cfg.n_head}")
print(f"ブロックのパラメータ数: {sum(p.numel() for p in block.parameters()):,}")
print("\n残差接続 x + attn(ln(x)) と x + mlp(ln(x)) により、深くしても勾配が流れやすい。")
print("次は 07_gpt.py で、埋め込み + このブロック×n_layer + 出力層 を GPT として学習する。")
