"""05_attention.py — self-attention を一から導く。

学ぶこと:
- 「各位置が、自分より前の位置の情報を重み付きで集める」のが self-attention。
- まず素朴な平均（過去の平均を取る）から始め、それを softmax(Q K^T / sqrt(d)) V に一般化する。
- 未来を見ないための因果マスク（下三角）。

ここは計算の中身を目で見るための小さなデモ。07 で GPT として組み上げる。
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(0)
B, T, C = 1, 8, 4   # batch, 系列長, チャネル
x = torch.randn(B, T, C)

# (1) 素朴版: 各位置 t は「0..t の平均」を取る（過去だけを見る）
tril = torch.tril(torch.ones(T, T))
w = tril / tril.sum(1, keepdim=True)       # 行ごとに正規化した重み（過去を一様平均）
avg = w @ x
print("素朴な過去平均の重み w:\n", w)

# (2) self-attention: 重みを「内容から」決める。Q(問い)・K(鍵)・V(値) を作る。
head_size = 4
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k, q, v = key(x), query(x), value(x)

att = q @ k.transpose(-2, -1) / head_size**0.5   # (B,T,T) 類似度スコア
att = att.masked_fill(tril == 0, float("-inf"))  # 未来を隠す（因果マスク）
att = F.softmax(att, dim=-1)                      # 行ごとに確率化
out = att @ v                                      # 値を重み付き集約

print("\n内容依存の attention 重み att[0]:\n", att[0])
print("出力の形:", tuple(out.shape))
print("\nポイント: sqrt(head_size) で割るのは、内積が大きくなりすぎて softmax が尖るのを防ぐため。")
print("この 1 ヘッド版を複数並べ、残差と FFN を足したのが Transformer ブロック（06）。")
