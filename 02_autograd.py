"""02_autograd.py — PyTorch の自動微分。

学ぶこと:
- requires_grad=True のテンソルに対して、loss.backward() が勾配を自動計算する。
- 00 で手で書いた逆伝播を、PyTorch が全部やってくれることを確認する。
"""
import torch

torch.manual_seed(0)

batch_size, D_in, H, D_out = 32, 16, 64, 8
lr = 1e-3
epochs = 200

x = torch.randn(batch_size, D_in)
y = torch.randn(batch_size, D_out)
w1 = (torch.randn(D_in, H) * 0.1).requires_grad_(True)
w2 = (torch.randn(H, D_out) * 0.1).requires_grad_(True)

for epoch in range(epochs):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)   # 順伝播（00 と同じ 2 層 ReLU）
    loss = (y_pred - y).pow(2).mean()

    loss.backward()                         # ← 逆伝播は自動。w1.grad, w2.grad に勾配が入る
    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad
        w1.grad.zero_()                     # 勾配は累積されるので毎回ゼロに戻す
        w2.grad.zero_()

    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}")

print("backward() が 00 の grad_w1 / grad_w2 を自動で計算している。")
