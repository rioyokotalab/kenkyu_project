"""00_numpy.py — ライブラリに頼らず、NumPy だけで学習の仕組みを体験する。

学ぶこと:
- 順伝播・損失・逆伝播（勾配）・パラメータ更新という学習の 1 ステップを、全部手で書く。
- PyTorch の autograd（02）が裏で何を自動化しているのかを、先に手で理解しておく。

ここでは「次の値を予測する」ごく小さな 2 層ネットワークを、ランダムなデータで回す。
"""
import numpy as np

rng = np.random.default_rng(0)

batch_size, D_in, H, D_out = 32, 16, 64, 8
lr = 1e-3
epochs = 200

x = rng.standard_normal((batch_size, D_in))
y = rng.standard_normal((batch_size, D_out))
w1 = rng.standard_normal((D_in, H)) * 0.1
w2 = rng.standard_normal((H, D_out)) * 0.1

for epoch in range(epochs):
    # --- 順伝播 ---
    h = x @ w1
    h_relu = np.maximum(h, 0)
    y_pred = h_relu @ w2
    loss = np.mean((y_pred - y) ** 2)

    # --- 逆伝播（連鎖律を手で）---
    grad_y_pred = 2.0 * (y_pred - y) / y.size
    grad_w2 = h_relu.T @ grad_y_pred
    grad_h_relu = grad_y_pred @ w2.T
    grad_h = grad_h_relu * (h > 0)        # ReLU の勾配
    grad_w1 = x.T @ grad_h

    # --- 更新（勾配降下）---
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2

    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  loss {loss:.4f}")

print("この forward/backward を自動でやってくれるのが PyTorch の autograd（02 で見る）。")
