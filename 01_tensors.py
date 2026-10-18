"""01_tensors.py — NumPy 配列と PyTorch テンソルの対応。

学ぶこと:
- PyTorch のテンソルは NumPy 配列とほぼ同じ API で、加えて GPU と自動微分に対応する。
- よく使う操作の NumPy <-> PyTorch 対応を確認する。
"""
import numpy as np
import torch

a = np.array([[1.0, 2.0], [3.0, 4.0]])
t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

print("NumPy と PyTorch の対応:")
print("  np.maximum(a, 0)   <-> t.clamp(min=0)")
print("  a.dot(a)           <-> t.mm(t)")
print("  np.square(a)       <-> t.pow(2)")
print("  a.copy()           <-> t.clone()")

print("mm:\n", t.mm(t))
print("要素積:\n", t * t)
print("NumPy へ変換:", t.numpy().tolist())
print("NumPy から変換:", torch.from_numpy(a).tolist())
print("GPU は使えるか:", torch.cuda.is_available())
