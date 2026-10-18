"""data_utils.py — データとトークナイザの共有ユーティリティ。

03_data.py がこの中身を解説する。04 以降のスクリプトは
`from data_utils import CharTokenizer, get_batch, load_text` で読み込んで使う。
（Python はファイル名が数字で始まるモジュールを import できないため、共有コードはここに置く。）
"""
from __future__ import annotations

import os
import urllib.request

import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "input.txt"


def load_text(path: str = DATA_PATH) -> str:
    if not os.path.exists(path):
        print(f"{path} が無いのでダウンロードします: {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    """文字 <-> 整数 の最も単純なトークナイザ。"""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_dataset(path: str = DATA_PATH):
    """テキストを読み込み、(train_data, val_data, tokenizer) を返す。"""
    text = load_text(path)
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], tok


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu"):
    """data（整数列）から (x, y) をランダムに切り出す。y は x を 1 つずらした「次のトークン」。"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
