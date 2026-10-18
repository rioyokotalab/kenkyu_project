"""data_utils.py — データとトークナイザ（03 で解説、04 以降が import して使う）。

Python はファイル名が数字で始まるモジュールを import できないので、共有コードはここに置く。
"""
from __future__ import annotations

import os
import urllib.request

import torch

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "input.txt"
JA_URL = "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip"
JA_PATH = "japanese.txt"


def _download_japanese(path: str) -> None:
    """青空文庫「吾輩は猫である」を取得し、ルビ・注記を除いた UTF-8 テキストにする。"""
    import io
    import re
    import zipfile

    print(f"{path} が無いので青空文庫（夏目漱石「吾輩は猫である」）をダウンロードします")
    req = urllib.request.Request(JA_URL, headers={"User-Agent": "kenkyu-project"})
    data = urllib.request.urlopen(req).read()
    zf = zipfile.ZipFile(io.BytesIO(data))
    raw = zf.read(zf.namelist()[0]).decode("cp932")
    body = re.split(r"-{20,}\r?\n", raw)[-1]          # 冒頭の注記ブロックを捨てる
    body = body.split("底本：")[0]                       # 末尾の書誌情報を捨てる
    body = re.sub(r"《[^》]*》|［＃[^］]*］|｜", "", body)  # ルビ・注記記号を除く
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def load_text(path: str = DATA_PATH) -> str:
    if not os.path.exists(path):
        if path == JA_PATH:
            _download_japanese(path)
        else:
            print(f"{path} が無いのでダウンロードします")
            urllib.request.urlretrieve(DATA_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    """文字 <-> 整数。最も単純なトークナイザ（語彙 = 出現する文字の集合）。"""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class BPETokenizer:
    """BPE（tiktoken）。実際の LLM が使う方式。12 で導入する。"""

    def __init__(self, name: str = "cl100k_base"):
        import tiktoken
        self.enc = tiktoken.get_encoding(name)
        self.vocab_size = self.enc.n_vocab

    def encode(self, s: str) -> list[int]:
        return self.enc.encode(s)

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)


def load_dataset(path: str = DATA_PATH, tokenizer: str = "char"):
    """(train_data, val_data, tok) を返す。tokenizer は "char" か "bpe"。"""
    text = load_text(path)
    tok = CharTokenizer(text) if tokenizer == "char" else BPETokenizer()
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], tok


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str = "cpu"):
    """ランダムな位置から (x, y) を切り出す。y は x を 1 つ左にずらした「次のトークン」。"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
