"""12_bpe.py — BPE トークナイザ（tiktoken）と日本語テキスト。

学ぶこと:
- 文字単位（char）ではなく、よく出る部分文字列をまとめる BPE（Byte Pair Encoding）。
  実際の LLM（GPT・Swallow など）はこの方式。
- char と BPE で「系列長」と「語彙数」がどうトレードオフになるか。
- 日本語は byte 単位の BPE で扱える（cl100k_base は多言語で効率が良い）。

準備:  uv pip install tiktoken
使い方:  python 12_bpe.py

GPT の学習は、トークナイザを BPE に差し替えるだけで同じコードで動く
（vocab_size を enc.n_vocab にするだけ）。
"""
from __future__ import annotations

import tiktoken

from data_utils import load_text

JP_SAMPLE = (
    "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"
    "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"
)


def main():
    enc = tiktoken.get_encoding("cl100k_base")   # 多言語に強い BPE
    print(f"BPE 語彙数: {enc.n_vocab:,}")

    # (1) 英語（Shakespeare）で char と BPE を比較
    text = load_text()[:2000]
    char_vocab = len(set(text))
    bpe_ids = enc.encode(text)
    print("\n[英語 2000 文字]")
    print(f"  char: 系列長 {len(text):,}  語彙 {char_vocab}")
    print(f"  BPE : 系列長 {len(bpe_ids):,}  語彙 {enc.n_vocab:,}")
    print(f"  -> BPE は系列が {len(text) / len(bpe_ids):.1f} 倍短い（＝同じ文脈をより少ないトークンで表せる）")

    # (2) 日本語
    jp_ids = enc.encode(JP_SAMPLE)
    print("\n[日本語サンプル]")
    print(f"  原文 {len(JP_SAMPLE)} 文字 -> BPE {len(jp_ids)} トークン")
    print("  復号:", enc.decode(jp_ids[:20]), "...")

    print("\nまとめ: BPE は系列を短くできるが語彙が巨大（埋め込み行列が大きい）。")
    print("GPT の学習は、data_utils の CharTokenizer を tiktoken に差し替え、")
    print("GPTConfig.vocab_size = enc.n_vocab にするだけで同じコードで動く。")


if __name__ == "__main__":
    main()
